import json
import re
import ast
import llm
import tools as tool_runner
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def safe_json_loads(s):
    """Parse JSON with multiple fallback strategies."""
    if not s or not isinstance(s, str):
        raise ValueError("Empty or non-string input")
    
    s = re.sub(r"[\x00-\x1F\x7F]", "", s)
    original = s.strip()
    
    # Strategy 1: Direct parse
    try:
        return json.loads(original)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Remove markdown
    s_clean = re.sub(r'^```json\s*', '', original)
    s_clean = re.sub(r'^```\s*', '', s_clean)
    s_clean = re.sub(r'\s*```$', '', s_clean)
    if s_clean != original:
        try:
            return json.loads(s_clean)
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Fix trailing commas
    s_clean = re.sub(r",\s*(\]|\})", r"\1", s_clean)
    try:
        return json.loads(s_clean)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Python literal eval (handles single quotes)
    if s_clean.strip().startswith(('{', '[')):
        try:
            return ast.literal_eval(s_clean)
        except (ValueError, SyntaxError):
            pass
    
    raise json.JSONDecodeError(f"Could not parse JSON: {original[:100]}...", original, 0)


def extract_json(s):
    """Extract JSON array or object from text."""
    if not s:
        return "[]"
    
    s = s.strip()
    
    # Direct match
    if (s.startswith('[') and s.endswith(']')) or (s.startswith('{') and s.endswith('}')):
        return s
    
    # Find balanced brackets
    for char, close_char in [('[', ']'), ('{', '}')]:
        start = s.find(char)
        if start == -1:
            continue
        
        count = 0
        for i in range(start, len(s)):
            if s[i] == char:
                count += 1
            elif s[i] == close_char:
                count -= 1
                if count == 0:
                    return s[start:i+1]
    
    # Fallback regex
    match = re.search(r'(\[[\s\S]*\]|\{[\s\S]*\})', s)
    if match:
        return match.group(0)
    
    return s


def format_result_for_context(result):
    """Convert query results to display-friendly strings."""
    if isinstance(result, bool):
        return "Yes" if result else "No"
    elif isinstance(result, (int, np.integer)):
        return f"{int(result):,}"
    elif isinstance(result, (float, np.floating)):
        if np.isnan(result):
            return "NaN"
        return f"{float(result):,.4f}"
    elif isinstance(result, pd.DataFrame):
        if result.empty:
            return "Empty DataFrame"
        preview = result.head(10).to_string(index=False)
        return f"DataFrame ({len(result)} rows):\n{preview}"
    elif isinstance(result, (list, tuple)):
        if not result:
            return "Empty list"
        preview = str(result[:10])
        if len(result) > 10:
            preview += f"... ({len(result)} total items)"
        return preview
    elif result is None:
        return "None"
    else:
        text = str(result)
        return text[:497] + "..." if len(text) > 500 else text


PLANNER_PROMPT = """You are a data analyst. Analyze the schema and produce a valid JSON array of steps.

Schema:
{schema}

Instructions:
- All data is in the single DataFrame named `df`
- Use df['column_name'] syntax exactly (single quotes inside code)
- Intermediate variables created in one step are available in subsequent steps
- Each query step MUST assign its final output to a variable named `result`
- Each plot step MUST assign its final output to a variable named `fig`

Step types:
- "query": Data manipulation (requires "code" field assigning to `result`)
- "plot": Visualization (requires "code" field assigning to `fig`)
- "reason": Analysis explanation (requires "context_keys" array)

Example output:
[
  {{"type": "query", "goal": "Filter WIPRO data", "code": "wipro_df = df[df['Symbol'] == 'WIPRO']; result = wipro_df"}},
  {{"type": "query", "goal": "Calculate average volume", "code": "result = wipro_df['Volume'].mean()"}},
  {{"type": "plot", "goal": "Volume trend", "code": "fig = px.line(wipro_df, x='Date', y='Volume')"}}
]

Rules:
- Output ONLY the JSON array, no markdown, no explanation
- Use double quotes " for JSON keys and string values
- Use single quotes for Python strings inside the code field
- Ensure each step assigns to `result` (for queries) or `fig` (for plots)
- No trailing commas

User question: {question}
History: {history}
"""

ANSWER_PROMPT = """Answer using ONLY the computed results below.

Question: {question}

Computed Results:
{context}

Guidelines:
- Answer based on the computed results above.
- Be specific: reference actual numbers and column names.
- Be concise (3-5 sentences).
- If errors occurred in some steps, answer with what succeeded.
- If something noteworthy stands out, add a final line: **Insight:** <one sentence>
"""



CODE_FIX_PROMPT = """The following Python/pandas code failed. Fix it and return ONLY the corrected code, nothing else.

Schema:
{schema}

Goal: {goal}
Original code:
{code}

Error:
{error}

Rules:
- Fix the error. Return ONLY the corrected Python code, no explanation, no markdown fences.
- The code must assign to `result` (for queries) or `fig` (for plots).
- Use only columns that exist in the schema.
"""

def sanitize_code(code):
    """Basic safety check for generated code."""
    if not code:
        raise ValueError("Empty code")
    dangerous = ['import os', 'import sys', '__import__', 'subprocess', 'open(', 'eval(', 'exec(', 'compile(']
    code_lower = code.lower()
    for d in dangerous:
        if d in code_lower:
            raise ValueError(f"Potentially unsafe code: {d}")
    return code


def run(question, df, history=None):
    if df is None or df.empty:
        return {"answer": "No data loaded.", "fig": None, "steps": []}
    
    history = history or []
    schema = tool_runner.get_schema(df)
    context = {}
    steps_log = []
    fig = None
    
    history_str = ""
    if history:
        recent = history[-3:]
        history_str = " | ".join([f"{'Q' if m.get('role')=='user' else 'A'}: {str(m.get('content',''))[:80]}" 
                                for m in recent])

    # CRITICAL: Persistent namespace across all steps
    exec_namespace = {
        "df": df.copy(),
        "pd": pd,
        "px": px,
        "go": go,
        "np": np,
    }

    # ── PLAN ───────────────────────────────────────────────────────────
    plan = None
    raw_response = ""
    
    for attempt in range(3):
        try:
            prompt = PLANNER_PROMPT.format(
                schema=schema, 
                question=question,
                history=history_str if history_str else "None"
            )
            raw_response = llm.call(prompt, "Generate plan.")
            
            if not raw_response:
                raise ValueError("Empty LLM response")
            
            extracted = extract_json(raw_response)
            plan = safe_json_loads(extracted)

            if not isinstance(plan, list):
                plan = [plan] if isinstance(plan, dict) else []
            
            clean_plan = []
            for step in plan:
                if not isinstance(step, dict):
                    continue
                    
                # Infer type if missing
                if "type" not in step:
                    code = step.get("code", "")
                    if "context_keys" in step:
                        step["type"] = "reason"
                    elif any(k in code for k in ["fig", "px.", "go.", "plot", "chart", "scatter", "bar(", "line("]):
                        step["type"] = "plot"
                    elif "code" in step:
                        step["type"] = "query"
                    else:
                        step["type"] = "unknown"
                
                # Validate
                if step.get("type") in ["query", "plot"] and "code" not in step:
                    continue
                if step.get("type") == "reason" and "context_keys" not in step:
                    step["context_keys"] = []
                    
                clean_plan.append(step)
            
            if clean_plan:
                plan = clean_plan
                break
            else:
                raise ValueError("No valid steps")

        except Exception as e:
            if attempt == 2:
                debug = raw_response[:200] + "..." if len(raw_response) > 200 else raw_response
                return {
                    "answer": f"Planning failed: {str(e)}. Try rephrasing.", 
                    "fig": None, 
                    "steps": [{"type": "error", "step": "planning", "error": str(e), "ok": False, "debug": debug}]
                }

    # ── ACT (Persistent Namespace) ─────────────────────────────────────
    for i, step in enumerate(plan):
        step_key = f"step_{i}"
        step_type = step.get("type", "unknown")
        goal = step.get("goal", f"Step {i}")
        
        try:
            if step_type == "query":
                code = step.get("code", "")
                last_error = None

                for attempt in range(2):  # try original, then LLM-fixed version
                    try:
                        sanitize_code(code)
                        exec_namespace.pop("result", None)
                        exec(code, exec_namespace)
                        result = exec_namespace.get("result")
                        if result is None:
                            raise ValueError("Code did not assign to 'result' variable")
                        context[step_key] = format_result_for_context(result)
                        steps_log.append({"step": step_key, "type": "query", "goal": goal, "ok": True})
                        last_error = None
                        break  # success
                    except Exception as e:
                        last_error = str(e)
                        if attempt == 0:
                            # Ask LLM to fix the broken code
                            try:
                                fixed = llm.call(
                                    CODE_FIX_PROMPT.format(
                                        schema=schema, goal=goal,
                                        code=code, error=last_error
                                    ),
                                    "Fix the code."
                                )
                                # Strip any accidental fences
                                fixed = fixed.strip()
                                if fixed.startswith("```"):
                                    fixed = fixed.split("```")[1]
                                    if fixed.startswith("python"):
                                        fixed = fixed[6:]
                                code = fixed.strip()
                            except Exception:
                                pass  # if fix call fails, fall through to error log

                if last_error:
                    raise RuntimeError(last_error)

            elif step_type == "plot":
                code = step.get("code", "")
                sanitize_code(code)
                
                exec_namespace.pop("fig", None)
                exec(code, exec_namespace)
                plot_fig = exec_namespace.get("fig")
                
                if plot_fig is None:
                    raise ValueError("Code did not assign to 'fig' variable")
                
                # Apply styling
                plot_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e6edf3"),
                    title_font=dict(color="#e6edf3"),
                    legend_font=dict(color="#8b949e"),
                    xaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d"),
                    yaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d"),
                )
                
                fig = plot_fig
                context[step_key] = "Plot generated"
                steps_log.append({
                    "step": step_key, 
                    "type": "plot", 
                    "goal": goal, 
                    "ok": True
                })

            elif step_type == "reason":
                keys = step.get("context_keys", [])
                relevant = {k: context.get(k, "N/A") for k in keys}
                context[step_key] = str(relevant)[:800]
                steps_log.append({
                    "step": step_key, 
                    "type": "reason", 
                    "goal": goal,
                    "ok": True
                })
            else:
                steps_log.append({
                    "step": step_key,
                    "type": "unknown",
                    "goal": goal,
                    "ok": False,
                    "error": f"Unknown type: {step_type}"
                })

        except Exception as e:
            steps_log.append({
                "step": step_key, 
                "type": step_type, 
                "goal": goal, 
                "ok": False, 
                "error": str(e)
            })
            context[step_key] = f"Error: {str(e)}"

    # ── ANSWER ─────────────────────────────────────────────────────────
    if not context:
        return {
            "answer": "Could not analyze the data.", 
            "fig": fig, 
            "steps": steps_log
        }
    
    context_str = "\n\n".join([f"[{k}]:\n{v}" for k, v in context.items()])
    
    try:
        answer = llm.call(
            ANSWER_PROMPT.format(question=question, context=context_str),
            "Generate answer."
        )
    except Exception as e:
        answer = f"Analysis done but summarization failed: {str(e)}"

    return {"answer": answer, "fig": fig, "steps": steps_log}