import json
import re
import llm
import tools as tool_runner
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def safe_json_loads(s):
    """Parse JSON with aggressive cleaning for common LLM errors."""
    if not s or not isinstance(s, str):
        raise ValueError("Empty or non-string input")
    
    s = re.sub(r"[\x00-\x1F\x7F]", "", s)
    s = re.sub(r",\s*(\]|\})", r"\1", s)
    s = re.sub(r"(?<!\\)'", '"', s)
    
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        s = re.sub(r'^```json\s*', '', s)
        s = re.sub(r'^```\s*', '', s)
        s = re.sub(r'\s*```$', '', s)
        return json.loads(s)


def extract_json(s):
    """Extract JSON array or object from text."""
    if not s:
        return "[]"
    
    s = s.strip()
    if s.startswith('[') and s.endswith(']'):
        return s
    if s.startswith('{') and s.endswith('}'):
        return s
    
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


PLANNER_PROMPT = """You are a data analyst. Analyze the schema and produce a JSON array of steps.

Schema:
{schema}

Instructions:
- Use ONLY column names from schema above
- Use df['column_name'] syntax exactly
- For text matching, check exact values in sample_rows first

Step types:
- "query": Data manipulation (requires "code" field assigning to `result`)
- "plot": Visualization (requires "code" field assigning to `fig`)
- "reason": Analysis (requires "context_keys" array referencing previous steps)

Valid JSON examples:
[
  {{"type": "query", "goal": "Count delayed rows", "code": "result = df[df['status'] == 'delayed'].shape[0]"}},
  {{"type": "plot", "goal": "Bar chart of sales", "code": "fig = px.bar(df.groupby('category')['sales'].sum().reset_index(), x='category', y='sales')"}},
  {{"type": "reason", "goal": "Analyze gap", "context_keys": ["step_0", "step_1"]}}
]

Rules:
- Output ONLY the JSON array, no markdown, no explanation
- No trailing commas
- Use SINGLE quotes for Python strings inside "code" fields
- DOUBLE quotes for JSON keys/values

User question: {question}
History context: {history}
"""

ANSWER_PROMPT = """Answer using ONLY the computed results below. Do not use external knowledge.

Question: {question}

Computed Results:
{context}

Guidelines:
- If results contain error messages, explain what went wrong
- If DataFrame shown, summarize key findings
- If boolean result, explain the implication
- Keep answer under 3 sentences unless detailed analysis required
"""


def sanitize_code(code):
    """Basic safety check for generated code."""
    dangerous = ['import os', 'import sys', '__import__', 'subprocess', 'open(', 'eval(', 'exec(', 'compile(']
    code_lower = code.lower()
    for d in dangerous:
        if d in code_lower:
            raise ValueError(f"Potentially unsafe code detected: {d}")
    return code


def run(question, df, history=None):
    if df is None or df.empty:
        return {"answer": "No data loaded. Please upload a file first.", "fig": None, "steps": []}
    
    history = history or []
    schema = tool_runner.get_schema(df)
    context = {}
    steps_log = []
    fig = None
    
    history_str = ""
    if history:
        recent = history[-3:]
        history_str = "\n".join([f"{'User' if m.get('role')=='user' else 'Assistant'}: {str(m.get('content',''))[:100]}" 
                                for m in recent])

    base_namespace = {
        "df": df.copy(),
        "pd": pd,
        "px": px,
        "go": go,
        "np": np,
    }

    # ── PLAN ───────────────────────────────────────────────────────────
    plan = None
    for attempt in range(3):
        try:
            prompt = PLANNER_PROMPT.format(
                schema=schema, 
                question=question,
                history=history_str if history_str else "None"
            )
            raw = llm.call(prompt, "Generate analysis plan.")
            raw = extract_json(raw)
            plan = safe_json_loads(raw)

            if not isinstance(plan, list):
                plan = [plan] if isinstance(plan, dict) else []
            
            clean_plan = []
            for step in plan:
                if not isinstance(step, dict):
                    continue
                    
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
                
                if step["type"] in ["query", "plot"] and "code" not in step:
                    continue
                if step["type"] == "reason" and "context_keys" not in step:
                    step["context_keys"] = []
                    
                clean_plan.append(step)
            
            if clean_plan:
                plan = clean_plan
                break
            else:
                raise ValueError("No valid steps in plan")

        except Exception as e:
            if attempt == 2:
                return {
                    "answer": f"Unable to create analysis plan: {str(e)}. Try rephrasing your question.", 
                    "fig": None, 
                    "steps": [{"type": "error", "step": "planning", "error": str(e), "ok": False}]
                }

    # ── ACT ────────────────────────────────────────────────────────────
    for i, step in enumerate(plan):
        step_key = f"step_{i}"
        step_type = step.get("type", "unknown")
        goal = step.get("goal", f"Step {i}")
        
        step_namespace = base_namespace.copy()
        step_namespace.update(context)
        
        try:
            if step_type == "query":
                code = step.get("code", "")
                sanitize_code(code)
                res = tool_runner.run_query(df, code, step_namespace)
                
                if not res["ok"]:
                    steps_log.append({
                        "step": step_key, 
                        "type": "query", 
                        "goal": goal, 
                        "ok": False, 
                        "error": res["error"]
                    })
                    context[step_key] = f"Error: {res['error']}"
                else:
                    context[step_key] = format_result_for_context(res["result"])
                    steps_log.append({
                        "step": step_key, 
                        "type": "query", 
                        "goal": goal, 
                        "ok": True
                    })

            elif step_type == "plot":
                code = step.get("code", "")
                sanitize_code(code)
                res = tool_runner.run_plot(df, code, step_namespace)
                
                if res["ok"]:
                    fig = res["fig"]
                    context[step_key] = "Plot generated successfully"
                    steps_log.append({
                        "step": step_key, 
                        "type": "plot", 
                        "goal": goal, 
                        "ok": True
                    })
                else:
                    context[step_key] = f"Plot error: {res['error']}"
                    steps_log.append({
                        "step": step_key, 
                        "type": "plot", 
                        "goal": goal, 
                        "ok": False, 
                        "error": res["error"]
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
                    "error": f"Unknown step type: {step_type}"
                })

        except Exception as e:
            steps_log.append({
                "step": step_key, 
                "type": step_type if step_type else "error", 
                "goal": goal, 
                "ok": False, 
                "error": str(e)
            })
            context[step_key] = f"Execution error: {str(e)}"

    # ── ANSWER ─────────────────────────────────────────────────────────
    if not context:
        return {
            "answer": "Could not analyze the data. The question might be outside the scope of available columns.", 
            "fig": fig, 
            "steps": steps_log
        }
    
    context_str = "\n\n".join([f"[{k}]:\n{v}" for k, v in context.items()])
    
    try:
        answer = llm.call(
            ANSWER_PROMPT.format(question=question, context=context_str),
            "Generate final answer."
        )
    except Exception as e:
        answer = f"Analysis completed but summarization failed: {str(e)}"

    return {"answer": answer, "fig": fig, "steps": steps_log}