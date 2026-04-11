import json
import re
import ast
import llm
import tools as tool_runner
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


# ── JSON parsing helpers ──────────────────────────────────────────────────────

def extract_json(s):
    """Extract first complete JSON array or object from raw LLM text."""
    if not s:
        return "[]"
    s = s.strip()
    if (s.startswith('[') and s.endswith(']')) or (s.startswith('{') and s.endswith('}')):
        return s
    for open_c, close_c in [('[', ']'), ('{', '}')]:
        start = s.find(open_c)
        if start == -1:
            continue
        count = 0
        for i in range(start, len(s)):
            if s[i] == open_c:
                count += 1
            elif s[i] == close_c:
                count -= 1
                if count == 0:
                    return s[start:i + 1]
    match = re.search(r'(\[[\s\S]*\]|\{[\s\S]*\})', s)
    return match.group(0) if match else s


def safe_json_loads(s):
    """Parse JSON with multiple fallback strategies."""
    if not s or not isinstance(s, str):
        raise ValueError("Empty or non-string input")

    s = re.sub(r"[\x00-\x1F\x7F]", "", s)
    original = s.strip()

    # 1. Direct parse
    try:
        return json.loads(original)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fences
    cleaned = re.sub(r'^```json\s*', '', original)
    cleaned = re.sub(r'^```\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Fix trailing commas
    cleaned = re.sub(r",\s*(\]|\})", r"\1", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 4. Python literal eval (handles single-quoted JSON)
    if cleaned.strip().startswith(('{', '[')):
        try:
            return ast.literal_eval(cleaned)
        except (ValueError, SyntaxError):
            pass

    raise json.JSONDecodeError(f"Could not parse: {original[:100]}", original, 0)


# ── Result formatting ─────────────────────────────────────────────────────────

def format_result_for_context(result):
    """Convert any exec() result to a readable string for LLM context."""
    if isinstance(result, bool):
        return "Yes" if result else "No"
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return "Empty DataFrame"
        preview = result.head(10).to_string(index=False)
        return f"DataFrame ({len(result)} rows x {len(result.columns)} cols):\n{preview}"
    if isinstance(result, pd.Series):
        return f"Series ({len(result)} values):\n{result.head(10).to_string()}"
    if isinstance(result, (int, np.integer)):
        return f"{int(result):,}"
    if isinstance(result, (float, np.floating)):
        return "NaN" if np.isnan(result) else f"{float(result):,.4f}"
    if isinstance(result, (list, tuple)):
        if not result:
            return "Empty list"
        preview = str(list(result)[:10])
        return preview + (f" ... ({len(result)} total)" if len(result) > 10 else "")
    if result is None:
        return "None"
    text = str(result)
    return text[:500] + "..." if len(text) > 500 else text


# ── Safety check ──────────────────────────────────────────────────────────────

DANGEROUS = ['import os', 'import sys', '__import__', 'subprocess', 'eval(', 'compile(']

def sanitize_code(code):
    if not code:
        raise ValueError("Empty code")
    code_lower = code.lower()
    for d in DANGEROUS:
        if d in code_lower:
            raise ValueError(f"Unsafe code pattern: {d}")
    return code


# ── Prompts ───────────────────────────────────────────────────────────────────

PLANNER_PROMPT = """You are a data analyst. Produce a JSON array of steps to answer the question.

Schema:
{schema}

Step types:
- {{"type": "query", "goal": "...", "code": "pandas code that assigns to `result`"}}
- {{"type": "plot",  "goal": "...", "code": "plotly code that assigns to `fig`"}}
- {{"type": "reason","goal": "...", "context_keys": ["step_0", ...]}}

Rules:
- Use ONLY columns that exist in the schema above.
- Use df['column'] syntax with single quotes inside code strings.
- Intermediate variables from earlier steps are available in later steps.
- query steps MUST assign to `result`. plot steps MUST assign to `fig`.
- Output ONLY the JSON array. No markdown, no explanation, no trailing commas.

Conversation history: {history}
Question: {question}
"""

ANSWER_PROMPT = """You are a data analyst. Answer the question using ONLY the computed results below.

Question: {question}

Computed Results:
{context}

Guidelines:
- Be specific and reference actual numbers from the results.
- If errors occurred in some steps, note them briefly and answer with what succeeded.
- Keep it concise (3-5 sentences).
- If something noteworthy stands out in the data, add a final line: **Insight:** <one sentence>
"""


# ── Main agent loop ───────────────────────────────────────────────────────────

def run(question, df, history=None):
    if df is None or df.empty:
        return {"answer": "No data loaded.", "fig": None, "steps": []}

    history = history or []
    schema = tool_runner.get_schema(df)
    context = {}
    steps_log = []
    fig = None

    history_str = " | ".join([
        f"{'Q' if m.get('role') == 'user' else 'A'}: {str(m.get('content', ''))[:80]}"
        for m in history[-4:]
    ]) if history else "None"

    # Persistent execution namespace — variables created in step N are
    # available to step N+1 (e.g. a filtered df used in a follow-up plot)
    exec_ns = {"df": df.copy(), "pd": pd, "px": px, "go": go, "np": np}

    # ── 1. PLAN ───────────────────────────────────────────────────────────
    plan = None
    raw_response = ""

    for attempt in range(3):
        try:
            raw_response = llm.call(
                PLANNER_PROMPT.format(schema=schema, question=question, history=history_str),
                "Generate the JSON plan."
            )
            if not raw_response:
                raise ValueError("Empty LLM response")

            plan = safe_json_loads(extract_json(raw_response))

            if not isinstance(plan, list):
                plan = [plan] if isinstance(plan, dict) else []

            # Validate and clean each step
            clean = []
            for step in plan:
                if not isinstance(step, dict):
                    continue
                # Infer type if missing
                if "type" not in step:
                    code = step.get("code", "")
                    if "context_keys" in step:
                        step["type"] = "reason"
                    elif any(k in code for k in ["fig", "px.", "go.", "scatter", "bar(", "line("]):
                        step["type"] = "plot"
                    elif "code" in step:
                        step["type"] = "query"
                    else:
                        continue
                step_type = step["type"]
                if step_type in ("query", "plot") and "code" not in step:
                    continue
                if step_type == "reason" and "context_keys" not in step:
                    step["context_keys"] = []
                clean.append(step)

            if clean:
                plan = clean
                break
            raise ValueError("No valid steps after cleaning")

        except Exception as e:
            if attempt == 2:
                debug = raw_response[:300] if raw_response else "no response"
                return {
                    "answer": f"Could not plan steps for this question. Try rephrasing.\n\nDebug: {str(e)}",
                    "fig": None,
                    "steps": [{"type": "error", "goal": "planning", "ok": False, "error": str(e)}]
                }

    # ── 2. ACT ────────────────────────────────────────────────────────────
    for i, step in enumerate(plan):
        step_key = f"step_{i}"
        step_type = step.get("type", "unknown")
        goal = step.get("goal", f"Step {i}")

        try:
            if step_type == "query":
                sanitize_code(step["code"])
                exec_ns.pop("result", None)
                exec(step["code"], exec_ns)
                result = exec_ns.get("result")
                if result is None:
                    raise ValueError("Code did not assign to `result`")
                context[step_key] = format_result_for_context(result)
                steps_log.append({"step": step_key, "type": "query", "goal": goal, "ok": True})

            elif step_type == "plot":
                sanitize_code(step["code"])
                exec_ns.pop("fig", None)
                exec(step["code"], exec_ns)
                plot_fig = exec_ns.get("fig")
                if plot_fig is None:
                    raise ValueError("Code did not assign to `fig`")
                if not hasattr(plot_fig, 'update_layout'):
                    raise ValueError("Result is not a Plotly figure")
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
                context[step_key] = "Plot generated successfully."
                steps_log.append({"step": step_key, "type": "plot", "goal": goal, "ok": True})

            elif step_type == "reason":
                keys = step.get("context_keys", [])
                relevant = {k: context.get(k, "N/A") for k in keys}
                context[step_key] = str(relevant)[:800]
                steps_log.append({"step": step_key, "type": "reason", "goal": goal, "ok": True})

            else:
                steps_log.append({"step": step_key, "type": step_type, "goal": goal,
                                   "ok": False, "error": f"Unknown type: {step_type}"})

        except Exception as e:
            steps_log.append({"step": step_key, "type": step_type, "goal": goal,
                               "ok": False, "error": str(e)})
            context[step_key] = f"Error in this step: {str(e)}"

    # ── 3. ANSWER ─────────────────────────────────────────────────────────
    if not context:
        return {"answer": "Could not analyze the data.", "fig": fig, "steps": steps_log}

    context_str = "\n\n".join([f"[{k}]:\n{v}" for k, v in context.items()])

    try:
        answer = llm.call(
            ANSWER_PROMPT.format(question=question, context=context_str),
            "Generate the final answer."
        )
    except Exception as e:
        answer = f"Analysis completed but summarization failed: {e}"

    return {"answer": answer, "fig": fig, "steps": steps_log}