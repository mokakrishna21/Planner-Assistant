import json
import re
import llm
import tools as tool_runner
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Helpers ─────────────────────────────────────────────────────────────

def safe_json_loads(s: str):
    s = re.sub(r"[\x00-\x1F\x7F]", "", s)
    return json.loads(s)

def extract_json(s: str):
    match = re.search(r'(\[.*\]|\{.*\})', s, re.DOTALL)
    if match:
        return match.group(0)
    return s

# ── Prompts ─────────────────────────────────────────────────────────────

PLANNER_PROMPT = """You are a data analyst planning how to answer a question about a spreadsheet.

Return a JSON list of steps.

Each step MUST include:
- "type": one of ["query", "plot", "reason"]
- "goal": short description
- "code": required for query/plot

Rules:
- Use ONLY schema columns
- Assign output to `result` or `fig`
- Use SINGLE quotes inside code
- NO line breaks in strings
- STRICT VALID JSON ONLY
- No markdown, no explanation

Schema:
{schema}
"""

ANSWER_PROMPT = """Answer clearly.

Question: {question}
Context:
{context}

- Be concise
- Use numbers if available
- If insight exists:
**Insight:** ...

If chart exists → say "Chart shown above."
"""

# ── Agent ───────────────────────────────────────────────────────────────

def run(question: str, df, history: list) -> dict:
    schema = tool_runner.get_schema(df)
    context = {}
    steps_log = []
    fig = None

    namespace = {
        "df": df.copy(),
        "pd": pd,
        "px": px,
        "go": go
    }

    # ── PLAN (robust) ───────────────────────────────────────────────────
    plan = None

    for attempt in range(2):
        try:
            raw = llm.call(
                PLANNER_PROMPT.format(schema=schema),
                f"Question: {question}"
            )
            raw = extract_json(raw.strip())
            parsed = safe_json_loads(raw)

            if not isinstance(parsed, list):
                parsed = [parsed]

            # ✅ FIX: ensure valid structure
            valid_plan = []
            for step in parsed:
                if isinstance(step, dict):
                    if "type" not in step:
                        step["type"] = "query"  # fallback
                    valid_plan.append(step)

            if not valid_plan:
                raise ValueError("Empty or invalid plan")

            plan = valid_plan
            break

        except Exception as e:
            if attempt == 1:
                return {
                    "answer": f"Planning failed: {e}",
                    "fig": None,
                    "steps": []
                }

    # ── ACT ─────────────────────────────────────────────────────────────
    for i, step in enumerate(plan):
        step_key = f"step_{i}"

        # ✅ defensive check
        if not isinstance(step, dict) or "type" not in step:
            steps_log.append({
                "step": step_key,
                "type": "error",
                "goal": "Invalid step",
                "ok": False
            })
            continue

        step_type = step["type"]
        goal = step.get("goal", "")

        if step_type == "query":
            res = tool_runner.run_query(df, step.get("code", ""), namespace)

            if res["ok"]:
                context[step_key] = str(res["result"])[:500]
                steps_log.append({"step": step_key, "type": "query", "goal": goal, "ok": True})
            else:
                context[step_key] = f"ERROR: {res['error']}"
                steps_log.append({"step": step_key, "type": "query", "goal": goal, "ok": False})

        elif step_type == "plot":
            res = tool_runner.run_plot(df, step.get("code", ""), namespace)

            if res["ok"]:
                fig = res["fig"]
                context[step_key] = "Plot generated"
                steps_log.append({"step": step_key, "type": "plot", "goal": goal, "ok": True})
            else:
                context[step_key] = f"ERROR: {res['error']}"
                steps_log.append({"step": step_key, "type": "plot", "goal": goal, "ok": False})

        elif step_type == "reason":
            relevant = {
                k: context[k]
                for k in step.get("context_keys", [])
                if k in context
            }
            context[step_key] = str(relevant)
            steps_log.append({"step": step_key, "type": "reason", "goal": goal})

    # ── ANSWER ─────────────────────────────────────────────────────────
    context_str = "\n".join([f"[{k}] {v}" for k, v in context.items()])

    answer = llm.call(
        ANSWER_PROMPT.format(question=question, context=context_str),
        "Answer."
    )

    return {
        "answer": answer,
        "fig": fig,
        "steps": steps_log
    }