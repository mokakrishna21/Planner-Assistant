import json
import re
import llm
import tools as tool_runner
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def safe_json_loads(s):
    s = re.sub(r"[\x00-\x1F\x7F]", "", s)
    return json.loads(s)


def extract_json(s):
    match = re.search(r'(\[.*\]|\{.*\})', s, re.DOTALL)
    return match.group(0) if match else s


# ── PROMPTS ─────────────────────────────────────────────────────────────

PLANNER_PROMPT = """You are a data analyst working on an UNKNOWN dataset.

Schema:
{schema}

Instructions:
- NEVER assume column names
- ALWAYS use column_names + sample_rows
- If filtering values, find correct column from sample_rows
- Use ONLY available columns

Each step:
- {"type": "query", "goal": "...", "code": "..."}
- {"type": "plot", "goal": "...", "code": "..."}
- {"type": "reason", "goal": "...", "context_keys": [...]}

Rules:
- Use df['exact_column']
- Assign result to `result`
- SINGLE quotes only
- STRICT JSON ONLY

Output ONLY JSON.
"""

ANSWER_PROMPT = """You MUST answer using ONLY computed results.

Question: {question}

Results:
{context}

Rules:
- DO NOT use external knowledge
- DO NOT say "I don’t have access"
- If ERROR present → explain briefly
- Otherwise answer using results

Be concise.
"""


# ── AGENT ───────────────────────────────────────────────────────────────

def run(question, df, history):
    schema = tool_runner.get_schema(df)
    context = {}
    steps_log = []
    fig = None

    namespace = {
        "df": df.copy(),
        "pd": pd,
        "px": px,
        "go": go,
    }

    # ── PLAN ───────────────────────────────────────────────────────────
    for attempt in range(2):
        try:
            raw = llm.call(
                PLANNER_PROMPT.format(schema=schema),
                f"Question: {question}"
            )
            raw = extract_json(raw)
            plan = safe_json_loads(raw)

            if not isinstance(plan, list):
                plan = [plan]

            # ensure structure
            clean_plan = []
            for step in plan:
                if isinstance(step, dict):
                    if "type" not in step:
                        step["type"] = "query"
                    clean_plan.append(step)

            if not clean_plan:
                raise ValueError("Invalid plan")

            plan = clean_plan
            break

        except Exception as e:
            if attempt == 1:
                return {"answer": f"Planning failed: {e}", "fig": None, "steps": []}

    # ── ACT ────────────────────────────────────────────────────────────
    for i, step in enumerate(plan):
        step_key = f"step_{i}"
        step_type = step.get("type")
        goal = step.get("goal", "")

        if step_type == "query":
            res = tool_runner.run_query(df, step.get("code", ""), namespace)

            if not res["ok"]:
                return {
                    "answer": f"Query failed: {res['error']}",
                    "fig": None,
                    "steps": steps_log,
                }

            context[step_key] = str(res["result"])[:500]
            steps_log.append({"step": step_key, "type": "query", "goal": goal, "ok": True})

        elif step_type == "plot":
            res = tool_runner.run_plot(df, step.get("code", ""), namespace)

            if res["ok"]:
                fig = res["fig"]
                context[step_key] = "Plot generated"
                steps_log.append({"step": step_key, "type": "plot", "goal": goal, "ok": True})

        elif step_type == "reason":
            relevant = {k: context[k] for k in step.get("context_keys", []) if k in context}
            context[step_key] = str(relevant)
            steps_log.append({"step": step_key, "type": "reason", "goal": goal})

    # ── ANSWER ─────────────────────────────────────────────────────────
    context_str = "\n".join([f"[{k}] {v}" for k, v in context.items()])

    answer = llm.call(
        ANSWER_PROMPT.format(question=question, context=context_str),
        "Answer."
    )

    return {"answer": answer, "fig": fig, "steps": steps_log}