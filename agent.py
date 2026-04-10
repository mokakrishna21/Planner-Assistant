"""
agent.py — Lightweight ReAct (Plan → Act → Reflect → Answer) loop.

This is the core differentiator. Instead of a single LLM call, we:
  1. PLAN  — LLM decides what steps are needed
  2. ACT   — Execute each step (pandas query or plotly chart)
  3. REFLECT — LLM validates intermediate result, replans if wrong
  4. ANSWER  — LLM synthesizes final answer with insight

This handles complex multi-hop questions like "why is there a gap between
task A and task B" — the kind that single-prompt approaches fail at.
"""

import json
import llm
import tools as tool_runner

# ── Prompts ────────────────────────────────────────────────────────────────────

PLANNER_PROMPT = """You are a data analyst planning how to answer a question about a spreadsheet.

Given the schema and question, output a JSON plan — a list of steps.
Each step is one of:
  - {{"type": "query", "goal": "...", "code": "pandas code that assigns to `result`"}}
  - {{"type": "plot",  "goal": "...", "code": "plotly code that assigns to `fig`"}}
  - {{"type": "reason","goal": "...", "context_keys": ["step_0", "step_1", ...]}}

Rules:
- Use only columns that exist in the schema.
- For query steps, code must assign to a variable named `result`.
- For plot steps, code must assign a plotly figure to `fig`. Use px or go.
- Output ONLY valid JSON. No markdown, no explanation.
- Keep it minimal: 1-3 steps usually enough.

Schema:
{schema}
"""

VALIDATOR_PROMPT = """You are checking if an intermediate result makes sense for answering a question.

Question: {question}
Step goal: {goal}
Result: {result}

Reply with JSON: {{"valid": true/false, "reason": "..."}}
Output ONLY JSON.
"""

REPLAN_PROMPT = """The previous step failed or produced an unexpected result. Replan.

Original question: {question}
Schema: {schema}
Failed step: {step}
Error: {error}
Context so far: {context}

Output a new JSON plan (list of steps) to recover and answer the question.
Same format as before. Output ONLY valid JSON.
"""

ANSWER_PROMPT = """You are a data analyst answering a question about a spreadsheet.

Question: {question}
Schema: {schema}
Intermediate results from analysis steps:
{context}

Instructions:
- Answer clearly and directly.
- If numbers or data back your answer, reference them.
- If something interesting stands out, end with "**Insight:** ..." on a new line.
- If a plot was generated, just say "Chart shown above." — don't describe it.
- Be concise. 3-5 sentences max unless the answer genuinely requires more.
"""

# ── Main agent loop ─────────────────────────────────────────────────────────────

def run(question: str, df, history: list) -> dict:
    """
    Run the full ReAct loop.
    Returns: {"answer": str, "fig": plotly_fig|None, "steps": list, "error": str|None}
    """
    schema = tool_runner.get_schema(df)
    steps_log = []
    context = {}
    fig = None

    # ── 1. PLAN ──────────────────────────────────────────────────────────────
    try:
        plan_raw = llm.call(
            PLANNER_PROMPT.format(schema=schema),
            f"Question: {question}"
        )
        # Strip markdown code fences if present
        plan_raw = plan_raw.strip()
        if plan_raw.startswith("```"):
            plan_raw = plan_raw.split("```")[1]
            if plan_raw.startswith("json"):
                plan_raw = plan_raw[4:]
        plan = json.loads(plan_raw)
        if not isinstance(plan, list):
            plan = [plan]
    except Exception as e:
        return {"answer": f"Planning failed: {e}", "fig": None, "steps": [], "error": str(e)}

    # ── 2. ACT + REFLECT ─────────────────────────────────────────────────────
    for i, step in enumerate(plan):
        step_key = f"step_{i}"
        step_type = step.get("type")
        goal = step.get("goal", "")

        if step_type == "query":
            res = tool_runner.run_query(df, step["code"])
            if res["ok"]:
                result_val = res["result"]
                # Validate intermediate result
                result_str = str(result_val)[:500]  # truncate for prompt
                try:
                    val_raw = llm.call(
                        VALIDATOR_PROMPT.format(
                            question=question, goal=goal, result=result_str
                        ),
                        "Validate."
                    )
                    val = json.loads(val_raw.strip())
                    if not val.get("valid", True):
                        # Replan
                        replan_raw = llm.call(
                            REPLAN_PROMPT.format(
                                question=question, schema=schema,
                                step=json.dumps(step), error=val["reason"],
                                context=json.dumps({k: str(v) for k, v in context.items()})
                            ),
                            "Replan."
                        )
                        new_plan = json.loads(replan_raw.strip())
                        plan = new_plan  # replace remaining plan
                        steps_log.append({"step": step_key, "type": "replan", "reason": val["reason"]})
                        continue
                except Exception:
                    pass  # validation is best-effort; don't block on it

                context[step_key] = result_str
                steps_log.append({"step": step_key, "type": "query", "goal": goal, "ok": True})
            else:
                context[step_key] = f"ERROR: {res['error']}"
                steps_log.append({"step": step_key, "type": "query", "goal": goal, "ok": False, "error": res["error"]})

        elif step_type == "plot":
            res = tool_runner.run_plot(df, step["code"])
            if res["ok"]:
                fig = res["fig"]
                context[step_key] = "Plot generated successfully."
                steps_log.append({"step": step_key, "type": "plot", "goal": goal, "ok": True})
            else:
                context[step_key] = f"Plot failed: {res['error']}"
                steps_log.append({"step": step_key, "type": "plot", "goal": goal, "ok": False, "error": res["error"]})

        elif step_type == "reason":
            # Pure reasoning step — just collect relevant context
            relevant = {k: context[k] for k in step.get("context_keys", []) if k in context}
            context[step_key] = f"Reasoning over: {json.dumps(relevant)}"
            steps_log.append({"step": step_key, "type": "reason", "goal": goal})

    # ── 3. ANSWER ─────────────────────────────────────────────────────────────
    # Build conversation-aware answer (include last 3 turns for follow-up support)
    recent_history = ""
    if history:
        for turn in history[-6:]:  # last 3 Q&A pairs
            recent_history += f"{turn['role'].upper()}: {turn['content']}\n"

    context_str = "\n".join([f"[{k}] {v}" for k, v in context.items()])
    if recent_history:
        context_str = f"Conversation so far:\n{recent_history}\n\nAnalysis results:\n{context_str}"

    answer = llm.call(
        ANSWER_PROMPT.format(question=question, schema=schema, context=context_str),
        "Provide the final answer."
    )

    return {"answer": answer, "fig": fig, "steps": steps_log, "error": None}