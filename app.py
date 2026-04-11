"""
app.py — Streamlit UI.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import agent
import llm
import tools as tool_runner

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Planner Assistant",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
  code, .stCode, pre { font-family: 'IBM Plex Mono', monospace !important; }
  #MainMenu, footer, header { visibility: hidden; }
  section[data-testid="stSidebar"] { background: #0f0f0f; border-right: 1px solid #222; }
  section[data-testid="stSidebar"] * { color: #ccc !important; }
  .user-msg {
    background: #1a1a2e; border-left: 3px solid #4a9eff;
    padding: 12px 16px; margin: 8px 0; border-radius: 0 6px 6px 0; font-size: 0.95rem;
  }
  .bot-msg {
    background: #0d1117; border-left: 3px solid #2ea043;
    padding: 12px 16px; margin: 8px 0; border-radius: 0 6px 6px 0;
    font-size: 0.95rem; color: #e6edf3;
  }
  .step-badge {
    display: inline-block; background: #21262d; color: #8b949e;
    font-size: 0.72rem; padding: 2px 8px; border-radius: 12px; margin: 2px;
    font-family: 'IBM Plex Mono', monospace;
  }
  .insight-box {
    background: #1c2128; border: 1px solid #30363d; border-radius: 6px;
    padding: 10px 14px; margin-top: 8px; font-size: 0.88rem; color: #79c0ff;
  }
  .stButton button {
    background: #21262d; color: #e6edf3; border: 1px solid #30363d;
    border-radius: 4px; font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem;
  }
  .stButton button:hover { background: #30363d; border-color: #8b949e; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("messages", []),
    ("df", None),
    ("filename", None),
    ("quick_qs", []),
    ("quick_qs_ready", False),  # flag: generation was attempted for current file
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Quick question generation ──────────────────────────────────────────────────
QUICK_Q_PROMPT = """You are a data analyst. Given a spreadsheet schema, generate exactly 6 short specific
questions a user might ask about THIS data.

Rules:
- Use actual column names from the schema.
- Mix: 1-2 aggregations, 1-2 distribution/plot questions, 1 anomaly/gap, 1 summary.
- Each question must be answerable from the data alone.
- Keep each question under 10 words.
- Output ONLY a JSON array of 6 strings. No markdown, no explanation.

Schema:
{schema}
"""

FALLBACK_QS = [
    "Summarize this dataset",
    "Show distribution of key columns",
    "What are the top 5 rows by value?",
    "Identify any gaps or anomalies",
    "Plot the main numeric columns",
    "What columns have missing values?",
]

def generate_quick_questions(df: pd.DataFrame) -> list:
    """Generate schema-aware questions via LLM. Falls back to generics on any error."""
    try:
        schema = tool_runner.get_schema(df)
        raw = llm.call(QUICK_Q_PROMPT.format(schema=schema), "Generate questions.")
        # Use agent's robust parser — handles fences, trailing commas, single quotes
        parsed = agent.safe_json_loads(agent.extract_json(raw))
        if isinstance(parsed, list) and len(parsed) >= 4:
            return [str(q) for q in parsed[:6]]
    except Exception:
        pass
    return FALLBACK_QS


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙ Planner Assistant")
    st.markdown("---")

    uploaded = st.file_uploader("Upload spreadsheet", type=["csv", "xlsx", "xls"])

    if uploaded and uploaded.name != st.session_state.filename:
        try:
            df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            st.session_state.df = df
            st.session_state.filename = uploaded.name
            st.session_state.messages = []
            st.session_state.quick_qs = []
            st.session_state.quick_qs_ready = False  # reset so new questions generate
            st.success(f"Loaded {len(df):,} rows × {len(df.columns)} cols")
        except Exception as e:
            st.error(f"Load failed: {e}")

    if st.session_state.df is not None:
        st.markdown("**Columns**")
        for col, dtype in st.session_state.df.dtypes.items():
            st.markdown(f'<span class="step-badge">{col} · {dtype}</span>', unsafe_allow_html=True)
        st.markdown("---")
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.markdown('<span style="font-size:0.75rem;color:#555;">Plan → Act → Answer<br>Swap LLM in llm.py</span>',
                unsafe_allow_html=True)

# ── Guard: no file loaded ──────────────────────────────────────────────────────
if st.session_state.df is None:
    st.markdown("## ⚙ Planner Assistant")
    st.markdown("Upload a spreadsheet in the sidebar to begin.")
    st.markdown("""
    **Handles:**
    - Simple lookups — *"How many rows have status = delayed?"*
    - Aggregations — *"What is total output by line?"*
    - Reasoning — *"Why is there a gap between task A and B?"*
    - Follow-ups — *"Show only the top 5 from that"*
    - Visualizations — *"Plot throughput by shift"*
    """)
    st.stop()

# ── File header ────────────────────────────────────────────────────────────────
st.markdown(f"### `{st.session_state.filename}`")
st.markdown(
    f"<small style='color:#555'>{len(st.session_state.df):,} rows · "
    f"{len(st.session_state.df.columns)} columns</small>",
    unsafe_allow_html=True
)
with st.expander("Preview data", expanded=False):
    st.dataframe(st.session_state.df.head(10), use_container_width=True)

st.markdown("---")

# ── Chat history ───────────────────────────────────────────────────────────────
with st.container():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            content = msg["content"]
            insight = ""
            if "**Insight:**" in content:
                parts = content.split("**Insight:**")
                content = parts[0].strip()
                insight = parts[1].strip() if len(parts) > 1 else ""

            st.markdown(f'<div class="bot-msg">{content}</div>', unsafe_allow_html=True)
            if insight:
                st.markdown(
                    f'<div class="insight-box">💡 <strong>Insight:</strong> {insight}</div>',
                    unsafe_allow_html=True
                )
            if "steps" in msg and msg["steps"]:
                step_html = "".join([
                    f'<span class="step-badge">{"✓" if s.get("ok", True) else "✗"} '
                    f'{s.get("type","step")}: {s.get("goal", s.get("error",""))[:40]}</span>'
                    for s in msg["steps"]
                ])
                st.markdown(f"<div style='margin:4px 0'>{step_html}</div>", unsafe_allow_html=True)
            if msg.get("fig") is not None:
                st.plotly_chart(msg["fig"], use_container_width=True)

# ── Quick questions (lazy, generated once per file) ────────────────────────────
st.markdown("---")

if not st.session_state.quick_qs_ready:
    # Only generate once per file upload — the ready flag prevents re-triggering
    # on every rerun (which happens after each chat message)
    with st.spinner("Generating suggestions..."):
        st.session_state.quick_qs = generate_quick_questions(st.session_state.df)
        st.session_state.quick_qs_ready = True

question = None
with st.expander("Suggested questions", expanded=True):
    cols = st.columns(2)
    for i, q in enumerate(st.session_state.quick_qs):
        if cols[i % 2].button(q, key=f"quick_{i}"):
            question = q

# ── Chat input ─────────────────────────────────────────────────────────────────
chat_question = st.chat_input("Ask anything about your data...")
if chat_question:
    question = chat_question

# ── Process ────────────────────────────────────────────────────────────────────
if question:
    st.session_state.messages.append({"role": "user", "content": question})

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-6:]
        if m["role"] in ("user", "assistant")
    ]

    with st.spinner("Thinking..."):
        result = agent.run(question, st.session_state.df, history)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "steps": result.get("steps", []),
        "fig": result.get("fig"),
    })
    st.rerun()