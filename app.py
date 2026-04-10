"""
app.py — Streamlit UI.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import agent

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Planner Assistant",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }
  code, .stCode, pre {
    font-family: 'IBM Plex Mono', monospace !important;
  }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0f0f0f;
    border-right: 1px solid #222;
  }
  section[data-testid="stSidebar"] * { color: #ccc !important; }

  /* Chat bubbles */
  .user-msg {
    background: #1a1a2e;
    border-left: 3px solid #4a9eff;
    padding: 12px 16px;
    margin: 8px 0;
    border-radius: 0 6px 6px 0;
    font-size: 0.95rem;
  }
  .bot-msg {
    background: #0d1117;
    border-left: 3px solid #2ea043;
    padding: 12px 16px;
    margin: 8px 0;
    border-radius: 0 6px 6px 0;
    font-size: 0.95rem;
    color: #e6edf3;
  }
  .step-badge {
    display: inline-block;
    background: #21262d;
    color: #8b949e;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 12px;
    margin: 2px;
    font-family: 'IBM Plex Mono', monospace;
  }
  .insight-box {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 10px 14px;
    margin-top: 8px;
    font-size: 0.88rem;
    color: #79c0ff;
  }
  h1 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -1px; }
  .stButton button {
    background: #21262d;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
  }
  .stButton button:hover {
    background: #30363d;
    border-color: #8b949e;
  }
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "filename" not in st.session_state:
    st.session_state.filename = None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙ Planner Assistant")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload spreadsheet",
        type=["csv", "xlsx", "xls"],
        help="CSV or Excel file"
    )

    if uploaded and uploaded.name != st.session_state.filename:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state.df = df
            st.session_state.filename = uploaded.name
            st.session_state.messages = []  # reset chat on new file
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
    st.markdown('<span style="font-size:0.75rem;color:#555;">Plan → Act → Reflect loop<br>Swap LLM in llm.py</span>', unsafe_allow_html=True)

# ── Main area ──────────────────────────────────────────────────────────────────
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

st.markdown(f"### `{st.session_state.filename}`")
st.markdown(f"<small style='color:#555'>{len(st.session_state.df):,} rows · {len(st.session_state.df.columns)} columns</small>", unsafe_allow_html=True)

with st.expander("Preview data", expanded=False):
    st.dataframe(st.session_state.df.head(10), use_container_width=True)

st.markdown("---")

# ── Chat history ───────────────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            # Split out insight line for special rendering
            content = msg["content"]
            insight = ""
            if "**Insight:**" in content:
                parts = content.split("**Insight:**")
                content = parts[0].strip()
                insight = parts[1].strip() if len(parts) > 1 else ""

            st.markdown(f'<div class="bot-msg">{content}</div>', unsafe_allow_html=True)
            if insight:
                st.markdown(f'<div class="insight-box">💡 <strong>Insight:</strong> {insight}</div>', unsafe_allow_html=True)

            # Show agent steps taken
            if "steps" in msg and msg["steps"]:
                step_html = "".join([
                    f'<span class="step-badge">{"✓" if s.get("ok", True) else "✗"} {s["type"]}: {s["goal"][:40]}</span>'
                    for s in msg["steps"]
                ])
                st.markdown(f"<div style='margin:4px 0'>{step_html}</div>", unsafe_allow_html=True)

            # Render chart if any
            if "fig" in msg and msg["fig"] is not None:
                st.plotly_chart(msg["fig"], use_container_width=True)

# ── Input ──────────────────────────────────────────────────────────────────────
st.markdown("---")

# Quick question chips
question = None
quick_qs = [
    "Summarize this dataset",
    "Show distribution of key columns",
    "What are the top 5 rows by value?",
    "Identify any gaps or anomalies",
]
with st.expander("Quick questions", expanded=False):
    cols = st.columns(2)
    for i, q in enumerate(quick_qs):
        if cols[i % 2].button(q, key=f"quick_{i}"):
            question = q

# Chat input must be at top level (not inside columns/expander/form)
chat_question = st.chat_input("Ask anything about your data...")
if chat_question:
    question = chat_question

# ── Process question ───────────────────────────────────────────────────────────
if question:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})

    # Build history for follow-up context (last 6 turns)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-6:]
        if m["role"] in ("user", "assistant")
    ]

    with st.spinner("Thinking..."):
        result = agent.run(question, st.session_state.df, history)

    # Store response with metadata
    bot_msg = {
        "role": "assistant",
        "content": result["answer"],
        "steps": result.get("steps", []),
        "fig": result.get("fig"),
    }
    st.session_state.messages.append(bot_msg)
    st.rerun()