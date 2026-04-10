"""
tools.py — Atomic operations the agent can invoke.
Each tool takes (df, params) and returns a result dict.
"""

import json
import traceback
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def get_schema(df: pd.DataFrame) -> str:
    """Return schema + sample as a compact string for LLM context."""
    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    sample = df.head(3).to_dict(orient="records")
    stats = {}
    for col in df.select_dtypes(include="number").columns:
        stats[col] = {"min": df[col].min(), "max": df[col].max(), "mean": round(df[col].mean(), 2)}
    return json.dumps({"columns": schema, "sample_rows": sample, "numeric_stats": stats}, default=str)


def run_query(df: pd.DataFrame, code: str) -> dict:
    """
    Execute pandas code against df.
    The code must assign result to a variable named `result`.
    Returns {"ok": True, "result": ..., "type": "dataframe"|"scalar"|"string"}
    """
    local_ns = {"df": df.copy(), "pd": pd}
    try:
        exec(code, local_ns)
        result = local_ns.get("result", None)
        if result is None:
            return {"ok": False, "error": "Code did not assign to `result`"}
        if isinstance(result, pd.DataFrame):
            return {"ok": True, "result": result, "type": "dataframe"}
        elif isinstance(result, (int, float)):
            return {"ok": True, "result": result, "type": "scalar"}
        else:
            return {"ok": True, "result": str(result), "type": "string"}
    except Exception as e:
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


def run_plot(df: pd.DataFrame, code: str):
    """
    Execute plotly code against df.
    The code must assign a plotly figure to `fig`.
    Returns {"ok": True, "fig": fig} or {"ok": False, "error": ...}
    """
    local_ns = {"df": df.copy(), "pd": pd, "px": px, "go": go}
    try:
        exec(code, local_ns)
        fig = local_ns.get("fig", None)
        if fig is None:
            return {"ok": False, "error": "Code did not assign to `fig`"}
        # Apply clean styling
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="monospace", size=12),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return {"ok": True, "fig": fig}
    except Exception as e:
        return {"ok": False, "error": str(e)}