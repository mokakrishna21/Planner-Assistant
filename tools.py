import traceback
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def get_schema(df: pd.DataFrame) -> str:
    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    sample = df.head(3).to_dict(orient="records")

    stats = {}
    for col in df.select_dtypes(include="number").columns:
        stats[col] = {
            "min": df[col].min(),
            "max": df[col].max(),
            "mean": round(df[col].mean(), 2),
        }

    return str({"columns": schema, "sample_rows": sample, "numeric_stats": stats})


def run_query(df, code, namespace):
    try:
        namespace.pop("result", None)
        exec(code, namespace)
        result = namespace.get("result")

        if result is None:
            return {"ok": False, "error": "No result"}

        if isinstance(result, pd.DataFrame):
            return {"ok": True, "result": result.head(50)}

        return {"ok": True, "result": result}

    except Exception as e:
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


def run_plot(df, code, namespace):
    try:
        namespace.pop("fig", None)
        exec(code, namespace)
        fig = namespace.get("fig")

        if fig is None:
            return {"ok": False, "error": "No fig"}

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="monospace", size=12),
            margin=dict(l=20, r=20, t=40, b=20),
        )

        return {"ok": True, "fig": fig}

    except Exception as e:
        return {"ok": False, "error": str(e)}