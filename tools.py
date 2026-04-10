import json
import traceback
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def get_schema(df: pd.DataFrame) -> str:
    """Return structured schema for LLM reasoning"""
    schema = {
        "columns": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "column_names": list(df.columns),
        "sample_rows": df.head(5).to_dict(orient="records"),
    }

    numeric_stats = {}
    for col in df.select_dtypes(include="number").columns:
        numeric_stats[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
        }

    schema["numeric_stats"] = numeric_stats

    return json.dumps(schema, default=str)


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
        )

        return {"ok": True, "fig": fig}

    except Exception as e:
        return {"ok": False, "error": str(e)}