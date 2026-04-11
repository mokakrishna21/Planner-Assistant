import json
import traceback
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')


def convert_to_serializable(obj):
    """Recursively convert numpy/pandas types to native Python.
    NOTE: Must check container types (DataFrame, Series, ndarray) BEFORE
    calling pd.isna() — pd.isna(DataFrame) returns a DataFrame, not a bool,
    which crashes on truth-value evaluation.
    """
    # Container types first — NEVER call pd.isna on these
    if isinstance(obj, pd.DataFrame):
        return obj.head(50).to_dict(orient='records')
    if isinstance(obj, pd.Series):
        return [convert_to_serializable(x) for x in obj.tolist()]
    if isinstance(obj, np.ndarray):
        return [convert_to_serializable(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    # Scalar types — safe to call pd.isna now
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)

    return obj


def get_schema(df: pd.DataFrame) -> str:
    """Return structured schema as JSON string for LLM context."""
    try:
        if df is None:
            return json.dumps({"error": "No DataFrame provided"})
        if df.empty:
            return json.dumps({
                "columns": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "column_names": list(df.columns),
                "sample_rows": [],
                "note": "DataFrame is empty"
            })

        columns = {col: str(dtype) for col, dtype in df.dtypes.items()}

        sample_rows = []
        for _, row in df.head(5).iterrows():
            clean_row = {}
            for col in df.columns:
                try:
                    clean_row[col] = convert_to_serializable(row[col])
                except Exception:
                    clean_row[col] = str(row[col])
            sample_rows.append(clean_row)

        numeric_stats = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            try:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    numeric_stats[col] = {
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "mean": round(float(col_data.mean()), 4),
                        "null_count": int(df[col].isna().sum())
                    }
            except Exception as e:
                numeric_stats[col] = {"error": str(e)}

        categorical_info = {}
        for col in df.select_dtypes(include=['object', 'category']).columns:
            try:
                unique_vals = df[col].dropna().unique()
                categorical_info[col] = {
                    "unique_count": int(len(unique_vals)),
                    "sample_values": [str(v) for v in unique_vals[:8]]
                }
            except Exception:
                pass

        schema = {
            "columns": columns,
            "column_names": list(df.columns),
            "sample_rows": sample_rows,
            "numeric_stats": numeric_stats,
            "categorical_info": categorical_info,
            "total_rows": len(df),
            "total_columns": len(df.columns)
        }

        return json.dumps(schema, default=str, indent=2)

    except Exception as e:
        return json.dumps({
            "error": f"Schema generation failed: {str(e)}",
            "column_names": list(df.columns) if df is not None else []
        })


def run_query(df, code, namespace):
    """Execute pandas query code. Code must assign to `result`."""
    try:
        namespace.pop("result", None)  # prevent stale value leak
        if not code or not code.strip():
            return {"ok": False, "error": "Empty code"}
        exec(code, namespace)
        result = namespace.get("result")
        if result is None:
            return {"ok": False, "error": "Code did not assign to `result`. Use: result = ..."}
        return {"ok": True, "result": result}
    except KeyError as e:
        return {"ok": False, "error": f"Column not found: {e}. Available: {list(df.columns)}"}
    except SyntaxError as e:
        return {"ok": False, "error": f"Syntax error: {e}"}
    except Exception as e:
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


def run_plot(df, code, namespace):
    """Execute plotly chart code. Code must assign to `fig`."""
    try:
        namespace.pop("fig", None)  # prevent stale fig leak
        if not code or not code.strip():
            return {"ok": False, "error": "Empty plot code"}
        exec(code, namespace)
        fig = namespace.get("fig")
        if fig is None:
            return {"ok": False, "error": "Code did not assign to `fig`. Use: fig = px..."}
        if not hasattr(fig, 'update_layout'):
            return {"ok": False, "error": "Result is not a Plotly figure"}
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3"),
            title_font=dict(color="#e6edf3"),
            legend_font=dict(color="#8b949e"),
            xaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d"),
            yaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d"),
        )
        return {"ok": True, "fig": fig}
    except Exception as e:
        return {"ok": False, "error": f"Plot failed: {e}"}