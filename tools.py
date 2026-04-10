import json
import traceback
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')


def convert_to_serializable(obj):
    """Recursively convert numpy/pandas types to native Python."""
    if pd.isna(obj):
        return None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [convert_to_serializable(x) for x in obj.tolist()]
    if isinstance(obj, (pd.Series,)):
        return [convert_to_serializable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.DataFrame):
        # Convert DataFrame to dict records for JSON safety
        return obj.head(50).to_dict(orient='records')
    return obj


def get_schema(df: pd.DataFrame) -> str:
    """Return structured schema with robust error handling."""
    try:
        if df is None:
            return json.dumps({"error": "No DataFrame provided"})
        
        # Handle empty DataFrame
        if df.empty:
            return json.dumps({
                "columns": list(df.columns),
                "column_names": list(df.columns),
                "sample_rows": [],
                "note": "DataFrame is empty"
            })
        
        # Safe dtype conversion
        columns = {}
        for col, dtype in df.dtypes.items():
            try:
                columns[col] = str(dtype)
            except:
                columns[col] = "unknown"
        
        # Safe sample rows (handle non-serializable values)
        sample_rows = []
        for idx, row in df.head(5).iterrows():
            clean_row = {}
            for col in df.columns:
                val = row[col]
                try:
                    # Try to serialize, fall back to string representation
                    json.dumps(val, default=str)
                    clean_row[col] = val
                except:
                    clean_row[col] = str(val)
            sample_rows.append(clean_row)
        
        # Numeric stats with error handling
        numeric_stats = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            try:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    numeric_stats[col] = {
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "mean": float(col_data.mean()),
                        "null_count": int(df[col].isna().sum())
                    }
            except Exception as e:
                numeric_stats[col] = {"error": str(e)}
        
        schema = {
            "columns": columns,
            "column_names": list(df.columns),
            "sample_rows": sample_rows,
            "numeric_stats": numeric_stats,
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
    """Execute query code with comprehensive error handling."""
    try:
        # Ensure clean state
        namespace.pop("result", None)
        namespace.pop("__result", None)
        
        # Validate code isn't empty
        if not code or not code.strip():
            return {"ok": False, "error": "Empty code provided"}
        
        # Execute with timeout protection (implicit via Streamlit/Groq timeouts)
        exec(code, namespace)
        result = namespace.get("result")
        
        if result is None:
            return {"ok": False, "error": "No result variable assigned. Use: result = ..."}
        
        # Convert to serializable form
        result = convert_to_serializable(result)
        
        # Size limits for display
        if isinstance(result, list) and len(result) > 1000:
            result = result[:1000] + [f"... ({len(result)-1000} more items)"]
        if isinstance(result, dict) and len(str(result)) > 10000:
            result = {"preview": str(result)[:1000] + "...", "note": "Result truncated"}
            
        return {"ok": True, "result": result}
        
    except SyntaxError as e:
        return {
            "ok": False, 
            "error": f"Syntax error in generated code: {str(e)}",
            "hint": "Check for proper quotes and parentheses"
        }
    except KeyError as e:
        return {
            "ok": False, 
            "error": f"Column not found: {str(e)}",
            "hint": f"Available columns: {list(df.columns)}"
        }
    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        
        # Sanitize error message (don't expose full paths)
        safe_error = error_msg.split('\n')[-1] if '\n' in error_msg else error_msg
        
        return {
            "ok": False, 
            "error": safe_error,
            "traceback": tb if "development" in str(e) else None  # Only in debug
        }


def run_plot(df, code, namespace):
    """Execute plotting code with error handling."""
    try:
        namespace.pop("fig", None)
        
        if not code or not code.strip():
            return {"ok": False, "error": "Empty plotting code"}
        
        exec(code, namespace)
        fig = namespace.get("fig")
        
        if fig is None:
            return {"ok": False, "error": "No figure assigned. Use: fig = px... or fig = go.Figure()"}
        
        # Validate it's a plotly figure
        if not hasattr(fig, 'update_layout'):
            return {"ok": False, "error": "Result is not a Plotly figure"}
        
        # Clean styling
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
        return {"ok": False, "error": f"Plot generation failed: {str(e)}"}