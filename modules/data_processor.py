"""
data_processor.py
Handles CSV loading, validation, and profiling.
Returns a clean DataFrame + a profile dict used as LLM context.
"""
import io
import pandas as pd


def load_and_profile(uploaded_file) -> tuple[pd.DataFrame, dict]:
    """
    Load a CSV from a Streamlit UploadedFile object.
    Returns (df, profile) where profile is a summary dict for the LLM.
    """
    content = uploaded_file.read()
    df = pd.read_csv(io.BytesIO(content))

    # Basic cleanup
    df.columns = df.columns.str.strip()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    total_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]

    # Build a compact text summary for the LLM context window
    describe_str = df.describe(include="all").to_string()
    sample_str = df.head(5).to_string()
    dtypes_str = df.dtypes.to_string()

    profile = {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "columns": df.columns.tolist(),
        "numeric_cols": len(numeric_cols),
        "numeric_col_names": numeric_cols,
        "categorical_col_names": categorical_cols,
        "missing_pct": (total_missing / total_cells * 100) if total_cells > 0 else 0,
        "describe": describe_str,
        "sample": sample_str,
        "dtypes": dtypes_str,
        "shape": df.shape,
    }

    return df, profile