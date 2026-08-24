from __future__ import annotations

import pandas as pd

from app.tools.errors import ToolExecutionError


def _role_for(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if series.nunique(dropna=True) <= max(20, int(len(series) * 0.05)):
        return "categorical"
    return "text"


def profile_dataset(df: pd.DataFrame) -> dict:
    if df.empty:
        raise ToolExecutionError("Dataset is empty.")

    role_buckets: dict[str, list[str]] = {"numeric": [], "categorical": [], "datetime": [], "boolean": []}
    column_info = []

    for col in df.columns:
        series = df[col]
        role = _role_for(series)
        missing = int(series.isna().sum())
        column_info.append(
            {
                "name": str(col),
                "dtype": str(series.dtype),
                "role": role,
                "missing_count": missing,
                "missing_pct": round(missing / len(df) * 100, 2) if len(df) else 0.0,
                "unique_count": int(series.nunique(dropna=True)),
            }
        )
        if role in role_buckets:
            role_buckets[role].append(str(col))

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_info": column_info,
        "numeric_columns": role_buckets["numeric"],
        "categorical_columns": role_buckets["categorical"],
        "date_columns": role_buckets["datetime"],
        "boolean_columns": role_buckets["boolean"],
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
