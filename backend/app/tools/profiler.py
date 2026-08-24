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

    role_buckets: dict[str, list[str]] = {"numeric": [], "categorical": [], "datetime": [], "boolean": [], "text": []}
    column_info = []
    date_ranges: dict[str, dict[str, str]] = {}

    for col in df.columns:
        series = df[col]
        role = _role_for(series)
        missing = int(series.isna().sum())
        min_date = max_date = None
        if role == "datetime":
            non_null = series.dropna()
            if not non_null.empty:
                min_date = non_null.min().strftime("%Y-%m-%d")
                max_date = non_null.max().strftime("%Y-%m-%d")
                date_ranges[str(col)] = {"min": min_date, "max": max_date}
        column_info.append(
            {
                "name": str(col),
                "dtype": str(series.dtype),
                "role": role,
                "missing_count": missing,
                "missing_pct": round(missing / len(df) * 100, 2) if len(df) else 0.0,
                "unique_count": int(series.nunique(dropna=True)),
                "min_date": min_date,
                "max_date": max_date,
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
        # High-cardinality non-numeric columns (e.g. free-text or ID-like fields such
        # as customer_id) — these don't fit the numeric/categorical/date/boolean
        # buckets but the agent still needs to know they exist (see agent.py's
        # dataset_context), or it can wrongly claim a column doesn't exist.
        "text_columns": role_buckets["text"],
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        # Quick lookup for date coverage without re-scanning column_info — used by the
        # agent's context message and by compare_periods' out-of-coverage detection.
        "date_ranges": date_ranges,
    }
