from __future__ import annotations

import pandas as pd

from app.tools.errors import ToolExecutionError
from app.tools.filtering import apply_filters


def describe_data(df: pd.DataFrame, columns: list[str] | None = None, filters: list[dict] | None = None) -> dict:
    working = apply_filters(df, filters) if filters else df
    if working.empty:
        raise ToolExecutionError("No rows match the given filters.")

    target_cols = columns or list(working.select_dtypes(include="number").columns)
    missing_cols = [c for c in target_cols if c not in working.columns]
    if missing_cols:
        raise ToolExecutionError(f"Unknown column(s): {', '.join(missing_cols)}")
    if not target_cols:
        raise ToolExecutionError("No numeric columns available to describe.")

    result: dict = {}
    for col in target_cols:
        series = working[col]
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            result[col] = {
                "count": int(desc.get("count", 0)),
                "mean": _round(desc.get("mean")),
                "std": _round(desc.get("std")),
                "min": _round(desc.get("min")),
                "p25": _round(desc.get("25%")),
                "median": _round(desc.get("50%")),
                "p75": _round(desc.get("75%")),
                "max": _round(desc.get("max")),
                "sum": _round(_safe_sum(series)),
            }
        else:
            top = series.value_counts(dropna=True).head(5)
            result[col] = {
                "count": int(series.count()),
                "unique": int(series.nunique(dropna=True)),
                "top_values": {str(k): int(v) for k, v in top.items()},
            }
    return {"row_count": int(len(working)), "columns": result}


def _safe_sum(series: pd.Series):
    """Same real int64-wraparound bug fixed in aggregation.py::group_and_aggregate
    -- pandas/numpy int64 sum silently overflows to a wildly wrong (often negative)
    value with no warning. Object-dtype sum forces Python's arbitrary-precision int
    arithmetic instead."""
    if pd.api.types.is_integer_dtype(series):
        return series.astype(object).sum()
    return series.sum()


def _round(value: float | None, ndigits: int = 4) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), ndigits)
