from __future__ import annotations

import pandas as pd

from app.tools.errors import ToolExecutionError
from app.tools.serialization import dataframe_to_records

_OPS = {"==", "!=", ">", ">=", "<", "<=", "in", "contains", "between"}


def apply_filters(df: pd.DataFrame, filters: list[dict] | None) -> pd.DataFrame:
    if not filters:
        return df
    working = df
    for condition in filters:
        working = _apply_one(working, condition)
    return working


def _apply_one(df: pd.DataFrame, condition: dict) -> pd.DataFrame:
    column = condition.get("column")
    op = condition.get("op")
    value = condition.get("value")

    if column not in df.columns:
        raise ToolExecutionError(f"Unknown column '{column}'.")
    if op not in _OPS:
        raise ToolExecutionError(f"Unsupported operator '{op}'. Use one of {sorted(_OPS)}.")

    series = df[column]
    if pd.api.types.is_datetime64_any_dtype(series) and op in {"==", "!=", ">", ">=", "<", "<=", "between"}:
        value = [pd.to_datetime(v) for v in value] if isinstance(value, list) else pd.to_datetime(value)

    try:
        if op == "==":
            mask = series == value
        elif op == "!=":
            mask = series != value
        elif op == ">":
            mask = series > value
        elif op == ">=":
            mask = series >= value
        elif op == "<":
            mask = series < value
        elif op == "<=":
            mask = series <= value
        elif op == "in":
            mask = series.isin(value if isinstance(value, list) else [value])
        elif op == "contains":
            mask = series.astype(str).str.contains(str(value), case=False, na=False)
        else:  # between
            lo, hi = value
            mask = series.between(lo, hi)
    except TypeError as exc:
        raise ToolExecutionError(f"Cannot apply '{op}' to column '{column}': {exc}") from exc

    return df[mask]


def filter_data(df: pd.DataFrame, filters: list[dict]) -> dict:
    if not filters:
        raise ToolExecutionError("At least one filter condition is required.")
    filtered = apply_filters(df, filters)
    preview = dataframe_to_records(filtered.head(10))
    return {
        "matched_rows": int(len(filtered)),
        "total_rows": int(len(df)),
        "match_pct": round(len(filtered) / len(df) * 100, 2) if len(df) else 0.0,
        "preview": preview,
    }
