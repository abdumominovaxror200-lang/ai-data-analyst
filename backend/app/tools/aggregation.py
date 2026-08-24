from __future__ import annotations

import pandas as pd

from app.tools.errors import ToolExecutionError
from app.tools.filtering import apply_filters

_AGG_FUNCS = {"sum", "mean", "median", "count", "min", "max"}


def group_and_aggregate(
    df: pd.DataFrame,
    group_by: str,
    agg_column: str,
    agg_func: str = "sum",
    filters: list[dict] | None = None,
    top_n: int | None = 20,
) -> dict:
    working = apply_filters(df, filters) if filters else df
    if group_by not in working.columns:
        raise ToolExecutionError(f"Unknown column '{group_by}'.")
    if agg_func not in _AGG_FUNCS:
        raise ToolExecutionError(f"Unsupported aggregation '{agg_func}'. Use one of {sorted(_AGG_FUNCS)}.")
    if agg_func != "count" and agg_column not in working.columns:
        raise ToolExecutionError(f"Unknown column '{agg_column}'.")
    if working.empty:
        raise ToolExecutionError("No rows match the given filters.")

    if agg_func == "count":
        grouped = working.groupby(group_by, dropna=False).size().rename("value")
    else:
        if not pd.api.types.is_numeric_dtype(working[agg_column]):
            raise ToolExecutionError(f"Column '{agg_column}' is not numeric; cannot compute '{agg_func}'.")
        grouped = working.groupby(group_by, dropna=False)[agg_column].agg(agg_func).rename("value")

    grouped = grouped.sort_values(ascending=False)
    if top_n:
        grouped = grouped.head(top_n)

    groups = [{"group": str(idx), "value": round(float(val), 4)} for idx, val in grouped.items()]
    return {
        "group_by": group_by,
        "agg_column": agg_column,
        "agg_func": agg_func,
        "groups": groups,
        "group_count": int(working[group_by].nunique(dropna=False)),
    }
