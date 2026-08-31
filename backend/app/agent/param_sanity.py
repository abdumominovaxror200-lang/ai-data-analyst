from __future__ import annotations

import pandas as pd

from app.tools.errors import ToolExecutionError
from app.tools.filtering import apply_filters

_MAX_TOP_N = 100
_MIN_GROUP_N = 5


def validate_tool_params(df: pd.DataFrame, tool: str, params: dict) -> None:
    """Central deterministic pre-execution validation for analytical tool calls."""
    filters = params.get("filters")
    working = apply_filters(df, filters) if isinstance(filters, list) and filters else df
    if working.empty:
        raise ToolExecutionError("The requested filters match zero rows; adjust the filters before analysis.")

    top_n = params.get("top_n")
    if top_n is not None and (not isinstance(top_n, int) or isinstance(top_n, bool) or not 1 <= top_n <= _MAX_TOP_N):
        raise ToolExecutionError(f"top_n must be an integer between 1 and {_MAX_TOP_N}.")

    agg_column = params.get("agg_column")
    if isinstance(agg_column, str):
        if agg_column not in working.columns:
            raise ToolExecutionError(f"Unknown aggregation column '{agg_column}'.")
        if params.get("agg_func", "sum") != "count" and not pd.api.types.is_numeric_dtype(working[agg_column]):
            raise ToolExecutionError(f"Aggregation column '{agg_column}' must be numeric.")

    date_column = params.get("date_column")
    if isinstance(date_column, str) and date_column in working.columns:
        dates = pd.to_datetime(working[date_column], errors="coerce").dropna()
        if not dates.empty:
            available_min, available_max = dates.min(), dates.max()
            for start_key, end_key in (("current_start", "current_end"), ("previous_start", "previous_end")):
                start, end = params.get(start_key), params.get(end_key)
                if start is None or end is None:
                    continue
                try:
                    requested_start, requested_end = pd.Timestamp(start), pd.Timestamp(end)
                except (TypeError, ValueError) as exc:
                    raise ToolExecutionError(f"Invalid date range in {start_key}/{end_key}.") from exc
                if requested_start > requested_end:
                    raise ToolExecutionError(f"{start_key} must not be after {end_key}.")
                if requested_end < available_min or requested_start > available_max:
                    raise ToolExecutionError(
                        f"Requested {start_key}/{end_key} range is outside the available data range "
                        f"{available_min.date()} to {available_max.date()}."
                    )

    group_column = params.get("group_column")
    group_a, group_b = params.get("group_a"), params.get("group_b")
    if isinstance(group_column, str) and group_a is not None and group_b is not None:
        if group_column not in working.columns:
            raise ToolExecutionError(f"Unknown group column '{group_column}'.")
        sizes = working[group_column].value_counts(dropna=False)
        for label in (group_a, group_b):
            size = int(sizes.get(label, 0))
            if size < _MIN_GROUP_N:
                raise ToolExecutionError(f"Group '{label}' has n={size}; at least {_MIN_GROUP_N} rows are required.")
