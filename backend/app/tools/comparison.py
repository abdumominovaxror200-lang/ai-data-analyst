from __future__ import annotations

import pandas as pd

from app.tools.errors import ToolExecutionError
from app.tools.filtering import apply_filters

_AGG_FUNCS = {"sum", "mean", "median", "count", "min", "max"}


def compare_periods(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    current_start: str,
    current_end: str,
    previous_start: str,
    previous_end: str,
    agg_func: str = "sum",
    filters: list[dict] | None = None,
) -> dict:
    working = apply_filters(df, filters) if filters else df
    if date_column not in working.columns:
        raise ToolExecutionError(f"Unknown column '{date_column}'.")
    if value_column not in working.columns:
        raise ToolExecutionError(f"Unknown column '{value_column}'.")
    if agg_func not in _AGG_FUNCS:
        raise ToolExecutionError(f"Unsupported aggregation '{agg_func}'. Use one of {sorted(_AGG_FUNCS)}.")

    dates = pd.to_datetime(working[date_column], errors="coerce")

    def _slice(start: str, end: str) -> pd.Series:
        start_ts, end_ts = pd.to_datetime(start), pd.to_datetime(end)
        mask = (dates >= start_ts) & (dates <= end_ts)
        return working.loc[mask, value_column]

    current = _slice(current_start, current_end)
    previous = _slice(previous_start, previous_end)
    if current.empty and previous.empty:
        raise ToolExecutionError("No data found in either period.")

    current_val = _agg(current, agg_func)
    previous_val = _agg(previous, agg_func)
    delta = None
    pct_change = None
    if current_val is not None and previous_val is not None:
        delta = round(current_val - previous_val, 4)
        pct_change = round((delta / previous_val) * 100, 2) if previous_val else None

    return {
        "current_period": {"start": current_start, "end": current_end, "value": current_val, "n": int(len(current))},
        "previous_period": {"start": previous_start, "end": previous_end, "value": previous_val, "n": int(len(previous))},
        "delta": delta,
        "pct_change": pct_change,
        "agg_func": agg_func,
    }


def _agg(series: pd.Series, agg_func: str) -> float | None:
    if series.empty:
        return None
    return round(float(getattr(series, agg_func)()), 4)
