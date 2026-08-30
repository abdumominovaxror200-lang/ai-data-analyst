from __future__ import annotations

import re

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

    dates = parse_date_series(working[date_column])
    available_min = dates.min()
    available_max = dates.max()

    def _slice(start: str, end: str) -> pd.Series:
        start_ts = _parse_boundary(start, "start")
        end_ts = _parse_boundary(end, "end")
        if start_ts > end_ts:
            raise ToolExecutionError(f"Period start {start!r} must not be after end {end!r}.")
        # A date-only end means the entire calendar day, including timestamped rows.
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", end.strip()):
            mask = (dates >= start_ts) & (dates < end_ts + pd.Timedelta(days=1))
        else:
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

    current_coverage = _coverage_note(current_start, current_end, available_min, available_max, "current")
    previous_coverage = _coverage_note(previous_start, previous_end, available_min, available_max, "previous")

    return {
        "current_period": {"start": current_start, "end": current_end, "value": current_val, "n": int(len(current))},
        "previous_period": {"start": previous_start, "end": previous_end, "value": previous_val, "n": int(len(previous))},
        "delta": delta,
        "pct_change": pct_change,
        "agg_func": agg_func,
        "dataset_date_coverage": {
            "min": available_min.strftime("%Y-%m-%d") if pd.notna(available_min) else None,
            "max": available_max.strftime("%Y-%m-%d") if pd.notna(available_max) else None,
        },
        "current_period_coverage_warning": current_coverage,
        "previous_period_coverage_warning": previous_coverage,
    }


def _coverage_note(
    requested_start: str,
    requested_end: str,
    available_min: pd.Timestamp,
    available_max: pd.Timestamp,
    label: str,
) -> dict | None:
    """Flags when a requested period isn't fully backed by real data, instead of
    silently computing a value over whatever partial data happens to fall in range."""
    if pd.isna(available_min) or pd.isna(available_max):
        return None

    req_start = _parse_boundary(requested_start, "start")
    req_end = _parse_boundary(requested_end, "end")
    requested_days = (req_end - req_start).days + 1

    overlap_start = max(req_start, available_min)
    overlap_end = min(req_end, available_max)

    if overlap_start > overlap_end:
        return {
            "full_coverage": False,
            "coverage_pct": 0.0,
            "note": (
                f"No data exists for the requested {label} period ({requested_start} to {requested_end}) at "
                f"all — the dataset only covers {available_min.date()} to {available_max.date()}."
            ),
        }

    covered_days = (overlap_end - overlap_start).days + 1
    if covered_days >= requested_days:
        return None

    coverage_pct = round(covered_days / requested_days * 100, 1)
    return {
        "full_coverage": False,
        "coverage_pct": coverage_pct,
        "note": (
            f"Requested {label} period ({requested_start} to {requested_end}, {requested_days} days) is only "
            f"{coverage_pct}% covered by actual data — the dataset spans {available_min.date()} to "
            f"{available_max.date()}. This period's figures reflect partial data only and should not be "
            f"presented as a full-period comparison without saying so."
        ),
    }


def _agg(series: pd.Series, agg_func: str) -> float | None:
    if series.empty:
        return None
    if agg_func == "sum" and pd.api.types.is_integer_dtype(series):
        # Same real int64-wraparound bug fixed in aggregation.py::group_and_aggregate
        # -- pandas/numpy int64 sum silently overflows to a wildly wrong (often
        # negative) value with no warning. Object-dtype sum forces Python's
        # arbitrary-precision int arithmetic instead.
        return round(float(series.astype(object).sum()), 4)
    return round(float(getattr(series, agg_func)()), 4)


def parse_date_series(series: pd.Series) -> pd.Series:
    """Parse mixed date representations identically for every temporal tool."""
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce")


def _parse_boundary(value: str, label: str) -> pd.Timestamp:
    try:
        parsed = pd.to_datetime(value, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ToolExecutionError(f"Invalid {label} date {value!r}; use an ISO date or timestamp.") from exc
    if pd.isna(parsed):
        raise ToolExecutionError(f"Invalid {label} date {value!r}; use an ISO date or timestamp.")
    return parsed
