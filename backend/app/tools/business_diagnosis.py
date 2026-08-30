from __future__ import annotations

import pandas as pd

from app.tools.anomaly import detect_anomalies
from app.tools.comparison import compare_periods, parse_date_series
from app.tools.errors import ToolExecutionError
from app.tools.filtering import apply_filters

# --- shared validation helpers ---------------------------------------------

_MIN_DISTINCT_DATES_FOR_TREND = 4
_MIN_SPAN_DAYS_FOR_TREND = 2


def _validate_filter_list(filters: list[dict] | None, name: str) -> None:
    if not isinstance(filters, list) or len(filters) == 0:
        raise ToolExecutionError(f"'{name}' must be a non-empty list of filter conditions.")


def _validate_top_n(top_n: int) -> int:
    try:
        top_n_int = int(top_n)
    except (TypeError, ValueError):
        raise ToolExecutionError("top_n must be a positive integer.")
    if top_n_int <= 0:
        raise ToolExecutionError("top_n must be a positive integer.")
    return top_n_int


# --- contribution_analysis ---------------------------------------------------


def contribution_analysis(
    df: pd.DataFrame,
    metric_column: str,
    dimension_column: str,
    current_filters: list[dict],
    baseline_filters: list[dict],
    filters: list[dict] | None = None,
    top_n: int = 10,
) -> dict:
    """Waterfall-style attribution of the CHANGE in `metric_column` (summed —
    the natural operation for "how much did this dimension contribute to the
    total change") between two conditions, broken down per category of
    `dimension_column`.

    `current_filters` and `baseline_filters` are each a list of filter
    conditions in the same shape `apply_filters` accepts (e.g.
    `[{"column": "date", "op": "between", "value": ["2025-01-01", "2025-06-30"]}]`),
    applied independently to the same working frame to define "current" vs
    "baseline". This is how "why did revenue fall this half vs last half" or
    "why did Region A underperform this month vs last month" gets decomposed:
    pass the two halves/periods/segments as `current_filters`/`baseline_filters`
    and `dimension_column` as the thing to attribute the change to (e.g.
    `region`, `product`, `category`).

    `filters` (the existing convention shared with every other tool) is
    applied FIRST, before the current/baseline split — use it to narrow to one
    product line, region, etc. before diagnosing the change within it.

    For each category: `current_value` / `baseline_value` are the sum of
    `metric_column` for rows matching `current_filters`/`baseline_filters`
    AND that category (0.0, not skipped, if the category has no rows in that
    side). `delta = current_value - baseline_value`. Categories present in
    only one side are flagged explicitly via `new_in_current` /
    `absent_in_current` rather than silently treated like any other row.

    Sorted by absolute contribution (`abs(delta)`) descending, capped at
    `top_n`. Categories beyond the cap are bundled into one explicit
    `"other"` bucket (with `category_count` of how many categories it
    represents) so the returned breakdown's deltas still sum exactly to
    `total_delta` — the entire point of a waterfall/attribution breakdown is
    that contributions account for the full total, not just the visible top
    slice.
    """
    working = apply_filters(df, filters) if filters else df
    if working.empty:
        raise ToolExecutionError("No rows match the given filters.")

    for col in (metric_column, dimension_column):
        if col not in working.columns:
            raise ToolExecutionError(f"Unknown column '{col}'.")
    if not pd.api.types.is_numeric_dtype(working[metric_column]):
        raise ToolExecutionError(f"Column '{metric_column}' must be numeric.")

    top_n = _validate_top_n(top_n)
    _validate_filter_list(current_filters, "current_filters")
    _validate_filter_list(baseline_filters, "baseline_filters")

    # Delegate all condition-level validation (unknown column, bad operator,
    # type mismatches) to apply_filters itself rather than reimplementing it.
    current = apply_filters(working, current_filters)
    baseline = apply_filters(working, baseline_filters)
    if current.empty and baseline.empty:
        raise ToolExecutionError(
            "No rows match either current_filters or baseline_filters — nothing to compare."
        )

    current_grouped = (
        current.groupby(dimension_column, dropna=False)[metric_column].sum() if not current.empty else pd.Series(dtype=float)
    )
    baseline_grouped = (
        baseline.groupby(dimension_column, dropna=False)[metric_column].sum() if not baseline.empty else pd.Series(dtype=float)
    )

    # Key by the stringified category (matching every category's final output
    # form) rather than the raw index value — avoids a real pitfall where a
    # NaN/missing category from `current` and one from `baseline` don't
    # compare equal to each other in a Python set, which would silently
    # double the "missing" category instead of merging it into one row.
    current_map = {str(idx): float(val) for idx, val in current_grouped.items()}
    baseline_map = {str(idx): float(val) for idx, val in baseline_grouped.items()}

    total_current = float(current[metric_column].sum()) if not current.empty else 0.0
    total_baseline = float(baseline[metric_column].sum()) if not baseline.empty else 0.0
    total_delta = total_current - total_baseline

    all_categories = set(current_map) | set(baseline_map)
    entries = []
    for cat in all_categories:
        cur_val = current_map.get(cat, 0.0)
        base_val = baseline_map.get(cat, 0.0)
        delta = cur_val - base_val
        in_current = cat in current_map
        in_baseline = cat in baseline_map
        entries.append(
            {
                "category": cat,
                "current_value": round(cur_val, 4),
                "baseline_value": round(base_val, 4),
                "delta": round(delta, 4),
                "pct_of_total_delta": round(delta / total_delta * 100, 2) if total_delta else None,
                "new_in_current": in_current and not in_baseline,
                "absent_in_current": in_baseline and not in_current,
            }
        )

    entries.sort(key=lambda e: abs(e["delta"]), reverse=True)
    top_entries = entries[:top_n]
    remainder = entries[top_n:]

    breakdown = list(top_entries)
    if remainder:
        other_current = sum(e["current_value"] for e in remainder)
        other_baseline = sum(e["baseline_value"] for e in remainder)
        other_delta = other_current - other_baseline
        breakdown.append(
            {
                "category": "other",
                "current_value": round(other_current, 4),
                "baseline_value": round(other_baseline, 4),
                "delta": round(other_delta, 4),
                "pct_of_total_delta": round(other_delta / total_delta * 100, 2) if total_delta else None,
                "new_in_current": False,
                "absent_in_current": False,
                "category_count": len(remainder),
            }
        )

    return {
        "metric_column": metric_column,
        "dimension_column": dimension_column,
        "total_current_value": round(total_current, 4),
        "total_baseline_value": round(total_baseline, 4),
        "total_delta": round(total_delta, 4),
        "category_count": len(all_categories),
        "top_n": top_n,
        "breakdown": breakdown,
    }


# --- executive_summary --------------------------------------------------------


def _trend_for_metric(
    working: pd.DataFrame,
    date_column: str,
    metric_column: str,
    current_start: str,
    current_end: str,
    previous_start: str,
    previous_end: str,
) -> dict:
    """Compare one metric through the shared explicit period contract."""
    result = compare_periods(
        working,
        date_column=date_column,
        value_column=metric_column,
        current_start=current_start,
        current_end=current_end,
        previous_start=previous_start,
        previous_end=previous_end,
        agg_func="sum",
    )
    delta = result["delta"] or 0.0
    direction = "up" if delta > 0 else ("down" if delta < 0 else "flat")
    trend = {
        "available": True,
        "current_period_value": result["current_period"]["value"],
        "previous_period_value": result["previous_period"]["value"],
        "delta": result["delta"],
        "pct_change": result["pct_change"],
        "direction": direction,
    }
    warning = result["current_period_coverage_warning"] or result["previous_period_coverage_warning"]
    if warning is not None:
        trend["coverage_warning"] = warning["note"]
    return trend


def executive_summary(
    df: pd.DataFrame,
    metrics: list[str],
    filters: list[dict] | None = None,
    date_column: str | None = None,
    current_start: str | None = None,
    current_end: str | None = None,
    previous_start: str | None = None,
    previous_end: str | None = None,
) -> dict:
    """Bounded, structured KPI summary for an executive audience: for each
    column in `metrics`, reports total/mean/count, plus (only when
    `date_column` is given and the data has enough date range) a
    period-over-period trend computed via `compare_periods` (its date-math is
    reused, not reimplemented), and whether `detect_anomalies` finds outliers
    worth flagging. When all four explicit boundaries are supplied, those exact
    windows are used; otherwise the backward-compatible recent-half vs earlier-
    half fallback is used.

    Trend eligibility (deterministic, documented, not silent): requires at
    least `_MIN_DISTINCT_DATES_FOR_TREND` (4) distinct dates and a total span
    of at least `_MIN_SPAN_DAYS_FOR_TREND` (2) days; below that, `trend` is
    returned as `{"available": False, "reason": ...}` for every metric rather
    than a misleading comparison built from a near-empty split. The split
    point is the midpoint of the actual [min_date, max_date] time range in
    the (filtered) data — not row-count-based — so both halves are backed by
    real calendar time.

    No raw row data is ever included (this project has a documented 413
    "payload too large" history traced to verbose tool outputs — see
    `insights.py`); anomaly detail is reduced to a count/pct flag, never the
    underlying flagged rows.
    """
    working = apply_filters(df, filters) if filters else df
    if working.empty:
        raise ToolExecutionError("No rows match the given filters.")

    if not isinstance(metrics, list) or len(metrics) == 0:
        raise ToolExecutionError("metrics must be a non-empty list of column names.")
    for m in metrics:
        if m not in working.columns:
            raise ToolExecutionError(f"Unknown column '{m}'.")
        if not pd.api.types.is_numeric_dtype(working[m]):
            raise ToolExecutionError(f"Column '{m}' must be numeric.")

    trend_supported = False
    trend_reason = None
    min_date = max_date = midpoint = None
    explicit_periods = (current_start, current_end, previous_start, previous_end)
    if any(value is not None for value in explicit_periods) and date_column is None:
        raise ToolExecutionError("Explicit period boundaries require date_column.")
    if any(value is not None for value in explicit_periods) and not all(
        value is not None for value in explicit_periods
    ):
        raise ToolExecutionError(
            "current_start, current_end, previous_start, and previous_end must be provided together."
        )
    if date_column is not None:
        if date_column not in working.columns:
            raise ToolExecutionError(f"Unknown column '{date_column}'.")
        parsed_dates = parse_date_series(working[date_column])
        if parsed_dates.notna().sum() == 0:
            raise ToolExecutionError(f"Column '{date_column}' does not contain parseable dates.")

        min_date = parsed_dates.min()
        max_date = parsed_dates.max()
        distinct_dates = parsed_dates.dropna().nunique()
        span_days = (max_date - min_date).days

        if all(value is not None for value in explicit_periods):
            trend_supported = True
        elif distinct_dates >= _MIN_DISTINCT_DATES_FOR_TREND and span_days >= _MIN_SPAN_DAYS_FOR_TREND:
            trend_supported = True
            midpoint = min_date + (max_date - min_date) / 2
        else:
            trend_reason = (
                f"Insufficient date range to compute a meaningful period-over-period split: "
                f"found {distinct_dates} distinct date(s) spanning {span_days} day(s); need at least "
                f"{_MIN_DISTINCT_DATES_FOR_TREND} distinct dates and a {_MIN_SPAN_DAYS_FOR_TREND}-day span."
            )

    kpis = []
    for m in metrics:
        series = working[m].dropna()
        kpi: dict = {
            "metric": m,
            "total": round(float(series.sum()), 4) if len(series) else None,
            "mean": round(float(series.mean()), 4) if len(series) else None,
            "count": int(series.count()),
        }

        if date_column is not None:
            if trend_supported:
                if current_start is None:
                    current_start = midpoint.isoformat()
                    current_end = max_date.isoformat()
                    previous_start = min_date.isoformat()
                    previous_end = (midpoint - pd.Timedelta(microseconds=1)).isoformat()
                kpi["trend"] = _trend_for_metric(
                    working,
                    date_column,
                    m,
                    current_start,
                    current_end,
                    previous_start,
                    previous_end,
                )
            else:
                kpi["trend"] = {"available": False, "reason": trend_reason}

        anomaly_flag: dict | None
        try:
            anomaly_result = detect_anomalies(working, m, method="iqr")
            anomaly_flag = {
                "flagged": anomaly_result["anomaly_count"] > 0,
                "anomaly_count": anomaly_result["anomaly_count"],
                "anomaly_pct": anomaly_result["anomaly_pct"],
            }
        except ToolExecutionError:
            # Not enough data points to run outlier detection on this
            # metric — not an error for the summary as a whole, just no
            # anomaly signal available for this one KPI.
            anomaly_flag = None
        kpi["anomalies"] = anomaly_flag

        kpis.append(kpi)

    return {
        "row_count": int(len(working)),
        "date_column": date_column,
        "date_range": (
            {"min": min_date.strftime("%Y-%m-%d"), "max": max_date.strftime("%Y-%m-%d")}
            if date_column is not None
            else None
        ),
        "trend_supported": trend_supported if date_column is not None else None,
        "metrics": kpis,
    }
