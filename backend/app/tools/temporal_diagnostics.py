"""Deterministic, scope-explicit diagnostics for two-period questions."""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from scipy import stats

from app.tools.comparison import parse_date_series
from app.tools.errors import ToolExecutionError
from app.tools.filtering import apply_filters
from app.tools.hypothesis import _ensure_finite_variance

_MIN_INFERENCE_N = 10
_MIN_SEGMENT_N = 10


def _period_samples(df, date_column, value_column, current_start, current_end, previous_start, previous_end, filters):
    working = apply_filters(df, filters) if filters else df
    for column in (date_column, value_column):
        if column not in working.columns:
            raise ToolExecutionError(f"Unknown column '{column}'.")
    if not pd.api.types.is_numeric_dtype(working[value_column]):
        raise ToolExecutionError(f"Column '{value_column}' must be numeric.")
    dates = parse_date_series(working[date_column])

    def sample(start, end):
        lo, hi = pd.to_datetime(start), pd.to_datetime(end)
        if pd.isna(lo) or pd.isna(hi) or lo > hi:
            raise ToolExecutionError(f"Invalid period boundaries: {start!r} to {end!r}.")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(end)):
            mask = (dates >= lo) & (dates < hi + pd.Timedelta(days=1))
        else:
            mask = (dates >= lo) & (dates <= hi)
        return working.loc[mask, value_column].dropna().astype(float), working.loc[mask]

    current, current_rows = sample(current_start, current_end)
    previous, previous_rows = sample(previous_start, previous_end)
    if current.empty or previous.empty:
        raise ToolExecutionError("Both periods require non-missing observations.")
    scope = {
        "current_period": {"start": current_start, "end": current_end},
        "previous_period": {"start": previous_start, "end": previous_end},
        "filters": filters or [],
    }
    return current, previous, current_rows, previous_rows, scope


def compare_periods_inference(
    df: pd.DataFrame, date_column: str, value_column: str,
    current_start: str, current_end: str, previous_start: str, previous_end: str,
    confidence: float = 0.95, filters: list[dict] | None = None,
) -> dict:
    """Welch inference for independent row-level units in two explicit periods."""
    if not 0 < confidence < 1:
        raise ToolExecutionError("confidence must be between 0 and 1.")
    current, previous, _cr, _pr, scope = _period_samples(
        df, date_column, value_column, current_start, current_end, previous_start, previous_end, filters
    )
    if len(current) < _MIN_INFERENCE_N or len(previous) < _MIN_INFERENCE_N:
        raise ToolExecutionError(
            f"Two-period inference requires at least {_MIN_INFERENCE_N} independent rows per period "
            f"(current n={len(current)}, previous n={len(previous)}). No p-value was computed."
        )
    _ensure_finite_variance(current, previous)
    mean_current, mean_previous = float(current.mean()), float(previous.mean())
    difference = mean_current - mean_previous
    var_current, var_previous = float(current.var(ddof=1)), float(previous.var(ddof=1))
    se2 = var_current / len(current) + var_previous / len(previous)
    se = math.sqrt(se2)
    dof = se2**2 / (
        (var_current / len(current)) ** 2 / (len(current) - 1)
        + (var_previous / len(previous)) ** 2 / (len(previous) - 1)
    ) if se2 else float("inf")
    statistic, p_value = stats.ttest_ind(current, previous, equal_var=False)
    critical = stats.t.ppf((1 + confidence) / 2, dof) if se else 0.0
    pooled = math.sqrt(
        ((len(current) - 1) * var_current + (len(previous) - 1) * var_previous)
        / (len(current) + len(previous) - 2)
    )
    d = difference / pooled if pooled else 0.0
    return {
        "test": "welch_two_sample_t_test",
        "analysis_unit": "row",
        "metric": value_column,
        **scope,
        "current": {"n": len(current), "mean": round(mean_current, 6)},
        "previous": {"n": len(previous), "mean": round(mean_previous, 6)},
        "mean_difference": round(difference, 6),
        "difference_confidence_interval": {
            "confidence": confidence,
            "lower": round(difference - critical * se, 6),
            "upper": round(difference + critical * se, 6),
        },
        "statistic": round(float(statistic), 6),
        "p_value": round(float(p_value), 10),
        "significant": bool(p_value < (1 - confidence)),
        "degrees_of_freedom": round(float(dof), 6),
        "effect_size": {"cohens_d": round(d, 6), "magnitude": _magnitude(d)},
        "causal_interpretation": "not_supported",
    }


def localized_period_change(
    df: pd.DataFrame, date_column: str, value_column: str, segment_column: str,
    current_start: str, current_end: str, previous_start: str, previous_end: str,
    filters: list[dict] | None = None,
) -> dict:
    current, previous, current_rows, previous_rows, scope = _period_samples(
        df, date_column, value_column, current_start, current_end, previous_start, previous_end, filters
    )
    if segment_column not in df.columns:
        raise ToolExecutionError(f"Unknown column '{segment_column}'.")
    current_rows = current_rows.dropna(subset=[segment_column, value_column])
    previous_rows = previous_rows.dropna(subset=[segment_column, value_column])
    current_stats = current_rows.groupby(segment_column)[value_column].agg(["count", "mean"])
    previous_stats = previous_rows.groupby(segment_column)[value_column].agg(["count", "mean"])
    rows, limitations = [], []
    for segment in sorted(set(current_stats.index) | set(previous_stats.index), key=str):
        nc = int(current_stats.loc[segment, "count"]) if segment in current_stats.index else 0
        np_ = int(previous_stats.loc[segment, "count"]) if segment in previous_stats.index else 0
        if nc < _MIN_SEGMENT_N or np_ < _MIN_SEGMENT_N:
            limitations.append({
                "code": "small_sample", "severity": "reduces_confidence", "segment": str(segment),
                "n_current": nc, "n_previous": np_,
                "message": f"Segment has fewer than {_MIN_SEGMENT_N} observations in at least one period and is not ranked.",
            })
            continue
        mc, mp = float(current_stats.loc[segment, "mean"]), float(previous_stats.loc[segment, "mean"])
        rows.append({
            "segment": str(segment), "n_current": nc, "n_previous": np_,
            "current_mean": round(mc, 6), "previous_mean": round(mp, 6),
            "delta": round(mc - mp, 6),
        })
    rows.sort(key=lambda item: (-abs(item["delta"]), item["segment"]))
    for rank, row in enumerate(rows, 1):
        row["rank_by_absolute_change"] = rank
    return {
        "metric": value_column, "segment_dimension": segment_column, **scope,
        "segments": rows, "limitations": limitations,
        "ranking_rule": "absolute current-minus-previous mean change; adequately sized common segments only",
        "causal_interpretation": "not_supported",
        "disclaimer": "This localizes descriptive change and does not establish why a segment changed.",
    }


def period_outlier_sensitivity(
    df: pd.DataFrame, date_column: str, value_column: str,
    current_start: str, current_end: str, previous_start: str, previous_end: str,
    iqr_multiplier: float = 1.5, materiality_threshold: float = 0.25,
    filters: list[dict] | None = None,
) -> dict:
    if iqr_multiplier <= 0 or not 0 <= materiality_threshold <= 1:
        raise ToolExecutionError("iqr_multiplier must be positive and materiality_threshold must be in [0, 1].")
    current, previous, _cr, _pr, scope = _period_samples(
        df, date_column, value_column, current_start, current_end, previous_start, previous_end, filters
    )
    combined = pd.concat([current, previous], ignore_index=True)
    q1, q3 = float(combined.quantile(.25)), float(combined.quantile(.75))
    iqr = q3 - q1
    lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
    current_kept = current.between(lower, upper)
    previous_kept = previous.between(lower, upper)
    robust_current, robust_previous = current[current_kept], previous[previous_kept]
    if robust_current.empty or robust_previous.empty:
        raise ToolExecutionError("The explicit IQR rule removed every observation from at least one period.")
    raw_delta = float(current.mean() - previous.mean())
    robust_delta = float(robust_current.mean() - robust_previous.mean())
    relative_change = abs(robust_delta - raw_delta) / max(abs(raw_delta), 1e-12)
    direction_changed = np.sign(raw_delta) != np.sign(robust_delta)
    return {
        "metric": value_column, **scope,
        "rule": {"method": "pooled_iqr_fence", "iqr_multiplier": iqr_multiplier, "lower": round(lower, 6), "upper": round(upper, 6)},
        "raw": {"n_current": len(current), "n_previous": len(previous), "current_mean": round(float(current.mean()), 6), "previous_mean": round(float(previous.mean()), 6), "delta": round(raw_delta, 6)},
        "robust": {"n_current": len(robust_current), "n_previous": len(robust_previous), "current_mean": round(float(robust_current.mean()), 6), "previous_mean": round(float(robust_previous.mean()), 6), "delta": round(robust_delta, 6)},
        "excluded": {"current": int((~current_kept).sum()), "previous": int((~previous_kept).sum()), "total": int((~current_kept).sum() + (~previous_kept).sum())},
        "materiality_threshold": materiality_threshold,
        "materially_changes_conclusion": bool(direction_changed or relative_change > materiality_threshold),
        "causal_interpretation": "not_supported",
    }


def _magnitude(value: float) -> str:
    absolute = abs(value)
    return "large" if absolute >= .8 else "medium" if absolute >= .5 else "small" if absolute >= .2 else "negligible"
