"""Unit tests for app.tools.mix_decomposition (Mix Decomposition Engine mission).

All fixtures here are independently, freshly authored for this test file --
none reuse any Blind Benchmark dataset, question, or hidden ground-truth value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.reasoning.causation_guard import find_causal_phrases
from app.tools.errors import ToolExecutionError
from app.tools.mix_decomposition import mix_decomposition


def _rows(segment: str, n: int, mean: float, period: str, std: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash((segment, period, n))) % (2**32))
    values = rng.normal(mean, std, n) if std else np.full(n, mean)
    dates = ["2024-01-15"] * n if period == "previous" else ["2024-06-15"] * n
    return pd.DataFrame({"date": dates, "segment": [segment] * n, "value": values})


def _build(rows: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(rows, ignore_index=True)


_PREV = ("2024-01-01", "2024-01-31")
_CUR = ("2024-06-01", "2024-06-30")


def _call(df, **overrides):
    kwargs = dict(
        date_column="date", value_column="value", segment_column="segment",
        current_start=_CUR[0], current_end=_CUR[1], previous_start=_PREV[0], previous_end=_PREV[1],
    )
    kwargs.update(overrides)
    return mix_decomposition(df, **kwargs)


# --- pure mix-shift case: means unchanged, only the mix shifts -----------------------


def test_pure_mix_shift_attributes_the_entire_change_to_mix_effect():
    df = _build([
        _rows("A", 80, 10.0, "previous"), _rows("B", 20, 20.0, "previous"),
        _rows("A", 20, 10.0, "current"), _rows("B", 80, 20.0, "current"),
    ])
    result = _call(df)
    assert result["components"]["mix_effect"] == pytest.approx(6.0, abs=1e-6)
    assert result["components"]["within_segment_effect"] == pytest.approx(0.0, abs=1e-6)
    assert result["components"]["interaction_or_residual"] == pytest.approx(0.0, abs=1e-6)
    assert result["decomposition_overall_change"]["delta"] == pytest.approx(6.0, abs=1e-6)
    assert result["reconciliation"]["reconciles"] is True


# --- pure within-segment case: mix unchanged, only each segment's own mean changes ----


def test_pure_within_segment_change_attributes_the_entire_change_to_within_effect():
    df = _build([
        _rows("A", 50, 10.0, "previous"), _rows("B", 50, 20.0, "previous"),
        _rows("A", 50, 14.0, "current"), _rows("B", 50, 24.0, "current"),
    ])
    result = _call(df)
    assert result["components"]["mix_effect"] == pytest.approx(0.0, abs=1e-6)
    assert result["components"]["within_segment_effect"] == pytest.approx(4.0, abs=1e-6)
    assert result["components"]["interaction_or_residual"] == pytest.approx(0.0, abs=1e-6)
    assert result["decomposition_overall_change"]["delta"] == pytest.approx(4.0, abs=1e-6)


# --- mixed case: both mix and rates change -> a real, nonzero interaction term -------


def test_combined_mix_and_within_change_produces_a_nonzero_interaction_that_still_reconciles():
    df = _build([
        _rows("A", 80, 10.0, "previous"), _rows("B", 20, 20.0, "previous"),
        _rows("A", 20, 15.0, "current"), _rows("B", 80, 20.0, "current"),
    ])
    result = _call(df)
    comps = result["components"]
    assert comps["mix_effect"] == pytest.approx(6.0, abs=1e-6)
    assert comps["within_segment_effect"] == pytest.approx(4.0, abs=1e-6)
    assert comps["interaction_or_residual"] == pytest.approx(-3.0, abs=1e-6)
    assert result["decomposition_overall_change"]["delta"] == pytest.approx(7.0, abs=1e-6)
    # the exact identity: components sum to the observed change, within tolerance
    total = comps["mix_effect"] + comps["within_segment_effect"] + comps["interaction_or_residual"]
    assert total == pytest.approx(result["reconciliation"]["observed_change"], abs=1e-6)
    assert result["reconciliation"]["reconciles"] is True


def test_reconciliation_diff_is_reported_and_within_documented_tolerance():
    df = _build([
        _rows("A", 30, 5.0, "previous"), _rows("B", 30, 8.0, "previous"), _rows("C", 30, 12.0, "previous"),
        _rows("A", 40, 6.0, "current"), _rows("B", 20, 9.0, "current"), _rows("C", 40, 11.0, "current"),
    ])
    result = _call(df)
    assert abs(result["reconciliation"]["difference"]) <= result["reconciliation"]["tolerance"]
    assert result["reconciliation"]["reconciles"] is True


# --- missing segment values ----------------------------------------------------------


def test_rows_with_missing_segment_value_are_excluded_and_counted():
    df = _build([_rows("A", 60, 10.0, "previous"), _rows("B", 60, 20.0, "previous"), _rows("A", 60, 12.0, "current"), _rows("B", 60, 22.0, "current")])
    # inject some missing-segment rows into both periods
    missing_prev = pd.DataFrame({"date": ["2024-01-10"] * 7, "segment": [None] * 7, "value": [99.0] * 7})
    missing_cur = pd.DataFrame({"date": ["2024-06-10"] * 4, "segment": [None] * 4, "value": [99.0] * 4})
    df = pd.concat([df, missing_prev, missing_cur], ignore_index=True)

    result = _call(df)
    assert result["sample_counts"]["n_previous_missing_segment_excluded"] == 7
    assert result["sample_counts"]["n_current_missing_segment_excluded"] == 4
    # the 99.0 values must never leak into any segment's mean
    for seg_row in result["segments"]:
        assert seg_row["mean_previous"] < 90
        assert seg_row["mean_current"] < 90


# --- tiny groups excluded --------------------------------------------------------------


def test_a_segment_below_the_minimum_group_size_is_excluded_not_fabricated():
    df = _build([
        _rows("A", 50, 10.0, "previous"), _rows("B", 50, 20.0, "previous"), _rows("tiny", 2, 999.0, "previous"),
        _rows("A", 50, 11.0, "current"), _rows("B", 50, 21.0, "current"), _rows("tiny", 3, 999.0, "current"),
    ])
    result = _call(df)
    segment_names = {s["segment"] for s in result["segments"]}
    assert "tiny" not in segment_names
    excluded_names = {e["segment"]: e for e in result["excluded_segments"]}
    assert "tiny" in excluded_names
    assert "minimum group size" in excluded_names["tiny"]["reason"]


def test_a_segment_present_in_only_one_period_is_excluded_with_a_clear_reason():
    df = _build([
        _rows("A", 50, 10.0, "previous"), _rows("B", 50, 20.0, "previous"), _rows("only_prev", 20, 5.0, "previous"),
        _rows("A", 50, 11.0, "current"), _rows("B", 50, 21.0, "current"), _rows("only_cur", 20, 30.0, "current"),
    ])
    result = _call(df)
    excluded_names = {e["segment"]: e["reason"] for e in result["excluded_segments"]}
    assert "absent in current period" in excluded_names["only_prev"]
    assert "absent in previous period" in excluded_names["only_cur"]


# --- no common segment support at all --------------------------------------------------


def test_no_common_segment_support_raises():
    df = _build([_rows("X", 30, 10.0, "previous"), _rows("Y", 30, 15.0, "current")])
    with pytest.raises(ToolExecutionError, match="No common segment support"):
        _call(df)


# --- insufficient total observations ----------------------------------------------------


def test_insufficient_total_observations_raises():
    df = _build([_rows("A", 3, 10.0, "previous"), _rows("A", 3, 12.0, "current")])
    with pytest.raises(ToolExecutionError, match="Insufficient observations"):
        _call(df)


# --- invalid inputs ----------------------------------------------------------------------


def test_invalid_date_raises():
    df = _build([_rows("A", 20, 10.0, "previous"), _rows("A", 20, 12.0, "current")])
    with pytest.raises(ToolExecutionError, match="Invalid"):
        _call(df, current_start="not-a-date")


def test_start_after_end_raises():
    df = _build([_rows("A", 20, 10.0, "previous"), _rows("A", 20, 12.0, "current")])
    with pytest.raises(ToolExecutionError, match="after its end"):
        _call(df, current_start="2024-06-30", current_end="2024-06-01")


def test_unknown_column_raises():
    df = _build([_rows("A", 20, 10.0, "previous"), _rows("A", 20, 12.0, "current")])
    with pytest.raises(ToolExecutionError, match="Unknown column"):
        _call(df, segment_column="does_not_exist")


def test_non_numeric_value_column_raises():
    df = _build([_rows("A", 20, 10.0, "previous"), _rows("A", 20, 12.0, "current")])
    df["value"] = df["value"].astype(str)
    with pytest.raises(ToolExecutionError, match="must be numeric"):
        _call(df)


# --- never claims causation -------------------------------------------------------------


def test_result_is_explicitly_labeled_descriptive_not_causal():
    df = _build([_rows("A", 50, 10.0, "previous"), _rows("B", 50, 20.0, "previous"), _rows("A", 50, 12.0, "current"), _rows("B", 50, 22.0, "current")])
    result = _call(df)
    assert result["causal_interpretation"] == "not_supported"
    assert find_causal_phrases(result["disclaimer"]) == []
    for assumption in result["assumptions"]:
        assert find_causal_phrases(assumption) == []
    for excluded in result["excluded_segments"]:
        assert find_causal_phrases(excluded["reason"]) == []


# --- coverage reporting when some segments are excluded --------------------------------


def test_full_sample_and_decomposition_overall_change_differ_when_coverage_is_partial():
    df = _build([
        _rows("A", 50, 10.0, "previous"), _rows("B", 50, 20.0, "previous"), _rows("only_prev", 30, 100.0, "previous"),
        _rows("A", 50, 11.0, "current"), _rows("B", 50, 21.0, "current"),
    ])
    result = _call(df)
    assert result["sample_counts"]["coverage_pct_previous"] < 100.0
    assert result["sample_counts"]["coverage_pct_current"] == 100.0
    assert result["full_sample_overall_change"]["delta"] != result["decomposition_overall_change"]["delta"]


def test_filters_are_applied_before_period_slicing():
    df = _build([
        _rows("A", 50, 10.0, "previous"), _rows("B", 50, 999.0, "previous"),
        _rows("A", 50, 12.0, "current"), _rows("B", 50, 999.0, "current"),
    ])
    result = _call(df, filters=[{"column": "segment", "op": "==", "value": "A"}])
    segment_names = {s["segment"] for s in result["segments"]}
    assert segment_names == {"A"}
