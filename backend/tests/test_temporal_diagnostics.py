"""Generic deterministic Phase B tool tests; no provider or benchmark data."""

import numpy as np
import pandas as pd
from scipy import stats

from app.tools.temporal_diagnostics import (
    compare_periods_inference,
    localized_period_change,
    period_outlier_sensitivity,
)


def _dated(values_a, values_b, segments_a=None, segments_b=None):
    n_a, n_b = len(values_a), len(values_b)
    return pd.DataFrame({
        "event_date": ["2026-02-01"] * n_a + ["2025-02-01"] * n_b,
        "duration": list(values_a) + list(values_b),
        "channel": (segments_a or ["web"] * n_a) + (segments_b or ["web"] * n_b),
    })


def test_welch_two_period_result_matches_scipy_fixture():
    current = np.array([13, 14, 15, 16, 17, 18, 12, 19, 20, 11, 16, 17], dtype=float)
    previous = np.array([9, 10, 11, 12, 13, 8, 10, 11, 9, 12, 10, 11], dtype=float)
    result = compare_periods_inference(
        _dated(current, previous), "event_date", "duration",
        "2026-01-01", "2026-06-30", "2025-01-01", "2025-06-30",
    )
    expected = stats.ttest_ind(current, previous, equal_var=False)
    assert result["test"] == "welch_two_sample_t_test"
    assert result["current"]["n"] == result["previous"]["n"] == 12
    assert result["mean_difference"] == round(float(current.mean() - previous.mean()), 6)
    assert result["p_value"] == round(float(expected.pvalue), 10)
    ci = result["difference_confidence_interval"]
    assert ci["lower"] < result["mean_difference"] < ci["upper"]
    assert "cohens_d" in result["effect_size"]


def test_localized_change_ranks_valid_segments_and_types_tiny_group_limit():
    frame = _dated(
        [30] * 12 + [11] * 12 + [99] * 2,
        [10] * 12 + [10] * 12 + [1] * 2,
        ["phone"] * 12 + ["web"] * 12 + ["rare"] * 2,
        ["phone"] * 12 + ["web"] * 12 + ["rare"] * 2,
    )
    result = localized_period_change(
        frame, "event_date", "duration", "channel",
        "2026-01-01", "2026-06-30", "2025-01-01", "2025-06-30",
    )
    assert result["segments"][0]["segment"] == "phone"
    assert result["segments"][0]["delta"] == 20
    assert all(row["segment"] != "rare" for row in result["segments"])
    assert result["limitations"] == [dict(result["limitations"][0], code="small_sample")]
    assert result["causal_interpretation"] == "not_supported"


def test_outlier_sensitivity_reports_raw_robust_and_every_exclusion():
    current = [20.0] * 20 + [1000.0]
    previous = [10.0] * 21
    frame = _dated(current, previous)
    original_rows = len(frame)
    result = period_outlier_sensitivity(
        frame, "event_date", "duration",
        "2026-01-01", "2026-06-30", "2025-01-01", "2025-06-30",
    )
    assert len(frame) == original_rows
    assert result["raw"]["n_current"] == 21
    assert result["robust"]["n_current"] == 20
    assert result["excluded"] == {"current": 1, "previous": 0, "total": 1}
    assert result["raw"]["delta"] != result["robust"]["delta"]
    assert result["rule"]["method"] == "pooled_iqr_fence"
