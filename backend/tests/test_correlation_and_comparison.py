from __future__ import annotations

import pandas as pd
import pytest

from app.tools.comparison import compare_periods
from app.tools.correlation import correlation_analysis
from app.tools.errors import ToolExecutionError


def test_correlation_analysis_perfect_positive_correlation():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]})
    result = correlation_analysis(df)
    assert result["strongest_pairs"][0]["correlation"] == pytest.approx(1.0)


def test_correlation_analysis_requires_two_numeric_columns():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ToolExecutionError):
        correlation_analysis(df)


def test_compare_periods_computes_delta_and_pct_change():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "revenue": [100, 100, 200, 200],
        }
    )
    result = compare_periods(
        df,
        date_column="date",
        value_column="revenue",
        current_start="2024-01-03",
        current_end="2024-01-04",
        previous_start="2024-01-01",
        previous_end="2024-01-02",
        agg_func="sum",
    )
    assert result["current_period"]["value"] == 400
    assert result["previous_period"]["value"] == 200
    assert result["delta"] == 200
    assert result["pct_change"] == 100.0


def test_compare_periods_unknown_column_raises():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=2), "revenue": [1, 2]})
    with pytest.raises(ToolExecutionError):
        compare_periods(
            df,
            date_column="nope",
            value_column="revenue",
            current_start="2024-01-01",
            current_end="2024-01-02",
            previous_start="2024-01-01",
            previous_end="2024-01-02",
        )
