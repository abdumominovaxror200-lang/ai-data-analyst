from __future__ import annotations

import pandas as pd
import pytest

from app.tools.business_diagnosis import executive_summary
from app.tools.comparison import compare_periods
from app.tools.errors import ToolExecutionError
from app.tools.hypothesis import t_test


@pytest.fixture
def two_year_sales() -> pd.DataFrame:
    dates_2024 = pd.date_range("2024-01-31 12:00:00", periods=12, freq="ME")
    dates_2025 = pd.date_range("2025-01-31 12:00:00", periods=12, freq="ME")
    return pd.DataFrame(
        {
            "date": dates_2024.append(dates_2025),
            "year": [2024] * 12 + [2025] * 12,
            "revenue": [100.0] * 12 + [110.0] * 12,
            "profit": [20.0] * 12 + [22.0] * 12,
        }
    )


def test_explicit_year_windows_match_between_comparison_and_executive_summary(two_year_sales):
    boundaries = {
        "current_start": "2025-01-01",
        "current_end": "2025-12-31",
        "previous_start": "2024-01-01",
        "previous_end": "2024-12-31",
    }
    summary = executive_summary(
        two_year_sales,
        ["revenue", "profit"],
        date_column="date",
        **boundaries,
    )

    for metric in ("revenue", "profit"):
        comparison = compare_periods(
            two_year_sales,
            date_column="date",
            value_column=metric,
            agg_func="sum",
            **boundaries,
        )
        trend = next(item["trend"] for item in summary["metrics"] if item["metric"] == metric)
        assert trend["current_period_value"] == comparison["current_period"]["value"]
        assert trend["previous_period_value"] == comparison["previous_period"]["value"]
        assert trend["pct_change"] == comparison["pct_change"] == 10.0


def test_tiny_temporal_groups_return_typed_limitation_instead_of_p_value():
    frame = pd.DataFrame(
        {
            "year": [2024] * 4 + [2025] * 5,
            "revenue": [100.0, 101.0, 99.0, 100.0, 110.0, 111.0, 109.0, 110.0, 112.0],
        }
    )

    with pytest.raises(ToolExecutionError, match="No p-value was computed"):
        t_test(frame, column="revenue", group_column="year", group_a=2025, group_b=2024)
