from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.tools.business_diagnosis import contribution_analysis, executive_summary
from app.tools.errors import ToolExecutionError

DEMO_XLSX = Path(__file__).resolve().parents[2] / "data" / "demo" / "sales_data.xlsx"


# --- contribution_analysis: known ground truth ------------------------------


def _period_revenue_df() -> pd.DataFrame:
    """Jan 2024 (baseline) vs Feb 2024 (current) revenue by region, with a
    hand-computable breakdown:

    North:  baseline 60+40=100 -> current 150   -> delta +50
    South:  baseline 200       -> current 150   -> delta -50
    East:   baseline 0 (absent)-> current 80     -> delta +80  (new_in_current)
    West:   baseline 30        -> current 0 (absent) -> delta -30 (absent_in_current)

    total_baseline = 330, total_current = 380, total_delta = 50
    (sanity check: 50 - 50 + 80 - 30 == 50)
    """
    rows = [
        {"date": "2024-01-05", "region": "North", "revenue": 60},
        {"date": "2024-01-15", "region": "North", "revenue": 40},
        {"date": "2024-02-10", "region": "North", "revenue": 150},
        {"date": "2024-01-10", "region": "South", "revenue": 200},
        {"date": "2024-02-05", "region": "South", "revenue": 150},
        {"date": "2024-02-20", "region": "East", "revenue": 80},
        {"date": "2024-01-20", "region": "West", "revenue": 30},
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


_CURRENT_FILTERS = [{"column": "date", "op": "between", "value": ["2024-02-01", "2024-02-28"]}]
_BASELINE_FILTERS = [{"column": "date", "op": "between", "value": ["2024-01-01", "2024-01-31"]}]


def test_contribution_analysis_matches_hand_computed_breakdown():
    df = _period_revenue_df()
    result = contribution_analysis(df, "revenue", "region", _CURRENT_FILTERS, _BASELINE_FILTERS)

    assert result["total_current_value"] == pytest.approx(380)
    assert result["total_baseline_value"] == pytest.approx(330)
    assert result["total_delta"] == pytest.approx(50)
    assert result["category_count"] == 4

    by_cat = {e["category"]: e for e in result["breakdown"]}
    assert by_cat["North"]["delta"] == pytest.approx(50)
    assert by_cat["North"]["current_value"] == pytest.approx(150)
    assert by_cat["North"]["baseline_value"] == pytest.approx(100)
    assert by_cat["North"]["new_in_current"] is False
    assert by_cat["North"]["absent_in_current"] is False

    assert by_cat["South"]["delta"] == pytest.approx(-50)

    assert by_cat["East"]["delta"] == pytest.approx(80)
    assert by_cat["East"]["baseline_value"] == pytest.approx(0)
    assert by_cat["East"]["new_in_current"] is True
    assert by_cat["East"]["absent_in_current"] is False

    assert by_cat["West"]["delta"] == pytest.approx(-30)
    assert by_cat["West"]["current_value"] == pytest.approx(0)
    assert by_cat["West"]["absent_in_current"] is True
    assert by_cat["West"]["new_in_current"] is False

    # The whole point of a waterfall breakdown: contributions must sum to the total.
    assert sum(e["delta"] for e in result["breakdown"]) == pytest.approx(result["total_delta"])

    # Sorted by absolute contribution descending.
    abs_deltas = [abs(e["delta"]) for e in result["breakdown"]]
    assert abs_deltas == sorted(abs_deltas, reverse=True)
    assert result["breakdown"][0]["category"] == "East"  # abs(80) is the largest


def test_contribution_analysis_filters_narrows_before_split():
    df = _period_revenue_df()
    # Narrow to North only before splitting into current/baseline -> East/South/West
    # must not appear at all.
    result = contribution_analysis(
        df,
        "revenue",
        "region",
        _CURRENT_FILTERS,
        _BASELINE_FILTERS,
        filters=[{"column": "region", "op": "==", "value": "North"}],
    )
    assert result["category_count"] == 1
    assert result["breakdown"][0]["category"] == "North"
    assert result["total_delta"] == pytest.approx(50)


def test_contribution_analysis_top_n_bundles_remainder_into_other_and_preserves_total():
    """5 categories with distinct |delta| (100, 80, 60, 40, 20) so ranking is
    unambiguous. top_n=3 keeps cat1/cat2/cat3, bundles cat4/cat5 into "other" —
    and the "other" bucket's delta must equal the sum of what it bundles, so the
    full breakdown (top 3 + other) still sums to total_delta."""
    rows = []
    # cat1: new in current, baseline 0 -> current 100 => delta +100
    rows.append({"period": "current", "cat": "cat1", "value": 100})
    # cat2: baseline 180 -> current 100 => delta -80
    rows.append({"period": "baseline", "cat": "cat2", "value": 180})
    rows.append({"period": "current", "cat": "cat2", "value": 100})
    # cat3: baseline 40 -> current 100 => delta +60
    rows.append({"period": "baseline", "cat": "cat3", "value": 40})
    rows.append({"period": "current", "cat": "cat3", "value": 100})
    # cat4: baseline 140 -> current 100 => delta -40
    rows.append({"period": "baseline", "cat": "cat4", "value": 140})
    rows.append({"period": "current", "cat": "cat4", "value": 100})
    # cat5: baseline 80 -> current 100 => delta +20
    rows.append({"period": "baseline", "cat": "cat5", "value": 80})
    rows.append({"period": "current", "cat": "cat5", "value": 100})
    df = pd.DataFrame(rows)

    current_filters = [{"column": "period", "op": "==", "value": "current"}]
    baseline_filters = [{"column": "period", "op": "==", "value": "baseline"}]

    result = contribution_analysis(df, "value", "cat", current_filters, baseline_filters, top_n=3)

    assert result["total_baseline_value"] == pytest.approx(0 + 180 + 40 + 140 + 80)
    assert result["total_current_value"] == pytest.approx(100 * 5)
    assert result["total_delta"] == pytest.approx(60)

    assert len(result["breakdown"]) == 4  # top 3 + 1 "other" bucket
    categories = [e["category"] for e in result["breakdown"]]
    assert categories[:3] == ["cat1", "cat2", "cat3"]
    other = result["breakdown"][3]
    assert other["category"] == "other"
    assert other["category_count"] == 2
    # cat4 (-40) + cat5 (+20) = -20
    assert other["delta"] == pytest.approx(-20)
    assert other["current_value"] == pytest.approx(200)
    assert other["baseline_value"] == pytest.approx(220)

    assert sum(e["delta"] for e in result["breakdown"]) == pytest.approx(result["total_delta"])


def test_contribution_analysis_validation_errors():
    df = _period_revenue_df()

    with pytest.raises(ToolExecutionError):  # unknown metric column
        contribution_analysis(df, "nope", "region", _CURRENT_FILTERS, _BASELINE_FILTERS)

    with pytest.raises(ToolExecutionError):  # unknown dimension column
        contribution_analysis(df, "revenue", "nope", _CURRENT_FILTERS, _BASELINE_FILTERS)

    with pytest.raises(ToolExecutionError):  # non-numeric metric column
        contribution_analysis(df, "region", "region", _CURRENT_FILTERS, _BASELINE_FILTERS)

    with pytest.raises(ToolExecutionError):  # empty current_filters
        contribution_analysis(df, "revenue", "region", [], _BASELINE_FILTERS)

    with pytest.raises(ToolExecutionError):  # empty baseline_filters
        contribution_analysis(df, "revenue", "region", _CURRENT_FILTERS, [])

    with pytest.raises(ToolExecutionError):  # both periods match nothing
        contribution_analysis(
            df,
            "revenue",
            "region",
            [{"column": "date", "op": "between", "value": ["2030-01-01", "2030-01-31"]}],
            [{"column": "date", "op": "between", "value": ["2031-01-01", "2031-01-31"]}],
        )

    with pytest.raises(ToolExecutionError):  # top_n not positive
        contribution_analysis(df, "revenue", "region", _CURRENT_FILTERS, _BASELINE_FILTERS, top_n=0)


# --- executive_summary: known ground truth ----------------------------------


def _trend_df() -> pd.DataFrame:
    """20 daily rows, value=10 for the first 10 days (Jan 1-10), value=20 for
    the last 10 days (Jan 11-20). Known: total=300, mean=15, count=20.
    midpoint of [Jan1, Jan20] falls at Jan10 12:00, so the current half is
    exactly Jan11-20 (sum 200) and the previous half is exactly Jan1-10
    (sum 100) -> delta=100, pct_change=100.0, direction='up'."""
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    values = [10] * 10 + [20] * 10
    return pd.DataFrame({"date": dates, "value": values})


def test_executive_summary_matches_hand_computed_kpis_and_trend():
    df = _trend_df()
    result = executive_summary(df, ["value"], date_column="date")

    assert result["row_count"] == 20
    assert result["trend_supported"] is True
    assert result["date_range"] == {"min": "2024-01-01", "max": "2024-01-20"}

    kpi = result["metrics"][0]
    assert kpi["metric"] == "value"
    assert kpi["total"] == pytest.approx(300)
    assert kpi["mean"] == pytest.approx(15)
    assert kpi["count"] == 20

    trend = kpi["trend"]
    assert trend["available"] is True
    assert trend["previous_period_value"] == pytest.approx(100)
    assert trend["current_period_value"] == pytest.approx(200)
    assert trend["delta"] == pytest.approx(100)
    assert trend["pct_change"] == pytest.approx(100.0)
    assert trend["direction"] == "up"


def test_executive_summary_trend_unavailable_with_too_little_date_range():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "value": [10, 20]})
    result = executive_summary(df, ["value"], date_column="date")
    assert result["trend_supported"] is False
    kpi = result["metrics"][0]
    assert kpi["trend"]["available"] is False
    assert "reason" in kpi["trend"]


def test_executive_summary_no_date_column_omits_trend_but_reports_kpis():
    df = _trend_df()
    result = executive_summary(df, ["value"])
    assert result["trend_supported"] is None
    assert result["date_range"] is None
    assert "trend" not in result["metrics"][0]
    assert result["metrics"][0]["total"] == pytest.approx(300)


def test_executive_summary_flags_anomaly():
    values = [100, 102, 98, 101, 99, 100, 103, 97, 100, 5000]
    df = pd.DataFrame({"revenue": values})
    result = executive_summary(df, ["revenue"])
    kpi = result["metrics"][0]
    assert kpi["anomalies"]["flagged"] is True
    assert kpi["anomalies"]["anomaly_count"] >= 1


def test_executive_summary_validation_errors():
    df = _trend_df()

    with pytest.raises(ToolExecutionError):  # empty metrics
        executive_summary(df, [])

    with pytest.raises(ToolExecutionError):  # unknown metric
        executive_summary(df, ["nope"])

    with pytest.raises(ToolExecutionError):  # non-numeric metric
        df2 = df.copy()
        df2["label"] = "x"
        executive_summary(df2, ["label"])

    with pytest.raises(ToolExecutionError):  # unknown date_column
        executive_summary(df, ["value"], date_column="nope")

    with pytest.raises(ToolExecutionError):  # filters eliminate all rows
        executive_summary(df, ["value"], filters=[{"column": "value", "op": ">", "value": 99999}])


# --- Real demo dataset --------------------------------------------------------


def test_contribution_analysis_and_executive_summary_run_against_demo_sales_data():
    df = pd.read_excel(DEMO_XLSX)

    contribution = contribution_analysis(
        df,
        "revenue",
        "region",
        current_filters=[{"column": "date", "op": "between", "value": ["2025-01-01", "2025-06-30"]}],
        baseline_filters=[{"column": "date", "op": "between", "value": ["2024-07-01", "2024-12-31"]}],
    )
    assert contribution["category_count"] > 0
    assert sum(e["delta"] for e in contribution["breakdown"]) == pytest.approx(contribution["total_delta"], abs=0.01)

    summary = executive_summary(df, ["revenue", "profit", "quantity"], date_column="date")
    assert summary["row_count"] == len(df)
    assert len(summary["metrics"]) == 3
    for kpi in summary["metrics"]:
        assert kpi["count"] > 0
        assert kpi["total"] is not None
