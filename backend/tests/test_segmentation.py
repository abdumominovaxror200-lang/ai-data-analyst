from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.tools.errors import ToolExecutionError
from app.tools.segmentation import churn_risk_analysis, cohort_analysis, rfm_analysis

DEMO_DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "demo" / "sales_data.xlsx"


# --- RFM ------------------------------------------------------------------


def _rfm_df() -> pd.DataFrame:
    rows = []
    # One obvious "Champion": very recent, frequent, high spend.
    for i in range(5):
        rows.append({"customer": "champion", "date": pd.Timestamp("2024-05-27") + pd.Timedelta(days=i), "value": 500.0})
    # One obvious "Lost": long-ago, rare, low spend.
    rows.append({"customer": "lost", "date": pd.Timestamp("2023-01-01"), "value": 5.0})
    rows.append({"customer": "lost", "date": pd.Timestamp("2023-01-10"), "value": 5.0})
    # Filler customers spread across the quintile range so quintile scoring has
    # enough distinct values to split on.
    for c in range(20):
        for i in range(3):
            rows.append(
                {
                    "customer": f"filler{c}",
                    "date": pd.Timestamp("2024-03-01") + pd.Timedelta(days=i * 10 + c),
                    "value": 50.0 + c * 5,
                }
            )
    return pd.DataFrame(rows)


def test_rfm_analysis_champion_and_lost_land_in_expected_segments():
    df = _rfm_df()
    result = rfm_analysis(df, "customer", "date", "value", reference_date="2024-06-01")

    by_segment = {s["segment"]: s for s in result["segments"]}
    assert "Champions" in by_segment
    assert by_segment["Champions"]["customer_count"] >= 1
    assert "Lost" in by_segment
    assert by_segment["Lost"]["customer_count"] >= 1

    total = sum(s["customer_count"] for s in result["segments"])
    assert total == result["n_customers"] == 22


def test_rfm_analysis_reference_date_defaults_to_max_date():
    df = _rfm_df()
    result = rfm_analysis(df, "customer", "date", "value")
    assert result["reference_date"] == df["date"].max().strftime("%Y-%m-%d")


def test_rfm_analysis_requires_minimum_customers():
    df = pd.DataFrame(
        {
            "customer": ["a", "b", "c"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "value": [10.0, 20.0, 30.0],
        }
    )
    with pytest.raises(ToolExecutionError):
        rfm_analysis(df, "customer", "date", "value")


def test_rfm_analysis_unknown_column_raises():
    df = _rfm_df()
    with pytest.raises(ToolExecutionError):
        rfm_analysis(df, "nope", "date", "value")


def test_rfm_analysis_non_numeric_value_column_raises():
    df = _rfm_df()
    df["value"] = df["value"].astype(str)
    with pytest.raises(ToolExecutionError):
        rfm_analysis(df, "customer", "date", "value")


# --- Cohort -----------------------------------------------------------------


def test_cohort_analysis_recovers_known_retention_numbers():
    rows = []
    # January cohort: 4 customers. 2 return in Feb, 1 returns in Mar -> [1.0, 0.5, 0.25].
    rows += [
        {"customer": "a", "date": "2024-01-05"},
        {"customer": "b", "date": "2024-01-10"},
        {"customer": "c", "date": "2024-01-15"},
        {"customer": "d", "date": "2024-01-20"},
        {"customer": "a", "date": "2024-02-05"},
        {"customer": "b", "date": "2024-02-10"},
        {"customer": "a", "date": "2024-03-05"},
    ]
    # February cohort: 2 new customers. 1 returns in Mar -> [1.0, 0.5].
    rows += [
        {"customer": "e", "date": "2024-02-01"},
        {"customer": "f", "date": "2024-02-02"},
        {"customer": "e", "date": "2024-03-01"},
    ]
    df = pd.DataFrame(rows)

    result = cohort_analysis(df, "customer", "date", period="M")
    by_cohort = {c["cohort"]: c for c in result["cohorts"]}

    assert by_cohort["2024-01"]["cohort_size"] == 4
    assert by_cohort["2024-01"]["periods_since_first"]["0"] == pytest.approx(1.0)
    assert by_cohort["2024-01"]["periods_since_first"]["1"] == pytest.approx(0.5)
    assert by_cohort["2024-01"]["periods_since_first"]["2"] == pytest.approx(0.25)

    assert by_cohort["2024-02"]["cohort_size"] == 2
    assert by_cohort["2024-02"]["periods_since_first"]["0"] == pytest.approx(1.0)
    assert by_cohort["2024-02"]["periods_since_first"]["1"] == pytest.approx(0.5)


def test_cohort_analysis_with_value_column_sums_totals():
    rows = [
        {"customer": "a", "date": "2024-01-05", "spend": 100.0},
        {"customer": "b", "date": "2024-01-10", "spend": 50.0},
        {"customer": "a", "date": "2024-02-05", "spend": 30.0},
    ]
    df = pd.DataFrame(rows)
    result = cohort_analysis(df, "customer", "date", value_column="spend", period="M")
    assert result["metric"] == "total_value"
    cohort = result["cohorts"][0]
    assert cohort["periods_since_first"]["0"] == pytest.approx(150.0)
    assert cohort["periods_since_first"]["1"] == pytest.approx(30.0)


def test_cohort_analysis_rejects_invalid_period():
    df = pd.DataFrame({"customer": ["a"], "date": ["2024-01-01"]})
    with pytest.raises(ToolExecutionError):
        cohort_analysis(df, "customer", "date", period="bogus")


# --- Churn --------------------------------------------------------------


def _churn_df(reference_offset_days: int = 6) -> tuple[pd.DataFrame, str]:
    rows = []
    # Two frequent buyers (~5 day cadence) to make threshold inference stable.
    for cust in ("freq_a", "freq_b"):
        for i in range(10):
            rows.append({"customer": cust, "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i * 5)})
    # One customer who vanished long ago relative to that cadence.
    rows.append({"customer": "gone", "date": pd.Timestamp("2023-01-01")})
    rows.append({"customer": "gone", "date": pd.Timestamp("2023-02-01")})
    df = pd.DataFrame(rows)
    reference_date = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=9 * 5 + reference_offset_days)).strftime(
        "%Y-%m-%d"
    )
    return df, reference_date


def test_churn_risk_analysis_flags_long_gap_customer_as_churned():
    df, reference_date = _churn_df()
    result = churn_risk_analysis(df, "customer", "date", reference_date=reference_date)
    assert result["counts"]["churned"] >= 1
    assert result["threshold_inferred"] is True
    assert result["churn_threshold_days"] > 0
    total = sum(result["counts"].values())
    assert total == result["n_customers"] == 3


def test_churn_risk_analysis_explicit_threshold_used_verbatim():
    df, reference_date = _churn_df()
    result = churn_risk_analysis(df, "customer", "date", reference_date=reference_date, churn_threshold_days=1000)
    assert result["threshold_inferred"] is False
    assert result["churn_threshold_days"] == 1000
    # With a 1000-day threshold nobody in this tiny dataset is "churned".
    assert result["counts"]["churned"] == 0


def test_churn_risk_analysis_requires_enough_history_to_infer_threshold():
    df = pd.DataFrame(
        {
            "customer": ["a", "b"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )
    with pytest.raises(ToolExecutionError):
        churn_risk_analysis(df, "customer", "date")


def test_churn_risk_analysis_rejects_non_positive_explicit_threshold():
    df, reference_date = _churn_df()
    with pytest.raises(ToolExecutionError):
        churn_risk_analysis(df, "customer", "date", reference_date=reference_date, churn_threshold_days=0)


# --- Real demo dataset --------------------------------------------------------


def test_rfm_cohort_churn_run_against_demo_sales_data():
    df = pd.read_excel(DEMO_DATASET_PATH)

    rfm = rfm_analysis(df, "customer_id", "date", "revenue")
    assert rfm["n_customers"] == df["customer_id"].nunique()
    assert sum(s["customer_count"] for s in rfm["segments"]) == rfm["n_customers"]

    cohort = cohort_analysis(df, "customer_id", "date", period="M")
    assert cohort["n_cohorts"] > 0
    assert all(c["cohort_size"] > 0 for c in cohort["cohorts"])

    churn = churn_risk_analysis(df, "customer_id", "date")
    assert sum(churn["counts"].values()) == churn["n_customers"]
    assert churn["threshold_inferred"] is True
    assert churn["churn_threshold_days"] > 0
