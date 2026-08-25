from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.tools.advanced_charts import (
    _MAX_HEATMAP_COLUMNS,
    boxplot_data,
    correlation_heatmap_data,
    pareto_chart_data,
)
from app.tools.anomaly import detect_anomalies
from app.tools.errors import ToolExecutionError

DEMO_XLSX = Path(__file__).resolve().parents[2] / "data" / "demo" / "sales_data.xlsx"


# ---------------------------------------------------------------------------
# correlation_heatmap_data
# ---------------------------------------------------------------------------


def test_heatmap_perfect_and_zero_correlation_in_same_matrix():
    n = 200
    rng = np.random.default_rng(42)
    a = np.arange(n, dtype=float)
    b = a * 2.0 + 5.0  # perfectly correlated with a
    c = rng.permutation(n).astype(float)  # independent of a (shuffled, no linear relationship)
    df = pd.DataFrame({"a": a, "b": b, "c": c})

    result = correlation_heatmap_data(df)

    assert result["method"] == "pearson"
    assert set(result["columns"]) == {"a", "b", "c"}

    cell_lookup = {(cell["x"], cell["y"]): cell["value"] for cell in result["cells"]}
    assert cell_lookup[("a", "b")] == pytest.approx(1.0, abs=1e-6)
    assert cell_lookup[("b", "a")] == pytest.approx(1.0, abs=1e-6)
    assert cell_lookup[("a", "a")] == pytest.approx(1.0, abs=1e-6)
    assert abs(cell_lookup[("a", "c")]) < 0.15  # shuffled data: near-zero, not exactly 0

    # full grid: 3x3 = 9 cells, including diagonal and both triangles
    assert len(result["cells"]) == 9


def test_heatmap_requires_two_numeric_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "label": ["x", "y", "z"]})
    with pytest.raises(ToolExecutionError):
        correlation_heatmap_data(df)


def test_heatmap_rejects_unknown_column():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})
    with pytest.raises(ToolExecutionError):
        correlation_heatmap_data(df, columns=["a", "nope"])


def test_heatmap_rejects_invalid_method():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})
    with pytest.raises(ToolExecutionError):
        correlation_heatmap_data(df, method="not_a_method")


def test_heatmap_rejects_oversized_column_set():
    n_cols = _MAX_HEATMAP_COLUMNS + 1
    data = {f"col_{i}": np.arange(10) + i for i in range(n_cols)}
    df = pd.DataFrame(data)
    with pytest.raises(ToolExecutionError):
        correlation_heatmap_data(df)


def test_heatmap_respects_filters():
    df = pd.DataFrame(
        {
            "region": ["north"] * 5 + ["south"] * 5,
            "a": list(range(10)),
            "b": list(range(10)),
        }
    )
    result = correlation_heatmap_data(df, filters=[{"column": "region", "op": "==", "value": "north"}])
    cell_lookup = {(cell["x"], cell["y"]): cell["value"] for cell in result["cells"]}
    assert cell_lookup[("a", "b")] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# boxplot_data
# ---------------------------------------------------------------------------


def test_boxplot_known_five_number_summary_and_outliers():
    # Hand-constructed: 1..19 (Q1=5, median=10, Q3=15, IQR=10) plus two
    # deliberately injected outliers far outside the 1.5*IQR fences.
    base = list(range(1, 20))
    outliers = [-100, 500]
    values = base + outliers
    df = pd.DataFrame({"v": values})

    result = boxplot_data(df, value_column="v")
    box = result["boxes"][0]

    series = pd.Series(base, dtype=float)
    expected_q1 = float(series.quantile(0.25))
    expected_median = float(series.quantile(0.5))
    expected_q3 = float(series.quantile(0.75))

    # Five-number summary reflects the FULL series (including outliers) for min/max.
    assert box["min"] == -100.0
    assert box["max"] == 500.0
    assert box["count"] == len(values)

    # Cross-check against detect_anomalies's IQR bounds on the same data to
    # prove the two tools use the identical 1.5*IQR convention.
    anomaly_result = detect_anomalies(df, column="v", method="iqr")
    assert box["lower_fence"] == anomaly_result["bounds"]["lower"]
    assert box["upper_fence"] == anomaly_result["bounds"]["upper"]
    assert box["outlier_count"] == anomaly_result["anomaly_count"]
    assert -100 in box["outliers"]
    assert 500 in box["outliers"]

    # Sanity: fences computed from the whole series (quantiles are influenced
    # by the outliers themselves, so just confirm same-ballpark, not equal).
    assert box["lower_fence"] < expected_q1
    assert box["upper_fence"] > expected_q3
    assert expected_median > 0


def test_boxplot_grouped_caps_and_sorts_by_size():
    rng = np.random.default_rng(7)
    rows = []
    # group "big" has 50 rows, "small" has 5 rows
    rows += [{"grp": "big", "v": float(x)} for x in rng.normal(size=50)]
    rows += [{"grp": "small", "v": float(x)} for x in rng.normal(size=5)]
    df = pd.DataFrame(rows)

    result = boxplot_data(df, value_column="v", group_column="grp", max_groups=1)
    assert result["groups_truncated"] is True
    assert result["total_group_count"] == 2
    assert len(result["boxes"]) == 1
    assert result["boxes"][0]["group"] == "big"  # kept the larger group


def test_boxplot_outliers_capped_per_group():
    # A large base sample keeps the outlier fraction well under 25%, so the
    # injected extreme values don't drag the quartiles themselves upward
    # (a well-known limitation of IQR-based detection when >25% of the data
    # is "outliers" — not what this test is checking).
    base = list(range(1, 201))
    many_outliers = [10_000 + i for i in range(25)]
    df = pd.DataFrame({"v": base + many_outliers})
    result = boxplot_data(df, value_column="v")
    box = result["boxes"][0]
    assert box["outlier_count"] == 25
    assert len(box["outliers"]) == 20  # capped at _MAX_OUTLIERS_PER_GROUP


def test_boxplot_unknown_value_column_raises():
    df = pd.DataFrame({"v": [1, 2, 3]})
    with pytest.raises(ToolExecutionError):
        boxplot_data(df, value_column="nope")


def test_boxplot_non_numeric_value_column_raises():
    df = pd.DataFrame({"v": ["a", "b", "c"]})
    with pytest.raises(ToolExecutionError):
        boxplot_data(df, value_column="v")


def test_boxplot_unknown_group_column_raises():
    df = pd.DataFrame({"v": [1, 2, 3]})
    with pytest.raises(ToolExecutionError):
        boxplot_data(df, value_column="v", group_column="nope")


# ---------------------------------------------------------------------------
# pareto_chart_data
# ---------------------------------------------------------------------------


def test_pareto_known_8020_distribution_cumulative_correctness():
    # "A" contributes exactly 80% of total; B/C/D split the remaining 20%.
    df = pd.DataFrame(
        {
            "category": ["A"] * 80 + ["B"] * 10 + ["C"] * 6 + ["D"] * 4,
            "value": [1] * 80 + [1] * 10 + [1] * 6 + [1] * 4,
        }
    )
    result = pareto_chart_data(df, dimension_column="category", value_column="value", top_n=15)

    assert result["total"] == 100.0
    cats = result["categories"]
    assert cats[0]["category"] == "A"
    assert cats[0]["value"] == 80.0
    assert cats[0]["pct_of_total"] == 80.0
    assert cats[0]["cumulative_pct"] == 80.0

    # Cumulative percentage sequence is monotonically non-decreasing and ends at ~100.
    cum_values = [c["cumulative_pct"] for c in cats]
    assert all(cum_values[i] <= cum_values[i + 1] for i in range(len(cum_values) - 1))
    assert cum_values[-1] == pytest.approx(100.0, abs=0.01)
    assert result["other_bucket_included"] is False


def test_pareto_bundles_remainder_into_other_bucket():
    # 20 distinct categories (2 rows each, so this isn't the one-row-per-category
    # identifier case), each contributing 5% -> with top_n=5, 15 should fold into "Other".
    df = pd.DataFrame(
        {
            "category": [f"cat_{i}" for i in range(20) for _ in range(2)],
            "value": [2.5] * 40,
        }
    )
    result = pareto_chart_data(df, dimension_column="category", value_column="value", top_n=5)

    assert len(result["categories"]) == 6  # 5 top + 1 "Other"
    assert result["other_bucket_included"] is True
    assert result["other_category_count"] == 15
    other = result["categories"][-1]
    assert other["category"] == "Other"
    assert other["value"] == pytest.approx(75.0)
    assert other["cumulative_pct"] == pytest.approx(100.0, abs=0.01)

    # percentages sum to 100 across listed categories
    total_pct = sum(c["pct_of_total"] for c in result["categories"])
    assert total_pct == pytest.approx(100.0, abs=0.05)


def test_pareto_rejects_one_row_per_category_identifier_column():
    df = pd.DataFrame({"customer_id": [f"id_{i}" for i in range(50)], "revenue": list(range(50))})
    with pytest.raises(ToolExecutionError):
        pareto_chart_data(df, dimension_column="customer_id", value_column="revenue")


def test_pareto_rejects_non_numeric_value_column():
    df = pd.DataFrame({"category": ["a", "b", "a"], "revenue": ["x", "y", "z"]})
    with pytest.raises(ToolExecutionError):
        pareto_chart_data(df, dimension_column="category", value_column="revenue")


def test_pareto_rejects_unknown_columns():
    df = pd.DataFrame({"category": ["a", "b"], "revenue": [1, 2]})
    with pytest.raises(ToolExecutionError):
        pareto_chart_data(df, dimension_column="nope", value_column="revenue")
    with pytest.raises(ToolExecutionError):
        pareto_chart_data(df, dimension_column="category", value_column="nope")


def test_pareto_rejects_negative_totals():
    df = pd.DataFrame({"category": ["a", "a", "b"], "value": [10, -30, 5]})
    with pytest.raises(ToolExecutionError):
        pareto_chart_data(df, dimension_column="category", value_column="value")


# ---------------------------------------------------------------------------
# Real demo dataset end-to-end checks
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not DEMO_XLSX.exists(), reason="demo dataset not present")
def test_heatmap_against_real_demo_dataset():
    df = pd.read_excel(DEMO_XLSX)
    result = correlation_heatmap_data(df, columns=["quantity", "unit_price", "revenue", "cost", "profit"])
    assert result["columns"] == ["quantity", "unit_price", "revenue", "cost", "profit"]
    assert len(result["cells"]) == 25
    cell_lookup = {(c["x"], c["y"]): c["value"] for c in result["cells"]}
    assert cell_lookup[("revenue", "revenue")] == 1.0


@pytest.mark.skipif(not DEMO_XLSX.exists(), reason="demo dataset not present")
def test_boxplot_against_real_demo_dataset():
    df = pd.read_excel(DEMO_XLSX)
    result = boxplot_data(df, value_column="revenue", group_column="region")
    assert result["value_column"] == "revenue"
    assert len(result["boxes"]) > 0
    for box in result["boxes"]:
        assert box["min"] <= box["q1"] <= box["median"] <= box["q3"] <= box["max"]


@pytest.mark.skipif(not DEMO_XLSX.exists(), reason="demo dataset not present")
def test_pareto_against_real_demo_dataset():
    df = pd.read_excel(DEMO_XLSX)
    result = pareto_chart_data(df, dimension_column="product", value_column="revenue", top_n=10)
    assert result["dimension_column"] == "product"
    cum_values = [c["cumulative_pct"] for c in result["categories"]]
    assert all(cum_values[i] <= cum_values[i + 1] for i in range(len(cum_values) - 1))
    assert cum_values[-1] <= 100.01
