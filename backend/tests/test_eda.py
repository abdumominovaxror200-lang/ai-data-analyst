from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.tools.eda import analyze_cardinality, analyze_distributions, automated_eda
from app.tools.errors import ToolExecutionError

DEMO_XLSX = Path(__file__).resolve().parents[2] / "data" / "demo" / "sales_data.xlsx"


def _known_properties_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 200

    # near-constant: 99% one value ("A"), 1% ("B")
    near_constant = ["A"] * 198 + ["B"] * 2

    # ID-like: every row unique, high-cardinality identifier text
    id_like = [f"ID-{i:05d}" for i in range(n)]

    # clearly right-skewed: exponential distribution
    skewed = rng.exponential(scale=2.0, size=n)

    # two columns constructed to be highly correlated
    corr_a = rng.normal(100, 10, n)
    corr_b = corr_a * 2.0 + rng.normal(0, 0.5, n)  # near-perfect linear relationship

    # boolean-like text column that profiler.py's role inference (bool dtype
    # check) would NOT classify as "boolean"
    bool_like = rng.choice(["Yes", "No"], n)

    # low-cardinality balanced categorical
    category = rng.choice(["North", "South", "East", "West"], n)

    return pd.DataFrame(
        {
            "near_constant_col": near_constant,
            "id_like_col": id_like,
            "skewed_col": skewed,
            "corr_a": corr_a,
            "corr_b": corr_b,
            "bool_like_col": bool_like,
            "category_col": category,
        }
    )


# --- analyze_cardinality ---


def test_cardinality_flags_near_constant_column():
    df = _known_properties_df()
    result = analyze_cardinality(df)
    col = next(c for c in result["columns"] if c["name"] == "near_constant_col")
    assert col["classification"] in {"constant", "near_constant"}
    assert col["top_value_pct"] >= 95.0


def test_cardinality_flags_id_like_column():
    df = _known_properties_df()
    result = analyze_cardinality(df)
    col = next(c for c in result["columns"] if c["name"] == "id_like_col")
    assert col["classification"] == "unique_id_like"
    assert col["unique_ratio"] == 1.0


def test_cardinality_flags_boolean_like_text_column():
    df = _known_properties_df()
    result = analyze_cardinality(df)
    col = next(c for c in result["columns"] if c["name"] == "bool_like_col")
    assert col["classification"] == "boolean_like"


def test_cardinality_low_cardinality_categorical():
    df = _known_properties_df()
    result = analyze_cardinality(df)
    col = next(c for c in result["columns"] if c["name"] == "category_col")
    assert col["classification"] == "low_cardinality_categorical"
    assert col["unique_count"] == 4


def test_cardinality_continuous_numeric_not_flagged_as_id():
    df = _known_properties_df()
    result = analyze_cardinality(df)
    col = next(c for c in result["columns"] if c["name"] == "skewed_col")
    # a continuous float measurement should not be mistaken for an identifier
    assert col["classification"] == "continuous_numeric"


def test_cardinality_summary_counts_match_columns():
    df = _known_properties_df()
    result = analyze_cardinality(df)
    assert sum(result["summary"].values()) == len(df.columns)


def test_cardinality_rejects_empty_result_after_filters():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ToolExecutionError):
        analyze_cardinality(df, filters=[{"column": "a", "op": ">", "value": 100}])


# --- analyze_distributions ---


def test_distributions_detects_right_skew():
    df = _known_properties_df()
    result = analyze_distributions(df, columns=["skewed_col"])
    item = result["numeric"][0]
    assert item["skewness"] > 0
    assert "right-skewed" in item["skew_label"]


def test_distributions_symmetric_normal_data_roughly_symmetric():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"normal_col": rng.normal(0, 1, 500)})
    result = analyze_distributions(df)
    item = result["numeric"][0]
    assert item["skew_label"] == "roughly symmetric"


def test_distributions_categorical_balance_for_balanced_column():
    df = _known_properties_df()
    result = analyze_distributions(df, columns=["category_col"])
    item = result["categorical"][0]
    assert item["balance_label"] in {"well balanced", "moderately balanced"}


def test_distributions_categorical_balance_for_imbalanced_column():
    df = _known_properties_df()
    result = analyze_distributions(df, columns=["near_constant_col"])
    item = result["categorical"][0]
    assert item["balance_label"] == "imbalanced"
    assert item["imbalance_ratio"] > 1


def test_distributions_unknown_column_raises():
    df = _known_properties_df()
    with pytest.raises(ToolExecutionError):
        analyze_distributions(df, columns=["does_not_exist"])


def test_distributions_skips_datetime_columns():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10), "value": range(10)})
    result = analyze_distributions(df)
    assert any(s["name"] == "date" for s in result["skipped_columns"])


# --- automated_eda ---


def test_automated_eda_end_to_end_structure():
    df = _known_properties_df()
    result = automated_eda(df)
    assert result["row_count"] == len(df)
    assert result["column_count"] == len(df.columns)
    for key in ("schema", "missingness", "duplicates", "cardinality", "distributions", "outliers", "relationships", "potential_problems"):
        assert key in result


def test_automated_eda_surfaces_high_correlation_in_problems_and_relationships():
    df = _known_properties_df()
    result = automated_eda(df)
    pair_names = {(p["column_a"], p["column_b"]) for p in result["relationships"]["strongest_pairs"]}
    assert ("corr_a", "corr_b") in pair_names or ("corr_b", "corr_a") in pair_names
    top_pair = result["relationships"]["strongest_pairs"][0]
    assert abs(top_pair["correlation"]) > 0.9

    problem_messages = " ".join(p["message"] for p in result["potential_problems"])
    assert "corr_a" in problem_messages and "corr_b" in problem_messages


def test_automated_eda_flags_near_constant_and_id_like_in_problems():
    df = _known_properties_df()
    result = automated_eda(df)
    problem_messages = " ".join(p["message"] for p in result["potential_problems"])
    assert "near_constant_col" in problem_messages
    assert "id_like_col" in problem_messages


def test_automated_eda_rejects_empty_dataframe():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ToolExecutionError):
        automated_eda(df, filters=[{"column": "a", "op": ">", "value": 100}])


def test_automated_eda_no_crash_on_single_numeric_column():
    df = pd.DataFrame({"only_col": [1, 2, 3, 4, 5]})
    result = automated_eda(df)
    assert result["relationships"]["strongest_pairs"] == []


# --- real demo dataset (end-to-end sanity check) ---


@pytest.mark.skipif(not DEMO_XLSX.exists(), reason="demo dataset not present in this checkout")
def test_automated_eda_runs_cleanly_on_demo_dataset():
    df = pd.read_excel(DEMO_XLSX)
    result = automated_eda(df)

    assert result["row_count"] == len(df)
    assert len(result["potential_problems"]) > 0

    problem_messages = " ".join(p["message"] for p in result["potential_problems"])
    # customer_id has ~0.6% missing values in the demo data
    assert "customer_id" in problem_messages

    # revenue/cost/profit are strongly correlated in the demo data
    pair_cols = set()
    for pair in result["relationships"]["strongest_pairs"]:
        pair_cols.add(pair["column_a"])
        pair_cols.add(pair["column_b"])
    assert {"revenue", "cost", "profit"} & pair_cols
