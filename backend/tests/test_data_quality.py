from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.tools.data_quality import data_quality_report, duplicate_analysis
from app.tools.errors import ToolExecutionError

DEMO_XLSX = Path(__file__).resolve().parents[2] / "data" / "demo" / "sales_data.xlsx"


# --- duplicate_analysis: full-row mode ---


def _exact_duplicates_df() -> pd.DataFrame:
    # Rows 0/1/2 are exact full-row duplicates of each other (3-row group).
    # Rows 3/4 are an exact 2-row duplicate group.
    # Row 5 is unique.
    return pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 3],
            "b": ["x", "x", "x", "y", "y", "z"],
        }
    )


def test_duplicate_analysis_finds_exact_full_row_duplicates_precisely():
    df = _exact_duplicates_df()
    result = duplicate_analysis(df)

    assert result["mode"] == "full_row"
    assert result["total_rows"] == 6
    # rows involved in ANY duplicate group: 3 (group of a=1) + 2 (group of a=2) = 5
    assert result["duplicate_row_count"] == 5
    assert result["duplicate_pct"] == round(5 / 6 * 100, 2)
    assert result["duplicate_group_count"] == 2

    # largest group first
    assert result["examples"][0]["occurrences"] == 3
    assert result["examples"][0]["key_values"] == {"a": 1, "b": "x"}
    assert sorted(result["examples"][0]["row_indices"]) == [0, 1, 2]

    assert result["examples"][1]["occurrences"] == 2
    assert result["examples"][1]["key_values"] == {"a": 2, "b": "y"}
    assert sorted(result["examples"][1]["row_indices"]) == [3, 4]

    assert result["examples_truncated"] is False


def test_duplicate_analysis_no_duplicates_returns_zeroed_result():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = duplicate_analysis(df)
    assert result["duplicate_row_count"] == 0
    assert result["duplicate_group_count"] == 0
    assert result["examples"] == []


def test_duplicate_analysis_max_examples_caps_and_flags_truncation():
    # 5 distinct duplicate groups, each of size 2 -> 10 duplicate rows total.
    a_vals = []
    for k in range(5):
        a_vals.extend([k, k])
    df = pd.DataFrame({"a": a_vals})
    result = duplicate_analysis(df, max_examples=2)
    assert result["duplicate_group_count"] == 5
    assert len(result["examples"]) == 2
    assert result["examples_truncated"] is True


# --- duplicate_analysis: subset_columns mode (the more useful real-world case) ---


def _subset_duplicates_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 3, 3, 3],
            "date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-02",
            ],
            # customer 1's two rows share (customer_id, date) but differ in amount
            # -> NOT a full-row duplicate. customer 3's three rows are identical
            # in every column -> IS a full-row duplicate too.
            "amount": [100, 150, 200, 10, 10, 10],
        }
    )


def test_duplicate_analysis_subset_columns_catches_what_full_row_mode_misses():
    df = _subset_duplicates_df()

    full_row = duplicate_analysis(df)
    subset = duplicate_analysis(df, subset_columns=["customer_id", "date"])

    # Full-row mode only catches customer 3's identical rows (amount also matches).
    assert full_row["duplicate_row_count"] == 3
    assert full_row["duplicate_group_count"] == 1
    full_row_indices = {i for ex in full_row["examples"] for i in ex["row_indices"]}
    assert full_row_indices == {3, 4, 5}
    # customer 1's rows (index 0, 1) must NOT be flagged in full-row mode,
    # since their differing `amount` makes them technically unique rows.
    assert 0 not in full_row_indices and 1 not in full_row_indices

    # Subset-column mode catches BOTH: customer 1 billed twice same day (different
    # amounts) AND customer 3's identical trio.
    assert subset["duplicate_row_count"] == 5  # rows 0,1 (customer1) + 3,4,5 (customer3)
    assert subset["duplicate_group_count"] == 2
    subset_groups_by_key = {tuple(ex["key_values"].values()): ex for ex in subset["examples"]}
    customer1_group = subset_groups_by_key[(1, "2024-01-01")]
    assert customer1_group["occurrences"] == 2
    assert sorted(customer1_group["row_indices"]) == [0, 1]


def test_duplicate_analysis_rejects_unknown_subset_column():
    df = _subset_duplicates_df()
    with pytest.raises(ToolExecutionError):
        duplicate_analysis(df, subset_columns=["does_not_exist"])


def test_duplicate_analysis_rejects_empty_subset_columns_list():
    df = _subset_duplicates_df()
    with pytest.raises(ToolExecutionError):
        duplicate_analysis(df, subset_columns=[])


def test_duplicate_analysis_rejects_empty_dataframe_after_filters():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ToolExecutionError):
        duplicate_analysis(df, filters=[{"column": "a", "op": ">", "value": 100}])


def test_duplicate_analysis_rejects_negative_max_examples():
    df = _exact_duplicates_df()
    with pytest.raises(ToolExecutionError):
        duplicate_analysis(df, max_examples=-1)


# --- data_quality_report: mixed-type detection ---


def _mixed_type_df() -> pd.DataFrame:
    # 20 non-null values: 17 clean numbers, 3 non-numeric strings ("1,234"
    # thousands-separator artifact, "N/A", "unknown") -> 3/20 = 15% failure
    # rate, which should land in the "medium" severity bucket (> 5%, <= 20%).
    values = [str(i) for i in range(17)] + ["1,234", "N/A", "unknown"]
    return pd.DataFrame({"amount": values, "other": range(20)})


def test_data_quality_report_flags_mixed_type_column_with_correct_fraction():
    df = _mixed_type_df()
    result = data_quality_report(df)
    mixed = {c["column"]: c for c in result["mixed_type_columns"]}
    assert "amount" in mixed
    col = mixed["amount"]
    assert col["non_null_count"] == 20
    assert col["numeric_coercible_count"] == 17
    assert col["non_numeric_count"] == 3
    assert col["non_numeric_fraction"] == pytest.approx(0.15)
    assert col["severity"] == "medium"
    assert set(col["example_non_numeric_values"]) >= {"1,234", "N/A", "unknown"}

    messages = " ".join(i["message"] for i in result["quality_issues"] if i["category"] == "mixed_type")
    assert "amount" in messages


def test_data_quality_report_does_not_flag_a_clean_numeric_or_text_column():
    df = pd.DataFrame(
        {
            "clean_numeric": [str(i) for i in range(20)],
            "clean_text": [f"category-{i % 3}" for i in range(20)],
        }
    )
    result = data_quality_report(df)
    flagged_cols = {c["column"] for c in result["mixed_type_columns"]}
    assert "clean_numeric" not in flagged_cols
    # clean_text is almost entirely non-numeric -> below the min-success-fraction
    # bar, correctly NOT treated as a "mistyped numeric" column.
    assert "clean_text" not in flagged_cols


# --- data_quality_report: missingness co-occurrence ---


def test_data_quality_report_flags_correlated_missingness():
    n = 200
    rng = np.random.default_rng(11)
    # col_a missing on a random ~25% of rows; col_b missing on EXACTLY the
    # same rows (perfect correlation) plus nothing else -> r == 1.0.
    missing_mask = rng.random(n) < 0.25
    col_a = pd.Series(range(n), dtype="float64")
    col_a[missing_mask] = np.nan
    col_b = pd.Series(range(n, 2 * n), dtype="float64")
    col_b[missing_mask] = np.nan
    df = pd.DataFrame({"col_a": col_a, "col_b": col_b, "filler": range(n)})

    result = data_quality_report(df)
    pairs = {frozenset((p["column_a"], p["column_b"])) for p in result["missingness_cooccurrence"]}
    assert frozenset({"col_a", "col_b"}) in pairs
    flagged = next(p for p in result["missingness_cooccurrence"] if {p["column_a"], p["column_b"]} == {"col_a", "col_b"})
    assert flagged["correlation"] == pytest.approx(1.0)
    assert flagged["severity"] == "high"

    messages = " ".join(i["message"] for i in result["quality_issues"] if i["category"] == "missingness_cooccurrence")
    assert "col_a" in messages and "col_b" in messages


def test_data_quality_report_does_not_flag_independent_missingness():
    n = 1000
    rng = np.random.default_rng(42)
    # Two independent Bernoulli missingness patterns at a non-trivial rate
    # each; with n=1000 the sample correlation between independent 0/1
    # indicators is essentially always well below the 0.5 flagging
    # threshold (std of the sampling distribution ~ 1/sqrt(n) ~ 0.03).
    col_c = pd.Series(range(n), dtype="float64")
    col_c[rng.random(n) < 0.2] = np.nan
    col_d = pd.Series(range(n, 2 * n), dtype="float64")
    col_d[rng.random(n) < 0.2] = np.nan
    df = pd.DataFrame({"col_c": col_c, "col_d": col_d, "filler": range(n)})

    result = data_quality_report(df)
    pairs = {frozenset((p["column_a"], p["column_b"])) for p in result["missingness_cooccurrence"]}
    assert frozenset({"col_c", "col_d"}) not in pairs


def test_data_quality_report_ignores_columns_with_near_zero_missing_rate():
    n = 500
    df = pd.DataFrame({"a": range(n), "b": range(n)}, dtype="float64")
    # both columns missing on the exact same single row -> perfectly
    # correlated, but the missing rate (0.2%) is below the "non-trivial"
    # bar and must not be flagged (avoids noise).
    df.loc[0, "a"] = np.nan
    df.loc[0, "b"] = np.nan
    result = data_quality_report(df)
    pairs = {frozenset((p["column_a"], p["column_b"])) for p in result["missingness_cooccurrence"]}
    assert frozenset({"a", "b"}) not in pairs


# --- data_quality_report: composite structure, score, and error handling ---


def test_data_quality_report_perfect_dataset_scores_100():
    df = pd.DataFrame({"a": range(50), "b": [f"cat-{i % 4}" for i in range(50)]})
    result = data_quality_report(df)
    assert result["quality_score"] == 100
    assert result["quality_issues"] == []


def test_data_quality_report_score_matches_documented_formula():
    # Construct a dataset with a KNOWN, precisely-controlled issue set:
    # - one column with 30% missing (> 20% => high, weight 15)
    # - one exact-duplicate group covering 2/10 rows = 20% duplicate pct (> 5% => high, weight 15)
    n = 10
    df = pd.DataFrame(
        {
            # rows 0,1 identical in BOTH columns -> a genuine full-row duplicate
            # (full-row mode requires every column to match).
            "mostly_missing": [1, 1, 3, None, None, None, 7, 8, 9, 10],  # 3/10 = 30% missing
            "dup_col": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )
    result = data_quality_report(df)

    categories = {i["category"] for i in result["quality_issues"]}
    assert "missingness" in categories
    assert "duplicates" in categories

    total_penalty = sum(
        {"high": 15, "medium": 7, "low": 2}[i["severity"]] for i in result["quality_issues"]
    )
    assert result["quality_score"] == max(0, 100 - total_penalty)


def test_data_quality_report_rejects_empty_dataframe_after_filters():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ToolExecutionError):
        data_quality_report(df, filters=[{"column": "a", "op": ">", "value": 100}])


def test_data_quality_report_structure_has_all_expected_keys():
    df = _mixed_type_df()
    result = data_quality_report(df)
    for key in (
        "row_count",
        "column_count",
        "quality_score",
        "quality_verdict",
        "quality_issues",
        "missingness",
        "duplicates",
        "mixed_type_columns",
        "missingness_cooccurrence",
    ):
        assert key in result
    assert 0 <= result["quality_score"] <= 100


# --- real demo dataset (end-to-end sanity check) ---


@pytest.mark.skipif(not DEMO_XLSX.exists(), reason="demo dataset not present in this checkout")
def test_data_quality_report_runs_cleanly_on_demo_dataset():
    df = pd.read_excel(DEMO_XLSX)
    result = data_quality_report(df)

    assert result["row_count"] == len(df)
    assert 0 <= result["quality_score"] <= 100
    assert isinstance(result["quality_issues"], list)
    # sanity: report should be JSON-serializable (no stray Timestamp/np types)
    import json

    json.dumps(result)
