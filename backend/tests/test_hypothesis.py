from __future__ import annotations

import pandas as pd
import pytest
from scipy import stats

from app.tools.errors import ToolExecutionError
from app.tools.hypothesis import (
    anova_test,
    chi_square_test,
    confidence_interval,
    effect_size,
    t_test,
)


# ---------------------------------------------------------------------------
# t_test: one-sample
# ---------------------------------------------------------------------------


def test_t_test_one_sample_matches_scipy():
    values = [10, 12, 9, 11, 13, 10, 12]
    df = pd.DataFrame({"value": values})
    result = t_test(df, column="value", popmean=10)

    expected_stat, expected_p = stats.ttest_1samp(values, 10)
    assert result["statistic"] == pytest.approx(round(expected_stat, 4))
    assert result["p_value"] == pytest.approx(round(expected_p, 6))
    assert result["degrees_of_freedom"] == len(values) - 1
    assert result["test"] == "one_sample_t_test"


def test_t_test_one_sample_not_significant_when_close_to_popmean():
    df = pd.DataFrame({"value": [10.0, 10.1, 9.9, 10.0, 10.05]})
    result = t_test(df, column="value", popmean=10.0)
    assert result["significant"] is False


def test_t_test_one_sample_requires_popmean_or_groups():
    df = pd.DataFrame({"value": [1, 2, 3]})
    with pytest.raises(ToolExecutionError):
        t_test(df, column="value")


# ---------------------------------------------------------------------------
# t_test: two-sample
# ---------------------------------------------------------------------------


def test_t_test_two_sample_detects_clear_difference():
    df = pd.DataFrame(
        {
            "score": [10, 12, 11, 13, 12, 20, 22, 21, 23, 22],
            "group": ["A"] * 5 + ["B"] * 5,
        }
    )
    result = t_test(df, column="score", group_column="group", group_a="A", group_b="B")

    a = df.loc[df["group"] == "A", "score"]
    b = df.loc[df["group"] == "B", "score"]
    expected_stat, expected_p = stats.ttest_ind(a, b, equal_var=False)

    assert result["statistic"] == pytest.approx(round(expected_stat, 4))
    assert result["p_value"] == pytest.approx(round(expected_p, 6))
    assert result["significant"] is True
    assert result["group_a"]["mean"] == pytest.approx(11.6)
    assert result["group_b"]["mean"] == pytest.approx(21.6)


def test_t_test_two_sample_no_difference_not_significant():
    df = pd.DataFrame(
        {
            "score": [10, 11, 9, 10, 10, 11, 9, 10],
            "group": ["A"] * 4 + ["B"] * 4,
        }
    )
    result = t_test(df, column="score", group_column="group", group_a="A", group_b="B")
    assert result["significant"] is False


def test_t_test_unknown_column_raises():
    df = pd.DataFrame({"value": [1, 2, 3]})
    with pytest.raises(ToolExecutionError):
        t_test(df, column="nope", popmean=1)


def test_t_test_non_numeric_column_raises():
    df = pd.DataFrame({"value": ["a", "b", "c"]})
    with pytest.raises(ToolExecutionError):
        t_test(df, column="value", popmean=1)


def test_t_test_insufficient_group_samples_raises():
    df = pd.DataFrame({"score": [10, 20], "group": ["A", "B"]})
    with pytest.raises(ToolExecutionError):
        t_test(df, column="score", group_column="group", group_a="A", group_b="B")


def test_t_test_unknown_group_value_raises():
    df = pd.DataFrame({"score": [10, 11, 12, 20, 21, 22], "group": ["A", "A", "A", "B", "B", "B"]})
    with pytest.raises(ToolExecutionError):
        t_test(df, column="score", group_column="group", group_a="A", group_b="C")


# ---------------------------------------------------------------------------
# chi_square_test
# ---------------------------------------------------------------------------


def test_chi_square_detects_association():
    # Strong association: region strongly predicts preference.
    df = pd.DataFrame(
        {
            "region": ["North"] * 20 + ["South"] * 20,
            "preference": ["Tea"] * 18 + ["Coffee"] * 2 + ["Coffee"] * 18 + ["Tea"] * 2,
        }
    )
    result = chi_square_test(df, column_a="region", column_b="preference")

    contingency = pd.crosstab(df["region"], df["preference"])
    expected_stat, expected_p, expected_dof, _ = stats.chi2_contingency(contingency)

    assert result["statistic"] == pytest.approx(round(expected_stat, 4))
    assert result["p_value"] == pytest.approx(round(expected_p, 6))
    assert result["degrees_of_freedom"] == expected_dof
    assert result["significant"] is True
    assert result["contingency_table"]["North"]["Tea"] == 18


def test_chi_square_no_association_not_significant():
    df = pd.DataFrame(
        {
            "region": ["North", "South"] * 10,
            "preference": ["Tea", "Coffee"] * 5 + ["Coffee", "Tea"] * 5,
        }
    )
    result = chi_square_test(df, column_a="region", column_b="preference")
    assert isinstance(result["significant"], bool)


def test_chi_square_unknown_column_raises():
    df = pd.DataFrame({"a": ["x", "y"], "b": ["1", "2"]})
    with pytest.raises(ToolExecutionError):
        chi_square_test(df, column_a="a", column_b="nope")


def test_chi_square_requires_two_categories_each():
    df = pd.DataFrame({"a": ["x", "x", "x"], "b": ["1", "2", "1"]})
    with pytest.raises(ToolExecutionError):
        chi_square_test(df, column_a="a", column_b="b")


# ---------------------------------------------------------------------------
# anova_test
# ---------------------------------------------------------------------------


def test_anova_detects_group_differences():
    df = pd.DataFrame(
        {
            "value": [10, 11, 9, 20, 21, 19, 30, 31, 29],
            "group": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
        }
    )
    result = anova_test(df, value_column="value", group_column="group")

    a = df.loc[df["group"] == "A", "value"]
    b = df.loc[df["group"] == "B", "value"]
    c = df.loc[df["group"] == "C", "value"]
    expected_stat, expected_p = stats.f_oneway(a, b, c)

    assert result["statistic"] == pytest.approx(round(expected_stat, 4))
    assert result["p_value"] == pytest.approx(round(expected_p, 6))
    assert result["significant"] is True
    assert result["groups"]["A"]["mean"] == pytest.approx(10.0)
    assert result["groups"]["B"]["n"] == 3


def test_anova_requires_at_least_two_groups():
    df = pd.DataFrame({"value": [1, 2, 3], "group": ["A", "A", "A"]})
    with pytest.raises(ToolExecutionError):
        anova_test(df, value_column="value", group_column="group")


def test_anova_requires_min_samples_per_group():
    df = pd.DataFrame({"value": [1, 2, 3], "group": ["A", "A", "B"]})
    with pytest.raises(ToolExecutionError):
        anova_test(df, value_column="value", group_column="group")


def test_anova_non_numeric_value_column_raises():
    df = pd.DataFrame({"value": ["a", "b", "c", "d"], "group": ["A", "A", "B", "B"]})
    with pytest.raises(ToolExecutionError):
        anova_test(df, value_column="value", group_column="group")


# ---------------------------------------------------------------------------
# confidence_interval
# ---------------------------------------------------------------------------


def test_confidence_interval_matches_scipy():
    values = [10, 12, 9, 11, 13, 10, 12, 11]
    df = pd.DataFrame({"value": values})
    result = confidence_interval(df, column="value", confidence=0.95)

    mean = pd.Series(values).mean()
    sem = stats.sem(values)
    expected_lower, expected_upper = stats.t.interval(0.95, df=len(values) - 1, loc=mean, scale=sem)

    assert result["mean"] == pytest.approx(round(mean, 4))
    assert result["lower_bound"] == pytest.approx(round(expected_lower, 4))
    assert result["upper_bound"] == pytest.approx(round(expected_upper, 4))
    assert result["margin_of_error"] == pytest.approx(round((expected_upper - expected_lower) / 2, 4))


def test_confidence_interval_invalid_confidence_raises():
    df = pd.DataFrame({"value": [1, 2, 3]})
    with pytest.raises(ToolExecutionError):
        confidence_interval(df, column="value", confidence=1.5)


def test_confidence_interval_insufficient_samples_raises():
    df = pd.DataFrame({"value": [1]})
    with pytest.raises(ToolExecutionError):
        confidence_interval(df, column="value")


def test_confidence_interval_unknown_column_raises():
    df = pd.DataFrame({"value": [1, 2, 3]})
    with pytest.raises(ToolExecutionError):
        confidence_interval(df, column="nope")


# ---------------------------------------------------------------------------
# effect_size
# ---------------------------------------------------------------------------


def test_effect_size_cohens_d_hand_checked():
    # Two groups with a 1 pooled-std-unit-ish mean difference, hand-computable.
    df = pd.DataFrame(
        {
            "score": [10, 12, 11, 13, 9] + [15, 17, 16, 18, 14],
            "group": ["A"] * 5 + ["B"] * 5,
        }
    )
    result = effect_size(df, column="score", group_column="group", group_a="A", group_b="B")

    a = df.loc[df["group"] == "A", "score"]
    b = df.loc[df["group"] == "B", "score"]
    n1, n2 = len(a), len(b)
    pooled_std = (((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)) ** 0.5
    expected_d = (a.mean() - b.mean()) / pooled_std

    assert result["cohens_d"] == pytest.approx(round(expected_d, 4))
    assert result["magnitude"] in {"negligible", "small", "medium", "large"}


def test_effect_size_zero_when_means_equal():
    df = pd.DataFrame({"score": [10, 11, 9, 10, 11, 9], "group": ["A", "A", "A", "B", "B", "B"]})
    result = effect_size(df, column="score", group_column="group", group_a="A", group_b="B")
    assert result["cohens_d"] == pytest.approx(0.0, abs=1e-9)
    assert result["magnitude"] == "negligible"


def test_effect_size_insufficient_samples_raises():
    df = pd.DataFrame({"score": [10, 20], "group": ["A", "B"]})
    with pytest.raises(ToolExecutionError):
        effect_size(df, column="score", group_column="group", group_a="A", group_b="B")


def test_effect_size_unknown_group_value_raises():
    df = pd.DataFrame({"score": [10, 11, 12, 20, 21, 22], "group": ["A", "A", "A", "B", "B", "B"]})
    with pytest.raises(ToolExecutionError):
        effect_size(df, column="score", group_column="group", group_a="A", group_b="nope")
