from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from app.tools.errors import ToolExecutionError
from app.tools.filtering import apply_filters

_MIN_GROUP_SIZE = 2


def t_test(
    df: pd.DataFrame,
    column: str,
    group_column: str | None = None,
    group_a: object | None = None,
    group_b: object | None = None,
    popmean: float | None = None,
    alpha: float = 0.05,
    filters: list[dict] | None = None,
) -> dict:
    """One-sample t-test (against `popmean`) or two-sample independent Welch's t-test
    (comparing `column` where `group_column == group_a` vs `== group_b`)."""
    working = apply_filters(df, filters) if filters else df
    if working.empty:
        raise ToolExecutionError("No rows match the given filters.")
    _validate_alpha(alpha)
    series = _numeric_series(working, column)

    if group_column is not None:
        if group_a is None or group_b is None:
            raise ToolExecutionError("group_a and group_b are required for a two-sample t-test.")
        if group_column not in working.columns:
            raise ToolExecutionError(f"Unknown column '{group_column}'.")

        groups = working.loc[series.index, group_column]
        sample_a = series[groups == group_a]
        sample_b = series[groups == group_b]
        if sample_a.empty:
            raise ToolExecutionError(f"No rows found where '{group_column}' == {group_a!r}.")
        if sample_b.empty:
            raise ToolExecutionError(f"No rows found where '{group_column}' == {group_b!r}.")
        if len(sample_a) < _MIN_GROUP_SIZE or len(sample_b) < _MIN_GROUP_SIZE:
            raise ToolExecutionError(
                f"Each group needs at least {_MIN_GROUP_SIZE} samples for a t-test "
                f"(group_a n={len(sample_a)}, group_b n={len(sample_b)})."
            )

        statistic, p_value = stats.ttest_ind(sample_a, sample_b, equal_var=False)
        dof = _welch_dof(sample_a, sample_b)
        return {
            "test": "two_sample_t_test",
            "column": column,
            "group_column": group_column,
            "group_a": {"label": str(group_a), "n": int(len(sample_a)), "mean": _round(sample_a.mean())},
            "group_b": {"label": str(group_b), "n": int(len(sample_b)), "mean": _round(sample_b.mean())},
            "statistic": _round(statistic),
            "p_value": _round(p_value, 6),
            "degrees_of_freedom": _round(dof),
            "alpha": alpha,
            "significant": bool(p_value < alpha),
        }

    if popmean is None:
        raise ToolExecutionError(
            "popmean is required for a one-sample t-test (or provide group_column/group_a/group_b for a two-sample test)."
        )
    if len(series) < _MIN_GROUP_SIZE:
        raise ToolExecutionError(f"At least {_MIN_GROUP_SIZE} samples are required for a t-test (found {len(series)}).")

    statistic, p_value = stats.ttest_1samp(series, popmean)
    return {
        "test": "one_sample_t_test",
        "column": column,
        "popmean": popmean,
        "n": int(len(series)),
        "mean": _round(series.mean()),
        "statistic": _round(statistic),
        "p_value": _round(p_value, 6),
        "degrees_of_freedom": int(len(series) - 1),
        "alpha": alpha,
        "significant": bool(p_value < alpha),
    }


def chi_square_test(
    df: pd.DataFrame,
    column_a: str,
    column_b: str,
    alpha: float = 0.05,
    filters: list[dict] | None = None,
) -> dict:
    """Chi-square test of independence between two categorical columns."""
    working = apply_filters(df, filters) if filters else df
    if working.empty:
        raise ToolExecutionError("No rows match the given filters.")
    _validate_alpha(alpha)

    missing = [c for c in (column_a, column_b) if c not in working.columns]
    if missing:
        raise ToolExecutionError(f"Unknown column(s): {', '.join(missing)}")

    sub = working[[column_a, column_b]].dropna()
    if sub.empty:
        raise ToolExecutionError(f"No non-null rows available for both '{column_a}' and '{column_b}'.")

    contingency = pd.crosstab(sub[column_a], sub[column_b])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        raise ToolExecutionError(
            f"Both '{column_a}' and '{column_b}' need at least 2 distinct categories for a chi-square test."
        )

    statistic, p_value, dof, _expected = stats.chi2_contingency(contingency)
    table = {
        str(row): {str(col): int(contingency.loc[row, col]) for col in contingency.columns}
        for row in contingency.index
    }
    return {
        "test": "chi_square_independence",
        "column_a": column_a,
        "column_b": column_b,
        "statistic": _round(statistic),
        "p_value": _round(p_value, 6),
        "degrees_of_freedom": int(dof),
        "alpha": alpha,
        "significant": bool(p_value < alpha),
        "contingency_table": table,
    }


def anova_test(
    df: pd.DataFrame,
    value_column: str,
    group_column: str,
    alpha: float = 0.05,
    filters: list[dict] | None = None,
) -> dict:
    """One-way ANOVA of `value_column` across the groups in `group_column`."""
    working = apply_filters(df, filters) if filters else df
    if working.empty:
        raise ToolExecutionError("No rows match the given filters.")
    _validate_alpha(alpha)

    series = _numeric_series(working, value_column)
    if group_column not in working.columns:
        raise ToolExecutionError(f"Unknown column '{group_column}'.")

    groups_series = working.loc[series.index, group_column]
    group_names = list(pd.unique(groups_series.dropna()))
    if len(group_names) < 2:
        raise ToolExecutionError(f"'{group_column}' needs at least 2 distinct groups for ANOVA.")

    samples = []
    group_stats: dict = {}
    for name in group_names:
        sample = series[groups_series == name]
        if len(sample) < _MIN_GROUP_SIZE:
            raise ToolExecutionError(
                f"Group '{name}' has only {len(sample)} sample(s); ANOVA requires at least "
                f"{_MIN_GROUP_SIZE} per group."
            )
        samples.append(sample)
        group_stats[str(name)] = {"n": int(len(sample)), "mean": _round(sample.mean())}

    statistic, p_value = stats.f_oneway(*samples)
    return {
        "test": "one_way_anova",
        "value_column": value_column,
        "group_column": group_column,
        "groups": group_stats,
        "statistic": _round(statistic),
        "p_value": _round(p_value, 6),
        "alpha": alpha,
        "significant": bool(p_value < alpha),
    }


def confidence_interval(
    df: pd.DataFrame,
    column: str,
    confidence: float = 0.95,
    filters: list[dict] | None = None,
) -> dict:
    """Mean, confidence-interval bounds, and margin of error for a numeric column."""
    working = apply_filters(df, filters) if filters else df
    if working.empty:
        raise ToolExecutionError("No rows match the given filters.")
    if not 0 < confidence < 1:
        raise ToolExecutionError("confidence must be between 0 and 1.")

    series = _numeric_series(working, column)
    n = len(series)
    if n < _MIN_GROUP_SIZE:
        raise ToolExecutionError(f"At least {_MIN_GROUP_SIZE} samples are required for a confidence interval (found {n}).")

    mean = float(series.mean())
    sem = float(stats.sem(series))
    if sem == 0:
        lower, upper, margin = mean, mean, 0.0
    else:
        lower, upper = stats.t.interval(confidence, df=n - 1, loc=mean, scale=sem)
        margin = (upper - lower) / 2

    return {
        "column": column,
        "n": int(n),
        "mean": _round(mean),
        "confidence": confidence,
        "lower_bound": _round(lower),
        "upper_bound": _round(upper),
        "margin_of_error": _round(margin),
    }


def effect_size(
    df: pd.DataFrame,
    column: str,
    group_column: str,
    group_a: object,
    group_b: object,
    filters: list[dict] | None = None,
) -> dict:
    """Cohen's d for the two-sample case (`column` split by `group_column`)."""
    working = apply_filters(df, filters) if filters else df
    if working.empty:
        raise ToolExecutionError("No rows match the given filters.")

    series = _numeric_series(working, column)
    if group_column not in working.columns:
        raise ToolExecutionError(f"Unknown column '{group_column}'.")

    groups = working.loc[series.index, group_column]
    sample_a = series[groups == group_a]
    sample_b = series[groups == group_b]
    if sample_a.empty:
        raise ToolExecutionError(f"No rows found where '{group_column}' == {group_a!r}.")
    if sample_b.empty:
        raise ToolExecutionError(f"No rows found where '{group_column}' == {group_b!r}.")
    if len(sample_a) < _MIN_GROUP_SIZE or len(sample_b) < _MIN_GROUP_SIZE:
        raise ToolExecutionError(
            f"Each group needs at least {_MIN_GROUP_SIZE} samples to compute an effect size "
            f"(group_a n={len(sample_a)}, group_b n={len(sample_b)})."
        )

    n1, n2 = len(sample_a), len(sample_b)
    var1, var2 = sample_a.var(ddof=1), sample_b.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        raise ToolExecutionError("Cannot compute effect size: pooled standard deviation is zero.")

    d = float((sample_a.mean() - sample_b.mean()) / pooled_std)
    ad = abs(d)
    if ad >= 0.8:
        magnitude = "large"
    elif ad >= 0.5:
        magnitude = "medium"
    elif ad >= 0.2:
        magnitude = "small"
    else:
        magnitude = "negligible"

    return {
        "column": column,
        "group_column": group_column,
        "group_a": {"label": str(group_a), "n": int(n1), "mean": _round(sample_a.mean())},
        "group_b": {"label": str(group_b), "n": int(n2), "mean": _round(sample_b.mean())},
        "cohens_d": _round(d),
        "magnitude": magnitude,
    }


def _numeric_series(working: pd.DataFrame, column: str) -> pd.Series:
    if column not in working.columns:
        raise ToolExecutionError(f"Unknown column '{column}'.")
    series = working[column]
    if not pd.api.types.is_numeric_dtype(series):
        raise ToolExecutionError(f"Column '{column}' is not numeric.")
    return series.dropna()


def _validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ToolExecutionError("alpha must be between 0 and 1.")


def _welch_dof(a: pd.Series, b: pd.Series) -> float:
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    n1, n2 = len(a), len(b)
    numerator = (v1 / n1 + v2 / n2) ** 2
    denominator = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    if denominator == 0:
        return float(n1 + n2 - 2)
    return float(numerator / denominator)


def _round(value: float | None, ndigits: int = 4) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), ndigits)
