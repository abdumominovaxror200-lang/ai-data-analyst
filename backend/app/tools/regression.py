from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

from app.tools.errors import ToolExecutionError
from app.tools.filtering import apply_filters

_MIN_SAMPLES = 3


def linear_regression(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    alpha: float = 0.05,
    filters: list[dict] | None = None,
) -> dict:
    """Ordinary least squares regression of `target_column` on `feature_columns`.

    Returns per-feature coefficients/std-errors/p-values, R-squared, and a
    plain-language summary of which features are statistically significant
    predictors at the given `alpha`.
    """
    working = apply_filters(df, filters) if filters else df
    if working.empty:
        raise ToolExecutionError("No rows match the given filters.")
    if not 0 < alpha < 1:
        raise ToolExecutionError("alpha must be between 0 and 1.")
    if not feature_columns:
        raise ToolExecutionError("At least one feature column is required.")

    missing = [c for c in [target_column, *feature_columns] if c not in working.columns]
    if missing:
        raise ToolExecutionError(f"Unknown column(s): {', '.join(missing)}")
    if target_column in feature_columns:
        raise ToolExecutionError("target_column cannot also be a feature column.")
    if len(set(feature_columns)) != len(feature_columns):
        raise ToolExecutionError("feature_columns contains duplicates.")

    non_numeric = [c for c in [target_column, *feature_columns] if not pd.api.types.is_numeric_dtype(working[c])]
    if non_numeric:
        raise ToolExecutionError(f"Column(s) must be numeric: {', '.join(non_numeric)}")

    subset = working[[target_column, *feature_columns]].dropna()
    if len(subset) < _MIN_SAMPLES:
        raise ToolExecutionError(
            f"At least {_MIN_SAMPLES} complete rows (no missing values) are required for regression "
            f"(found {len(subset)})."
        )
    if len(subset) <= len(feature_columns) + 1:
        raise ToolExecutionError(
            "Not enough observations relative to the number of features: need more rows than "
            "features + 1 to fit a regression."
        )

    y = subset[target_column].astype(float)
    X = sm.add_constant(subset[feature_columns].astype(float), has_constant="add")

    try:
        model = sm.OLS(y, X).fit()
    except Exception as exc:  # e.g. perfectly collinear features -> singular matrix
        raise ToolExecutionError(f"Could not fit regression model: {exc}") from exc

    coefficients: dict = {}
    for name in feature_columns:
        p_value = model.pvalues.get(name)
        coefficients[name] = {
            "coefficient": _round(model.params.get(name)),
            "std_err": _round(model.bse.get(name)),
            "p_value": _round(p_value, 6),
            "significant": bool(pd.notna(p_value) and p_value < alpha),
        }

    significant_features = [name for name, c in coefficients.items() if c["significant"]]
    if significant_features:
        summary = f"Statistically significant predictors at alpha={alpha}: {', '.join(significant_features)}."
    else:
        summary = f"No feature was a statistically significant predictor of '{target_column}' at alpha={alpha}."

    return {
        "target_column": target_column,
        "feature_columns": feature_columns,
        "n_observations": int(len(subset)),
        "intercept": _round(model.params.get("const")),
        "coefficients": coefficients,
        "r_squared": _round(model.rsquared),
        "adj_r_squared": _round(model.rsquared_adj),
        "f_statistic": _round(model.fvalue),
        "f_p_value": _round(model.f_pvalue, 6),
        "alpha": alpha,
        "significant_features": significant_features,
        "summary": summary,
    }


def _round(value: float | None, ndigits: int = 4) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), ndigits)
