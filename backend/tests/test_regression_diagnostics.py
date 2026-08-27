from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.tools.errors import ToolExecutionError
from app.tools.regression_diagnostics import outlier_analysis_multivariate, regression_diagnostics

DEMO_DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "demo" / "sales_data.xlsx"


# --- regression_diagnostics --------------------------------------------------


def test_regression_diagnostics_flags_known_multicollinearity():
    rng = np.random.default_rng(1)
    n = 100
    x1 = rng.normal(0, 1, n)
    x2 = x1 + rng.normal(0, 0.01, n)  # near-duplicate of x1 -> severe VIF
    x3 = rng.normal(0, 1, n)  # independent -> low VIF
    y = 2 * x1 + 3 * x3 + rng.normal(0, 0.1, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})

    result = regression_diagnostics(df, target_column="y", feature_columns=["x1", "x2", "x3"])

    vif = result["multicollinearity"]["vif"]
    assert vif["x1"] >= 10 or vif["x1"] is None
    assert vif["x2"] >= 10 or vif["x2"] is None
    assert vif["x3"] < 5
    assert "x1" in result["multicollinearity"]["flag"] and "x2" in result["multicollinearity"]["flag"]
    assert result["base_regression"]["r_squared"] > 0.9


def test_regression_diagnostics_no_multicollinearity_for_independent_features():
    rng = np.random.default_rng(2)
    n = 100
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = x1 + x2 + rng.normal(0, 0.1, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    result = regression_diagnostics(df, target_column="y", feature_columns=["x1", "x2"])
    vif = result["multicollinearity"]["vif"]
    assert vif["x1"] < 5
    assert vif["x2"] < 5
    assert "No meaningful multicollinearity" in result["multicollinearity"]["flag"]


def test_regression_diagnostics_single_feature_skips_vif():
    rng = np.random.default_rng(3)
    n = 50
    x1 = rng.normal(0, 1, n)
    y = 2 * x1 + rng.normal(0, 0.1, n)
    df = pd.DataFrame({"x1": x1, "y": y})

    result = regression_diagnostics(df, target_column="y", feature_columns=["x1"])
    assert result["multicollinearity"]["vif"] is None
    assert "Not applicable" in result["multicollinearity"]["flag"]


def test_regression_diagnostics_includes_normality_and_heteroscedasticity_checks():
    rng = np.random.default_rng(4)
    n = 60
    x1 = rng.normal(0, 1, n)
    y = 3 * x1 + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"x1": x1, "y": y})

    result = regression_diagnostics(df, target_column="y", feature_columns=["x1"])
    assert result["residual_normality"]["test"] == "shapiro_wilk"
    assert result["residual_normality"]["p_value"] is not None
    assert result["heteroscedasticity"]["test"] == "breusch_pagan"
    assert result["heteroscedasticity"]["p_value"] is not None
    assert "flag" in result["residual_normality"]
    assert "flag" in result["heteroscedasticity"]


def test_regression_diagnostics_detects_heteroscedasticity():
    # Residual variance grows with x -> classic heteroscedastic setup.
    rng = np.random.default_rng(5)
    n = 300
    x1 = rng.uniform(1, 50, n)
    noise = rng.normal(0, 1, n) * x1  # variance scales with x1
    y = 2 * x1 + noise
    df = pd.DataFrame({"x1": x1, "y": y})

    result = regression_diagnostics(df, target_column="y", feature_columns=["x1"])
    assert result["heteroscedasticity"]["p_value"] < 0.05
    assert "heteroscedasticity" in result["heteroscedasticity"]["flag"].lower()


def test_regression_diagnostics_requires_minimum_rows():
    df = pd.DataFrame({"x1": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
    with pytest.raises(ToolExecutionError):
        regression_diagnostics(df, target_column="y", feature_columns=["x1"])


def test_regression_diagnostics_applies_filters():
    rng = np.random.default_rng(6)
    n = 60
    x1 = rng.normal(0, 1, n)
    y = 2 * x1 + rng.normal(0, 0.1, n)
    region = ["keep"] * 40 + ["drop"] * 20
    df = pd.DataFrame({"x1": x1, "y": y, "region": region})

    result = regression_diagnostics(
        df, target_column="y", feature_columns=["x1"], filters=[{"column": "region", "op": "==", "value": "keep"}]
    )
    assert result["n_observations"] == 40


# --- outlier_analysis_multivariate --------------------------------------------


def _df_with_joint_outliers() -> pd.DataFrame:
    rng = np.random.default_rng(8)
    n = 200
    c1 = rng.normal(0, 1, n)
    c2 = rng.normal(0, 1, n)
    # Injected points: individually unremarkable on each axis (well within the
    # normal per-axis range of roughly [-3.5, 3.5]) but jointly unusual because
    # they sit far from the (positively correlated-by-chance) bulk of the data
    # along the anti-diagonal.
    outliers_c1 = np.array([3.0, -3.0, 3.2])
    outliers_c2 = np.array([-3.0, 3.0, -3.1])
    df = pd.DataFrame({"a": np.concatenate([c1, outliers_c1]), "b": np.concatenate([c2, outliers_c2])})
    return df


def test_outlier_analysis_multivariate_mahalanobis_flags_joint_outliers():
    df = _df_with_joint_outliers()
    result = outlier_analysis_multivariate(df, columns=["a", "b"], method="mahalanobis", contamination=0.02)

    flagged_indices = {o["row_index"] for o in result["outliers"]}
    injected_indices = {200, 201, 202}
    assert injected_indices.issubset(flagged_indices)
    assert result["outlier_count"] >= 3
    assert result["method"] == "mahalanobis"
    assert result["threshold"] is not None


def test_outlier_analysis_multivariate_elliptic_envelope_flags_joint_outliers():
    df = _df_with_joint_outliers()
    result = outlier_analysis_multivariate(df, columns=["a", "b"], method="elliptic_envelope", contamination=0.03)

    flagged_indices = {o["row_index"] for o in result["outliers"]}
    injected_indices = {200, 201, 202}
    assert len(injected_indices & flagged_indices) >= 2


def _df_with_exact_linear_dependency() -> pd.DataFrame:
    """revenue/cost/profit shape: profit is an exact linear combination of the other
    two (profit = revenue - cost), making the 3-column covariance matrix singular --
    the real shape that exposed this bug against the project's own demo dataset."""
    rng = np.random.default_rng(9)
    n = 500
    revenue = rng.normal(400, 150, n)
    cost = rng.normal(250, 100, n)
    profit = revenue - cost
    return pd.DataFrame({"revenue": revenue, "cost": cost, "profit": profit})


def test_outlier_analysis_multivariate_rejects_exactly_collinear_columns():
    """Real bug found via this project's own hard-benchmark work (final_100_cases.json
    case out3, columns=["revenue","cost","profit"]): the Mahalanobis method's
    covariance matrix is singular whenever one column is an exact/near-exact linear
    combination of the others (condition number ~8e16, rank 2 of 3 on the real demo
    dataset), and `np.linalg.inv` does NOT raise for a matrix this ill-conditioned --
    it silently returns a garbage inverse whose quadratic form goes negative for a
    meaningful fraction of rows (540/4000 real rows), which is mathematically
    impossible for a true Mahalanobis distance and turns into NaN after `np.sqrt`.
    Must now refuse cleanly instead of returning corrupted results."""
    df = _df_with_exact_linear_dependency()
    with pytest.raises(ToolExecutionError, match="linearly dependent"):
        outlier_analysis_multivariate(df, columns=["revenue", "cost", "profit"], method="mahalanobis")


def test_outlier_analysis_multivariate_mahalanobis_still_works_on_non_collinear_columns():
    """Regression guard: the new condition-number check must not reject ordinary,
    non-collinear columns -- only genuinely near-singular ones."""
    df = _df_with_joint_outliers()
    result = outlier_analysis_multivariate(df, columns=["a", "b"], method="mahalanobis", contamination=0.02)
    assert result["outlier_count"] >= 3


def test_outlier_analysis_multivariate_elliptic_envelope_unaffected_by_collinearity_guard():
    """The condition-number guard is specific to the mahalanobis path (it needs a
    literal matrix inverse); elliptic_envelope uses a different, regularized
    covariance estimator and should be unaffected -- offered as the tool's own
    suggested alternative in the ToolExecutionError message above."""
    df = _df_with_exact_linear_dependency()
    result = outlier_analysis_multivariate(df, columns=["revenue", "cost", "profit"], method="elliptic_envelope", contamination=0.05)
    assert result["method"] == "elliptic_envelope"


def test_outlier_analysis_multivariate_univariate_marginals_look_normal():
    """Confirms the injected points are NOT flagged by simple per-axis z-score
    thresholds -- proving this is a genuinely different capability from the
    existing univariate `detect_anomalies`, not a restatement of it."""
    df = _df_with_joint_outliers()
    for col in ("a", "b"):
        z = (df[col] - df[col].mean()) / df[col].std()
        # None of the injected rows exceed a conventional |z| > 3 threshold on
        # either axis individually.
        assert (z.iloc[200:203].abs() < 3.5).all()


def test_outlier_analysis_multivariate_requires_two_columns():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    with pytest.raises(ToolExecutionError):
        outlier_analysis_multivariate(df, columns=["a"])


def test_outlier_analysis_multivariate_rejects_bad_contamination():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
    with pytest.raises(ToolExecutionError):
        outlier_analysis_multivariate(df, columns=["a", "b"], contamination=0.9)


def test_outlier_analysis_multivariate_rejects_unknown_method():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
    with pytest.raises(ToolExecutionError):
        outlier_analysis_multivariate(df, columns=["a", "b"], method="bogus")


def test_outlier_analysis_multivariate_rejects_non_numeric_column():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"]})
    with pytest.raises(ToolExecutionError):
        outlier_analysis_multivariate(df, columns=["a", "b"])


# --- Real demo dataset --------------------------------------------------------


def test_regression_diagnostics_and_outliers_on_demo_sales_data():
    df = pd.read_excel(DEMO_DATASET_PATH)

    reg = regression_diagnostics(df, target_column="revenue", feature_columns=["quantity", "unit_price"])
    assert reg["n_observations"] > 0
    assert reg["multicollinearity"]["vif"] is not None

    outliers = outlier_analysis_multivariate(df, columns=["quantity", "revenue", "profit"], contamination=0.02)
    assert outliers["n_rows_used"] == len(df)
    assert outliers["outlier_count"] >= 0
    assert outliers["outliers_returned"] <= 50
