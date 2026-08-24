from __future__ import annotations

import pandas as pd
import pytest

from app.tools.errors import ToolExecutionError
from app.tools.regression import linear_regression


def test_linear_regression_recovers_known_slope():
    # y = 2x + 1 exactly -> should recover coefficient ~2, intercept ~1, R^2 ~1.
    x = list(range(1, 11))
    y = [2 * v + 1 for v in x]
    df = pd.DataFrame({"x": x, "y": y})

    result = linear_regression(df, target_column="y", feature_columns=["x"])

    assert result["coefficients"]["x"]["coefficient"] == pytest.approx(2.0, abs=1e-6)
    assert result["intercept"] == pytest.approx(1.0, abs=1e-6)
    assert result["r_squared"] == pytest.approx(1.0, abs=1e-6)
    assert result["coefficients"]["x"]["significant"] is True
    assert "x" in result["significant_features"]
    assert result["n_observations"] == 10


def test_linear_regression_with_noise_recovers_approximate_slope():
    # y = 2x + small deterministic noise -> coefficient should be close to 2, R^2 high but not 1.
    x = list(range(1, 21))
    noise = [((-1) ** i) * (i % 3) * 0.5 for i in range(20)]
    y = [2 * xi + n for xi, n in zip(x, noise)]
    df = pd.DataFrame({"x": x, "y": y})

    result = linear_regression(df, target_column="y", feature_columns=["x"])

    assert result["coefficients"]["x"]["coefficient"] == pytest.approx(2.0, abs=0.1)
    assert result["r_squared"] > 0.9


def test_linear_regression_multiple_features():
    # y = 3*x1 - 2*x2 + 5 exactly.
    x1 = list(range(1, 11))
    x2 = [v % 4 for v in range(1, 11)]
    y = [3 * a - 2 * b + 5 for a, b in zip(x1, x2)]
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    result = linear_regression(df, target_column="y", feature_columns=["x1", "x2"])

    assert result["coefficients"]["x1"]["coefficient"] == pytest.approx(3.0, abs=1e-6)
    assert result["coefficients"]["x2"]["coefficient"] == pytest.approx(-2.0, abs=1e-6)
    assert result["intercept"] == pytest.approx(5.0, abs=1e-6)
    assert result["r_squared"] == pytest.approx(1.0, abs=1e-6)


def test_linear_regression_identifies_non_significant_feature():
    # y depends only on x1; x2 is unrelated noise-like values -> should not be significant.
    x1 = list(range(1, 31))
    x2 = [((-1) ** i) * (i % 5) for i in range(30)]
    y = [4 * v + 2 for v in x1]
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    result = linear_regression(df, target_column="y", feature_columns=["x1", "x2"])

    assert result["coefficients"]["x1"]["significant"] is True
    assert result["coefficients"]["x2"]["significant"] is False
    assert "x2" not in result["significant_features"]


def test_linear_regression_unknown_column_raises():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    with pytest.raises(ToolExecutionError):
        linear_regression(df, target_column="y", feature_columns=["nope"])


def test_linear_regression_unknown_target_raises():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    with pytest.raises(ToolExecutionError):
        linear_regression(df, target_column="nope", feature_columns=["x"])


def test_linear_regression_non_numeric_feature_raises():
    df = pd.DataFrame({"x": ["a", "b", "c"], "y": [2, 4, 6]})
    with pytest.raises(ToolExecutionError):
        linear_regression(df, target_column="y", feature_columns=["x"])


def test_linear_regression_non_numeric_target_raises():
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    with pytest.raises(ToolExecutionError):
        linear_regression(df, target_column="y", feature_columns=["x"])


def test_linear_regression_requires_feature_columns():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    with pytest.raises(ToolExecutionError):
        linear_regression(df, target_column="y", feature_columns=[])


def test_linear_regression_target_cannot_be_feature():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    with pytest.raises(ToolExecutionError):
        linear_regression(df, target_column="y", feature_columns=["y"])


def test_linear_regression_insufficient_rows_raises():
    df = pd.DataFrame({"x": [1, 2], "y": [2, 4]})
    with pytest.raises(ToolExecutionError):
        linear_regression(df, target_column="y", feature_columns=["x"])


def test_linear_regression_too_few_rows_relative_to_features_raises():
    # 3 rows but 3 features (+intercept) -> not enough degrees of freedom.
    df = pd.DataFrame({"x1": [1, 2, 3], "x2": [2, 3, 4], "x3": [3, 4, 5], "y": [1, 2, 3]})
    with pytest.raises(ToolExecutionError):
        linear_regression(df, target_column="y", feature_columns=["x1", "x2", "x3"])


def test_linear_regression_applies_filters():
    x = list(range(1, 21))
    y = [2 * v for v in x]
    region = ["keep"] * 10 + ["drop"] * 10
    df = pd.DataFrame({"x": x, "y": y, "region": region})

    result = linear_regression(
        df,
        target_column="y",
        feature_columns=["x"],
        filters=[{"column": "region", "op": "==", "value": "keep"}],
    )
    assert result["n_observations"] == 10
