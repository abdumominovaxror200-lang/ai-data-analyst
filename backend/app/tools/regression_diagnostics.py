from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.covariance import EllipticEnvelope
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from app.tools.errors import ToolExecutionError
from app.tools.filtering import apply_filters
from app.tools.regression import linear_regression

_MIN_SAMPLES = 8  # Shapiro-Wilk and Breusch-Pagan both need more than the bare regression minimum to be meaningful
_SHAPIRO_MAX_N = 5000  # scipy's Shapiro-Wilk is documented as unreliable/slow well past this


def regression_diagnostics(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    filters: list[dict] | None = None,
) -> dict:
    """Extends `app.tools.regression.linear_regression` (coefficients, R^2,
    per-feature p-values) with the diagnostics that tell you whether the
    regression's assumptions actually hold — a high R^2 with, say, severe
    multicollinearity or heteroscedastic residuals can still be misleading.

    Implementation choice: this refits the model directly via `statsmodels`
    (rather than calling `linear_regression` and trying to recover residuals
    from its rounded summary output) so the diagnostic tests run on the
    *unrounded* fitted values and residuals — but it still calls
    `linear_regression` once for the base coefficients/R^2/p-values, so the
    caller gets numbers from the exact same code path used everywhere else
    in the app rather than a second, potentially-divergent implementation of
    the same arithmetic.

    Three checks, each with a plain-language flag:
    - **Residual normality** (Shapiro-Wilk test on the OLS residuals): many
      of OLS's inferential guarantees (the p-values `linear_regression`
      reports) assume normally-distributed residuals. Flags "residuals look
      normal" (p >= 0.05, fail to reject normality) or "residuals do not
      look normal (Shapiro-Wilk p < 0.05)". Skipped (null) above
      `_SHAPIRO_MAX_N` observations, per scipy's own documented guidance
      that the test loses reliability at large n.
    - **Heteroscedasticity** (Breusch-Pagan test): checks whether residual
      variance depends on the fitted values / predictors — if it does,
      standard-error-based significance calls become unreliable even though
      the coefficients themselves stay unbiased. Flags "no evidence of
      heteroscedasticity" or "possible heteroscedasticity detected
      (Breusch-Pagan p < 0.05)".
    - **Multicollinearity** (Variance Inflation Factor per feature): a VIF
      measures how much a feature's coefficient variance is inflated by its
      linear relationship with the *other* features. VIF >= 10 is the
      standard rule-of-thumb line for "seriously collinear" (5-10 is
      flagged as moderate); a feature perfectly explained by the others
      (the classic "x2 = x1 + noise" case) will show a very large or
      infinite VIF. Requires 2+ features (VIF is undefined for a single
      predictor).
    """
    working = apply_filters(df, filters) if filters else df

    base = linear_regression(working, target_column, feature_columns, filters=None)

    subset = working[[target_column, *feature_columns]].dropna()
    if len(subset) < _MIN_SAMPLES:
        raise ToolExecutionError(
            f"At least {_MIN_SAMPLES} complete rows are required for regression diagnostics "
            f"(found {len(subset)}); linear_regression's own minimum is lower because it doesn't "
            f"run statistical tests that need more data to be meaningful."
        )

    y = subset[target_column].astype(float)
    X = sm.add_constant(subset[feature_columns].astype(float), has_constant="add")
    model = sm.OLS(y, X).fit()
    residuals = model.resid
    fitted = model.fittedvalues

    # --- Residual normality ------------------------------------------------
    normality: dict = {}
    if len(residuals) <= _SHAPIRO_MAX_N:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        normality = {
            "test": "shapiro_wilk",
            "statistic": round(float(shapiro_stat), 4),
            "p_value": round(float(shapiro_p), 6),
            "flag": (
                "Residuals look approximately normal (fail to reject normality at alpha=0.05)."
                if shapiro_p >= 0.05
                else "Residuals do not look normal (Shapiro-Wilk p < 0.05) — p-value-based inference "
                "on the coefficients may be unreliable."
            ),
        }
    else:
        normality = {
            "test": "shapiro_wilk",
            "statistic": None,
            "p_value": None,
            "flag": f"Skipped: Shapiro-Wilk is not reliable above {_SHAPIRO_MAX_N} observations "
            f"(n={len(residuals)}).",
        }

    # --- Heteroscedasticity --------------------------------------------------
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X)
    heteroscedasticity = {
        "test": "breusch_pagan",
        "statistic": round(float(bp_stat), 4),
        "p_value": round(float(bp_p), 6),
        "flag": (
            "No evidence of heteroscedasticity (fail to reject constant variance at alpha=0.05)."
            if bp_p >= 0.05
            else "Possible heteroscedasticity detected (Breusch-Pagan p < 0.05) — standard errors and "
            "p-values may be unreliable even though coefficients remain unbiased."
        ),
    }

    # --- Multicollinearity (VIF) ---------------------------------------------
    multicollinearity = None
    if len(feature_columns) >= 2:
        vif_values = {}
        X_features = X.to_numpy(dtype=float)
        for i, name in enumerate(["const", *feature_columns]):
            if name == "const":
                continue
            vif = variance_inflation_factor(X_features, i)
            vif_values[name] = None if np.isinf(vif) or np.isnan(vif) else round(float(vif), 4)

        flagged = {
            name: v for name, v in vif_values.items() if v is None or v >= 10
        }
        moderate = {
            name: v for name, v in vif_values.items() if v is not None and 5 <= v < 10
        }
        if flagged:
            flag = (
                f"Severe multicollinearity detected (VIF >= 10, or undefined/infinite): "
                f"{', '.join(flagged.keys())}. Coefficient estimates for these features are unstable "
                f"and hard to interpret individually."
            )
        elif moderate:
            flag = f"Moderate multicollinearity (5 <= VIF < 10): {', '.join(moderate.keys())}."
        else:
            flag = "No meaningful multicollinearity (all VIF < 5)."

        multicollinearity = {
            "test": "variance_inflation_factor",
            "vif": vif_values,
            "flag": flag,
        }
    else:
        multicollinearity = {
            "test": "variance_inflation_factor",
            "vif": None,
            "flag": "Not applicable: VIF requires 2 or more feature columns.",
        }

    return {
        "target_column": target_column,
        "feature_columns": feature_columns,
        "n_observations": int(len(subset)),
        "base_regression": base,
        "residual_normality": normality,
        "heteroscedasticity": heteroscedasticity,
        "multicollinearity": multicollinearity,
    }


_OUTLIER_METHODS = {"mahalanobis", "elliptic_envelope"}
_MAX_OUTLIERS_RETURNED = 50


def outlier_analysis_multivariate(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "mahalanobis",
    contamination: float = 0.05,
    filters: list[dict] | None = None,
) -> dict:
    """Multivariate outlier detection: flags rows that are jointly unusual
    across `columns` even when every individual column value looks normal on
    its own — the case the existing univariate `detect_anomalies`
    (`app/tools/anomaly.py`, IQR/z-score on *one* column at a time)
    structurally cannot catch. E.g. (age=70, income=200000) can be entirely
    unremarkable on age alone and on income alone, while being a striking
    combination jointly (a 70-year-old with an entry-level income, or vice
    versa) — only a method that looks at the columns together will flag it.

    Two methods, both operating on the same underlying idea (distance from
    the multivariate center, scaled by how the data covaries):

    - `"mahalanobis"` (default): Mahalanobis distance of every row from the
      column-wise mean, using the sample covariance matrix — the
      multivariate generalization of "how many standard deviations away is
      this point," accounting for correlation between columns (unlike
      Euclidean distance, it doesn't over- or under-weight correlated
      features). Under approximate multivariate normality, squared
      Mahalanobis distance follows a chi-squared distribution with
      `len(columns)` degrees of freedom, so the flagging threshold is the
      `(1 - contamination)` quantile of that chi-squared distribution — a
      statistically grounded cutoff rather than an arbitrary one.
    - `"elliptic_envelope"`: `sklearn.covariance.EllipticEnvelope`, which
      fits a robust (outlier-resistant) covariance estimate instead of the
      plain sample covariance, then flags the `contamination` fraction of
      points furthest from that robust center. More resistant to the
      outliers themselves distorting the very covariance matrix used to
      detect them ("masking"), at the cost of being a fitted/randomized
      procedure rather than a closed-form distance.

    `contamination` (default 0.05, i.e. ~5%) is the expected outlier
    fraction — for `"elliptic_envelope"` it directly controls how many
    points get flagged; for `"mahalanobis"` it sets the chi-squared
    threshold's quantile, so it's an expected rate under normality, not a
    hard cap (an unusually heavy-tailed dataset can flag more or fewer than
    exactly `contamination` fraction).

    Requires at least 2 columns (a single column has no "jointly unusual" to
    detect — that's what `detect_anomalies` is for) and more rows than
    columns (the covariance matrix is singular otherwise).
    """
    working = apply_filters(df, filters) if filters else df
    if not columns or len(columns) < 2:
        raise ToolExecutionError("At least two columns are required for multivariate outlier detection.")
    if len(set(columns)) != len(columns):
        raise ToolExecutionError("columns contains duplicates.")
    if method not in _OUTLIER_METHODS:
        raise ToolExecutionError(f"method must be one of {sorted(_OUTLIER_METHODS)}.")
    if not 0 < contamination < 0.5:
        raise ToolExecutionError("contamination must be between 0 and 0.5.")

    missing = [c for c in columns if c not in working.columns]
    if missing:
        raise ToolExecutionError(f"Unknown column(s): {', '.join(missing)}")
    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(working[c])]
    if non_numeric:
        raise ToolExecutionError(f"Column(s) must be numeric: {', '.join(non_numeric)}")

    subset = working[columns].dropna()
    if len(subset) <= len(columns):
        raise ToolExecutionError(
            f"Need more complete rows than columns for a non-singular covariance matrix "
            f"(found {len(subset)} rows for {len(columns)} columns)."
        )

    X = subset.to_numpy(dtype=float)

    if method == "mahalanobis":
        mean = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)
        diff = X - mean
        distances_sq = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
        threshold_sq = float(stats.chi2.ppf(1 - contamination, df=len(columns)))
        mask = distances_sq > threshold_sq
        scores = np.sqrt(distances_sq)
        threshold_used = float(np.sqrt(threshold_sq))
    else:
        model = EllipticEnvelope(contamination=contamination, random_state=42)
        predictions = model.fit_predict(X)  # -1 = outlier, 1 = inlier
        mask = predictions == -1
        scores = -model.score_samples(X)  # higher = more outlying, for consistent sorting below
        threshold_used = None

    flagged_idx = np.where(mask)[0]
    order = np.argsort(-scores[flagged_idx]) if len(flagged_idx) else np.array([], dtype=int)
    flagged_idx_sorted = flagged_idx[order]

    outliers = []
    for pos in flagged_idx_sorted[:_MAX_OUTLIERS_RETURNED]:
        row_index = subset.index[pos]
        values = {col: round(float(subset.iloc[pos][col]), 4) for col in columns}
        outliers.append(
            {
                "row_index": int(row_index),
                "values": values,
                "score": round(float(scores[pos]), 4),
            }
        )

    return {
        "columns": columns,
        "method": method,
        "contamination": contamination,
        "threshold": round(threshold_used, 4) if threshold_used is not None else None,
        "n_rows_used": int(len(subset)),
        "outlier_count": int(mask.sum()),
        "outlier_pct": round(float(mask.mean()) * 100, 2),
        "outliers_returned": len(outliers),
        "outliers": outliers,
    }
