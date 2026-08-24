"""Automated exploratory data analysis (EDA).

This module composes the existing deterministic tools (`profile_dataset`,
`detect_anomalies`, `correlation_analysis`) plus new cardinality and
distribution-shape analysis into one "first pass" report that answers
"Analyze this dataset" the way a human analyst would: schema -> types ->
missingness -> duplicates -> cardinality -> distributions -> outliers ->
relationships -> a prioritized, plain-language list of things worth
looking at.

Every number here comes from pandas/numpy/scipy — no LLM ever computes a
statistic. The LLM's job is to read this structured output and narrate it.

Payload size discipline: like `app.tools.insights.generate_business_insights`,
this module never returns raw row data. Per-column detail lists are capped
(see `_MAX_DETAIL_COLUMNS`) and the outlier summaries strip the row-level
`anomalies` list that `detect_anomalies` normally returns, keeping only
counts/bounds. This avoids the project's documented history of 413
"payload too large" errors from verbose tool outputs.

--- Cardinality thresholds (see `analyze_cardinality`) ---
Given a column's non-null values (n = non-null count, u = distinct count,
top_pct = share of non-null values equal to the most common value):

- ``all_missing``            n == 0
- ``constant``                u <= 1
- ``boolean_like``            u == 2 and the two values normalize to a known
                               boolean pair (0/1, true/false, yes/no, y/n, t/f)
- ``near_constant``           top_pct >= 95.0
- ``continuous_numeric``      float dtype and none of the above matched
                               (high uniqueness is *expected* for a
                               continuous measurement, so it is not flagged
                               as an "ID-like" column even at 90%+ unique)
- ``unique_id_like``          unique_ratio (u / n) >= 0.98
- ``high_cardinality``        unique_ratio >= 0.5 OR u > 50
- ``low_cardinality_categorical``  everything else

--- Distribution thresholds (see `analyze_distributions`) ---
Skewness (scipy.stats.skew, Fisher-Pearson, bias-corrected=False):
  |skew| < 0.5           -> "roughly symmetric"
  0.5 <= |skew| < 1.0     -> "moderately {right,left}-skewed"
  |skew| >= 1.0           -> "highly {right,left}-skewed"
  (sign > 0 => right-skewed / long tail toward high values)

Excess kurtosis (scipy.stats.kurtosis, Fisher definition, normal == 0):
  |kurtosis| < 0.5        -> "normal-like tails (mesokurtic)"
  kurtosis >= 0.5         -> "heavy-tailed (leptokurtic) - outlier-prone"
  kurtosis <= -0.5        -> "light-tailed (platykurtic)"

Categorical balance: normalized Shannon entropy = H / log2(u), where
H = -sum(p_i * log2(p_i)) over category proportions p_i, and u = distinct
category count (normalized entropy is defined as 0.0 when u <= 1).
  normalized_entropy >= 0.8   -> "well balanced"
  0.5 <= normalized_entropy < 0.8 -> "moderately balanced"
  normalized_entropy < 0.5    -> "imbalanced"
`imbalance_ratio` = count of most common category / count of least common
category (>= 1.0; higher means more lopsided).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew as scipy_skew

from app.tools.anomaly import detect_anomalies
from app.tools.correlation import correlation_analysis
from app.tools.errors import ToolExecutionError
from app.tools.filtering import apply_filters
from app.tools.profiler import profile_dataset

# --- tunable thresholds (documented in the module docstring above) ---
NEAR_CONSTANT_TOP_PCT = 95.0
ID_LIKE_UNIQUE_RATIO = 0.98
HIGH_CARDINALITY_UNIQUE_RATIO = 0.5
HIGH_CARDINALITY_MIN_COUNT = 50

SKEW_MODERATE = 0.5
SKEW_HIGH = 1.0
KURTOSIS_NOTABLE = 0.5

ENTROPY_WELL_BALANCED = 0.8
ENTROPY_MODERATE = 0.5

# Payload-size guardrails: cap how many columns get full per-column detail
# in the bundled automated_eda report (the standalone tools have no cap).
_MAX_DETAIL_COLUMNS = 60
_MAX_OUTLIER_COLUMNS = 12
_MAX_PROBLEMS = 25

_BOOL_TOKEN_SETS = [
    frozenset({"0", "1"}),
    frozenset({"true", "false"}),
    frozenset({"yes", "no"}),
    frozenset({"y", "n"}),
    frozenset({"t", "f"}),
]

_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


def _normalize_token(value: object) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        fv = float(value)
        if fv.is_integer():
            return str(int(fv))
        return str(fv)
    return str(value).strip().lower()


def _is_boolean_like(non_null_uniques: np.ndarray) -> bool:
    if len(non_null_uniques) != 2:
        return False
    tokens = frozenset(_normalize_token(v) for v in non_null_uniques)
    return tokens in _BOOL_TOKEN_SETS


def _classify_column(series: pd.Series) -> dict:
    total = int(len(series))
    non_null = series.dropna()
    n = int(len(non_null))
    missing_count = total - n

    if n == 0:
        return {
            "name": str(series.name),
            "dtype": str(series.dtype),
            "non_null_count": 0,
            "missing_count": missing_count,
            "unique_count": 0,
            "unique_ratio": 0.0,
            "top_value": None,
            "top_value_count": 0,
            "top_value_pct": 0.0,
            "least_value": None,
            "least_value_count": 0,
            "classification": "all_missing",
            "note": "Column is entirely missing/null.",
        }

    value_counts = non_null.value_counts()
    unique_count = int(value_counts.shape[0])
    unique_ratio = round(unique_count / n, 4)
    top_value = value_counts.index[0]
    top_count = int(value_counts.iloc[0])
    top_pct = round(top_count / n * 100, 2)
    least_value = value_counts.index[-1]
    least_count = int(value_counts.iloc[-1])
    is_bool = _is_boolean_like(non_null.unique())
    is_float = pd.api.types.is_float_dtype(series)

    if unique_count <= 1:
        classification = "constant"
        note = f"Only one distinct value ('{top_value}') across {n} non-null rows — provides no analytical signal."
    elif is_bool:
        classification = "boolean_like"
        note = f"Exactly two values ('{least_value}'/'{top_value}') resembling a boolean encoding."
    elif top_pct >= NEAR_CONSTANT_TOP_PCT:
        classification = "near_constant"
        note = f"{top_pct}% of non-null values are '{top_value}' — column may add little analytical value."
    elif is_float:
        classification = "continuous_numeric"
        note = f"Continuous numeric column, {unique_count} distinct values ({unique_ratio * 100:.1f}% unique)."
    elif unique_ratio >= ID_LIKE_UNIQUE_RATIO:
        classification = "unique_id_like"
        note = f"{unique_ratio * 100:.1f}% of non-null values are unique — likely an identifier, not useful for aggregation/statistics."
    elif unique_ratio >= HIGH_CARDINALITY_UNIQUE_RATIO or unique_count > HIGH_CARDINALITY_MIN_COUNT:
        classification = "high_cardinality"
        note = f"{unique_count} distinct values ({unique_ratio * 100:.1f}% unique) — high cardinality; verify whether this is an identifier before grouping by it."
    else:
        classification = "low_cardinality_categorical"
        note = f"{unique_count} distinct values — suitable for grouping/categorical analysis."

    return {
        "name": str(series.name),
        "dtype": str(series.dtype),
        "non_null_count": n,
        "missing_count": missing_count,
        "unique_count": unique_count,
        "unique_ratio": unique_ratio,
        "top_value": str(top_value),
        "top_value_count": top_count,
        "top_value_pct": top_pct,
        "least_value": str(least_value),
        "least_value_count": least_count,
        "classification": classification,
        "note": note,
    }


def _cardinality_for_df(df: pd.DataFrame) -> list[dict]:
    return [_classify_column(df[col]) for col in df.columns]


def analyze_cardinality(df: pd.DataFrame, filters: list[dict] | None = None) -> dict:
    """Classify every column's cardinality pattern.

    Categories: constant, near_constant, boolean_like, continuous_numeric,
    unique_id_like, high_cardinality, low_cardinality_categorical,
    all_missing. See module docstring for exact thresholds.
    """
    working = apply_filters(df, filters) if filters else df
    if working.empty:
        raise ToolExecutionError("No rows match the given filters.")

    columns = _cardinality_for_df(working)
    summary: dict[str, int] = {}
    for c in columns:
        summary[c["classification"]] = summary.get(c["classification"], 0) + 1

    return {"row_count": int(len(working)), "columns": columns, "summary": summary}


def _skew_kurtosis_for_series(series: pd.Series) -> dict:
    values = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    count = int(len(values))
    base = {
        "name": str(series.name),
        "count": count,
        "mean": None,
        "median": None,
        "std": None,
        "skewness": None,
        "kurtosis": None,
        "skew_label": None,
        "kurtosis_label": None,
    }
    if count == 0:
        base["skew_label"] = "no data"
        return base

    base["mean"] = round(float(values.mean()), 4)
    base["median"] = round(float(values.median()), 4)
    base["std"] = round(float(values.std()), 4) if count > 1 else 0.0

    if count < 3 or float(values.std()) == 0.0:
        base["skew_label"] = "undefined (constant or too few values)"
        base["kurtosis_label"] = "undefined (constant or too few values)"
        return base

    arr = values.to_numpy(dtype=float)
    skew_val = float(scipy_skew(arr, bias=False))
    kurt_val = float(scipy_kurtosis(arr, fisher=True, bias=False))
    if math.isnan(skew_val) or math.isnan(kurt_val):
        base["skew_label"] = "undefined (constant or too few values)"
        base["kurtosis_label"] = "undefined (constant or too few values)"
        return base

    base["skewness"] = round(skew_val, 4)
    base["kurtosis"] = round(kurt_val, 4)

    abs_skew = abs(skew_val)
    if abs_skew < SKEW_MODERATE:
        base["skew_label"] = "roughly symmetric"
    else:
        direction = "right-skewed" if skew_val > 0 else "left-skewed"
        magnitude = "highly" if abs_skew >= SKEW_HIGH else "moderately"
        base["skew_label"] = f"{magnitude} {direction}"

    if kurt_val >= KURTOSIS_NOTABLE:
        base["kurtosis_label"] = "heavy-tailed (leptokurtic) - outlier-prone"
    elif kurt_val <= -KURTOSIS_NOTABLE:
        base["kurtosis_label"] = "light-tailed (platykurtic)"
    else:
        base["kurtosis_label"] = "normal-like tails (mesokurtic)"

    return base


def _balance_for_series(series: pd.Series) -> dict:
    non_null = series.dropna()
    n = int(len(non_null))
    base = {
        "name": str(series.name),
        "count": n,
        "unique_count": 0,
        "entropy": None,
        "normalized_entropy": None,
        "most_common_value": None,
        "most_common_count": 0,
        "least_common_value": None,
        "least_common_count": 0,
        "imbalance_ratio": None,
        "balance_label": "no data",
    }
    if n == 0:
        return base

    value_counts = non_null.value_counts()
    unique_count = int(value_counts.shape[0])
    probs = (value_counts / n).to_numpy(dtype=float)
    entropy = float(-(probs * np.log2(probs)).sum()) if unique_count > 1 else 0.0
    normalized_entropy = round(entropy / math.log2(unique_count), 4) if unique_count > 1 else 0.0
    most_common_count = int(value_counts.iloc[0])
    least_common_count = int(value_counts.iloc[-1])

    base.update(
        {
            "unique_count": unique_count,
            "entropy": round(entropy, 4),
            "normalized_entropy": normalized_entropy,
            "most_common_value": str(value_counts.index[0]),
            "most_common_count": most_common_count,
            "least_common_value": str(value_counts.index[-1]),
            "least_common_count": least_common_count,
            "imbalance_ratio": round(most_common_count / least_common_count, 2) if least_common_count else None,
        }
    )

    if unique_count <= 1:
        base["balance_label"] = "constant (single category)"
    elif normalized_entropy >= ENTROPY_WELL_BALANCED:
        base["balance_label"] = "well balanced"
    elif normalized_entropy >= ENTROPY_MODERATE:
        base["balance_label"] = "moderately balanced"
    else:
        base["balance_label"] = "imbalanced"

    return base


def _distributions_for_df(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ToolExecutionError(f"Unknown column(s): {', '.join(missing)}")
        target_cols = columns
    else:
        target_cols = list(df.columns)

    numeric_results = []
    categorical_results = []
    skipped = []

    for col in target_cols:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            skipped.append({"name": col, "reason": "datetime column — not applicable to skew/kurtosis or category balance"})
            continue
        if pd.api.types.is_bool_dtype(series):
            categorical_results.append(_balance_for_series(series))
            continue
        if pd.api.types.is_numeric_dtype(series):
            numeric_results.append(_skew_kurtosis_for_series(series))
        else:
            categorical_results.append(_balance_for_series(series))

    return {"numeric": numeric_results, "categorical": categorical_results, "skipped_columns": skipped}


def analyze_distributions(df: pd.DataFrame, columns: list[str] | None = None, filters: list[dict] | None = None) -> dict:
    """Skewness/kurtosis for numeric columns, balance/entropy for categorical columns.

    See module docstring for the exact thresholds behind each plain-language
    label. `columns` restricts analysis to the given column names (numeric or
    categorical — each is routed to the right computation); omit to analyze
    every non-datetime column.
    """
    working = apply_filters(df, filters) if filters else df
    if working.empty:
        raise ToolExecutionError("No rows match the given filters.")

    result = _distributions_for_df(working, columns)
    return {"row_count": int(len(working)), **result}


def _severity_sort_key(problem: dict) -> tuple:
    return (_SEVERITY_RANK.get(problem["severity"], 3),)


def _build_problems(
    profile: dict,
    cardinality_columns: list[dict],
    distributions: dict,
    outlier_summaries: list[dict],
    correlation_pairs: list[dict],
) -> list[dict]:
    problems: list[dict] = []

    for c in profile["column_info"]:
        pct = c["missing_pct"]
        if pct <= 0:
            continue
        if pct > 20:
            severity = "high"
        elif pct > 5:
            severity = "medium"
        else:
            severity = "low"
        problems.append(
            {
                "severity": severity,
                "category": "missingness",
                "message": f"Column '{c['name']}' is missing {pct}% of values ({c['missing_count']} rows) — consider imputation or excluding it from analysis.",
            }
        )

    if profile["duplicate_rows"] > 0:
        dup_pct = round(profile["duplicate_rows"] / profile["rows"] * 100, 2) if profile["rows"] else 0.0
        severity = "high" if dup_pct > 5 else "medium"
        problems.append(
            {
                "severity": severity,
                "category": "duplicates",
                "message": f"{profile['duplicate_rows']} duplicate rows found ({dup_pct}% of the dataset) — consider de-duplicating before aggregate analysis.",
            }
        )

    profiler_bool_cols = set(profile.get("boolean_columns", []))
    profiler_date_cols = set(profile.get("date_columns", []))
    for c in cardinality_columns:
        name, cls = c["name"], c["classification"]
        if cls in {"high_cardinality", "unique_id_like"} and name in profiler_date_cols:
            # Date columns are naturally high-cardinality/near-unique; that's
            # expected, not a sign of a mis-typed identifier column.
            continue
        if cls == "constant":
            problems.append(
                {
                    "severity": "high",
                    "category": "cardinality",
                    "message": f"Column '{name}' is constant (only one value) — provides no analytical signal; consider dropping it.",
                }
            )
        elif cls == "near_constant":
            problems.append(
                {
                    "severity": "medium",
                    "category": "cardinality",
                    "message": f"Column '{name}' is {c['top_value_pct']}% one value ('{c['top_value']}') and may not be useful for analysis.",
                }
            )
        elif cls == "unique_id_like":
            problems.append(
                {
                    "severity": "low",
                    "category": "cardinality",
                    "message": f"Column '{name}' looks like an identifier ({c['unique_ratio'] * 100:.1f}% unique) — exclude it from statistics/aggregation, use it only for joins/lookups.",
                }
            )
        elif cls == "high_cardinality":
            problems.append(
                {
                    "severity": "low",
                    "category": "cardinality",
                    "message": f"Column '{name}' has high cardinality ({c['unique_count']} distinct values) — verify it isn't an identifier before grouping by it.",
                }
            )
        elif cls == "boolean_like" and name not in profiler_bool_cols:
            problems.append(
                {
                    "severity": "low",
                    "category": "cardinality",
                    "message": f"Column '{name}' looks boolean-like (values '{c['least_value']}'/'{c['top_value']}') but is stored as {c['dtype']} — consider converting to a real boolean.",
                }
            )

    for item in distributions.get("numeric", []):
        skewness = item.get("skewness")
        if skewness is not None and abs(skewness) >= SKEW_HIGH:
            problems.append(
                {
                    "severity": "medium",
                    "category": "distribution",
                    "message": f"Column '{item['name']}' is {item['skew_label']} (skewness={skewness}) — consider a log transform or median-based statistics instead of the mean.",
                }
            )

    for item in distributions.get("categorical", []):
        if item.get("balance_label") == "imbalanced" and item.get("unique_count", 0) > 1:
            problems.append(
                {
                    "severity": "low",
                    "category": "distribution",
                    "message": f"Column '{item['name']}' categories are imbalanced — '{item['most_common_value']}' accounts for {item['most_common_count']} rows vs. '{item['least_common_value']}' at {item['least_common_count']} (ratio {item['imbalance_ratio']}x).",
                }
            )

    for summary in outlier_summaries:
        pct = summary["anomaly_pct"]
        if pct > 10:
            severity = "high"
        elif pct > 2:
            severity = "medium"
        else:
            continue
        problems.append(
            {
                "severity": severity,
                "category": "outliers",
                "message": f"Column '{summary['column']}' has {pct}% of values flagged as outliers (IQR method, {summary['anomaly_count']} rows) — investigate before relying on mean/std-based metrics.",
            }
        )

    for pair in correlation_pairs:
        r = pair["correlation"]
        if abs(r) >= 0.95:
            severity = "high"
        elif abs(r) >= 0.8:
            severity = "medium"
        else:
            continue
        problems.append(
            {
                "severity": severity,
                "category": "relationships",
                "message": f"Columns '{pair['column_a']}' and '{pair['column_b']}' are highly correlated (r={r}) — check for redundancy or multicollinearity.",
            }
        )

    problems.sort(key=_severity_sort_key)
    return problems


def automated_eda(df: pd.DataFrame, filters: list[dict] | None = None) -> dict:
    """Full automatic exploratory pass: schema -> types -> missingness ->
    duplicates -> cardinality -> distributions -> outliers -> relationships
    -> a prioritized list of plain-language "potential problems".

    Composes `profile_dataset`, `detect_anomalies`, and `correlation_analysis`
    rather than recomputing their logic, and adds cardinality classification
    plus distribution-shape analysis on top. Output is bounded: outlier
    summaries never include row-level data, and per-column detail sections
    are capped at `_MAX_DETAIL_COLUMNS` columns to avoid oversized payloads
    on very wide datasets.
    """
    working = apply_filters(df, filters) if filters else df
    if working.empty:
        raise ToolExecutionError("No rows match the given filters.")

    profile = profile_dataset(working)
    numeric_cols = profile["numeric_columns"]
    categorical_cols = profile["categorical_columns"]

    detail_columns = list(working.columns)[:_MAX_DETAIL_COLUMNS]
    columns_truncated = len(working.columns) > _MAX_DETAIL_COLUMNS

    cardinality_columns = _cardinality_for_df(working[detail_columns])
    cardinality_summary: dict[str, int] = {}
    for c in cardinality_columns:
        cardinality_summary[c["classification"]] = cardinality_summary.get(c["classification"], 0) + 1

    distributions = _distributions_for_df(working, detail_columns)

    outlier_summaries = []
    for col in numeric_cols[:_MAX_OUTLIER_COLUMNS]:
        try:
            result = detect_anomalies(working, col, method="iqr")
        except ToolExecutionError:
            continue
        outlier_summaries.append(
            {
                "column": col,
                "method": result["method"],
                "bounds": result["bounds"],
                "anomaly_count": result["anomaly_count"],
                "anomaly_pct": result["anomaly_pct"],
            }
        )

    correlation_pairs: list[dict] = []
    if len(numeric_cols) >= 2:
        try:
            corr_result = correlation_analysis(working, numeric_cols)
            correlation_pairs = corr_result["strongest_pairs"]
        except ToolExecutionError:
            correlation_pairs = []

    problems = _build_problems(profile, cardinality_columns, distributions, outlier_summaries, correlation_pairs)
    problems_truncated = len(problems) > _MAX_PROBLEMS
    problems = problems[:_MAX_PROBLEMS]

    return {
        "row_count": profile["rows"],
        "column_count": profile["columns"],
        "schema": {
            "column_info": profile["column_info"],
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "date_columns": profile["date_columns"],
            "boolean_columns": profile["boolean_columns"],
        },
        "missingness": {
            "missing_total": profile["missing_total"],
            "columns_with_missing": [c["name"] for c in profile["column_info"] if c["missing_count"] > 0],
        },
        "duplicates": {
            "duplicate_rows": profile["duplicate_rows"],
            "duplicate_pct": round(profile["duplicate_rows"] / profile["rows"] * 100, 2) if profile["rows"] else 0.0,
        },
        "cardinality": {
            "columns": cardinality_columns,
            "summary": cardinality_summary,
            "columns_truncated": columns_truncated,
        },
        "distributions": distributions,
        "outliers": outlier_summaries,
        "relationships": {"method": "pearson", "strongest_pairs": correlation_pairs},
        "potential_problems": problems,
        "problems_truncated": problems_truncated,
    }
