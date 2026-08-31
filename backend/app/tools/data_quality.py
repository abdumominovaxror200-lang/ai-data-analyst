"""Data-quality / "can I trust this data" checks.

This module is deliberately narrower than `app.tools.eda`: `eda.py` already
owns the broad exploratory pass (cardinality classification, distribution
shape/skewness/kurtosis, a basic duplicate-row *count* folded into its
`potential_problems` list). This module does not recompute any of that — it
composes `profile_dataset` (and, inside `data_quality_report`, this module's
own `duplicate_analysis`) and adds exactly three things `eda.py` does not
cover:

1. **Detailed duplicate-row analysis** (`duplicate_analysis`) — not just a
   count, but which specific rows form which duplicate groups, in both
   full-row and subset-column modes. Subset-column mode matters because the
   most common real-world "this data is untrustworthy" case is NOT a
   byte-for-byte repeated row — it's the same natural key (e.g.
   `customer_id` + `date`) appearing more than once with different values,
   which usually means an upstream double-charge, a duplicate import, or a
   join fan-out. Full-row duplicate detection misses this entirely, because
   the differing value column(s) make every row unique.

2. **Mixed-type detection within object-dtype columns** — a column that
   *should* be numeric but has some values stored as non-numeric strings
   (`"1,234"`, `"N/A"`, stray text) mixed in with real numbers. This is a
   distinct failure mode from `eda.py`'s cardinality/entropy angle: a mixed
   column can have perfectly reasonable cardinality and still silently
   coerce to `NaN` (or worse, get treated as a categorical column) the
   moment it's used in an aggregate function.

3. **Missingness co-occurrence** — whether two columns' missingness is
   correlated (rows missing one column are disproportionately likely to
   also be missing the other), which usually signals a shared upstream
   cause (same failed join, same optional form section, same broken
   ingestion step) rather than independent, incidental gaps.

`data_quality_report` synthesizes all of the above (plus `profile_dataset`'s
missingness/dtype baseline and this module's own `duplicate_analysis`) into
one severity-ranked `quality_issues` list and a single deterministic
`quality_score`. See `data_quality_report`'s docstring for the exact scoring
formula.

Payload-size discipline matches `app.tools.insights` /
`app.tools.eda.automated_eda`: duplicate-row output is capped by
`max_examples` (never the full duplicate set), mixed-type example values are
capped at `_MIXED_TYPE_EXAMPLE_VALUES`, and `data_quality_report`'s issue
list is capped at `_MAX_ISSUES`.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.tools.errors import ToolExecutionError
from app.tools.filtering import apply_filters
from app.tools.profiler import profile_dataset
from app.tools.serialization import dataframe_to_records

# --- duplicate_analysis tunables ---
_DEFAULT_MAX_EXAMPLES = 20

# --- mixed-type check tunables ---
# Only consider an object column "supposed to be numeric" if at least this
# fraction of its non-null values cleanly coerce via pd.to_numeric — below
# this it's more likely a genuine free-text/categorical column that
# incidentally contains a few numeric-looking tokens, not a mistyped
# numeric column.
MIXED_TYPE_MIN_SUCCESS_FRACTION = 0.5
# Ignore columns with too few non-null values to draw a meaningful
# conclusion from (avoids noise on near-empty columns).
_MIXED_TYPE_MIN_NON_NULL = 5
_MIXED_TYPE_EXAMPLE_VALUES = 5
MIXED_TYPE_HIGH_FRACTION = 0.2
MIXED_TYPE_MEDIUM_FRACTION = 0.05

# --- missingness co-occurrence tunables ---
# A column's missing rate must clear this bar before it's even considered
# for co-occurrence analysis — avoids flagging noise between two columns
# that are each missing only 1-2 rows, where any correlation reading is
# essentially random.
MISSINGNESS_COOCCURRENCE_MIN_RATE = 0.05
# Pearson correlation between two 0/1 missingness indicators must clear
# this bar to be reported at all.
MISSINGNESS_COOCCURRENCE_CORR_THRESHOLD = 0.5
MISSINGNESS_COOCCURRENCE_HIGH_CORR = 0.8

# --- data_quality_report tunables ---
MISSING_HIGH_PCT = 20.0
MISSING_MEDIUM_PCT = 5.0
DUPLICATE_HIGH_PCT = 5.0
_REPORT_DUPLICATE_EXAMPLES = 5
_MAX_ISSUES = 30

_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}
# The exact, documented weights behind `quality_score` (see
# `data_quality_report`'s docstring for the full formula).
_SEVERITY_SCORE_WEIGHTS = {"high": 15, "medium": 7, "low": 2}


def _row_index_to_json_safe(index_value: object) -> object:
    if isinstance(index_value, (bool, np.bool_)):
        return bool(index_value)
    if isinstance(index_value, (int, np.integer)):
        return int(index_value)
    return str(index_value)


def duplicate_analysis(
    df: pd.DataFrame,
    subset_columns: list[str] | None = None,
    filters: list[dict] | None = None,
    max_examples: int = _DEFAULT_MAX_EXAMPLES,
) -> dict:
    """Detect duplicate rows, in two modes.

    - ``subset_columns=None`` (default): **full-row** duplicates — every
      column must match. Equivalent to ``df.duplicated(keep=False)``.
    - ``subset_columns=[...]``: duplicates considering only the given
      columns. This is the more useful real-world case: e.g.
      ``["customer_id", "date"]`` finds a customer billed twice on the same
      day even when the amount differs between the two charges — which is
      NOT a full-row duplicate (the differing `amount` value makes every row
      technically unique) but is exactly the kind of "same natural key,
      recorded more than once" pattern that indicates a double-charge, a
      duplicate import, or a join fan-out upstream. Full-row-only detection
      would silently miss this.

    Reports (all JSON-safe, via `dataframe_to_records` for the sampled
    group key values):
    - ``total_rows``: row count after `filters`.
    - ``duplicate_row_count``: rows that are part of ANY duplicate group —
      i.e. ``df.duplicated(subset=subset_columns, keep=False).sum()``. This
      is every row involved, not just "extra" copies beyond the first.
    - ``duplicate_pct``: `duplicate_row_count` / `total_rows` * 100.
    - ``duplicate_group_count``: number of DISTINCT duplicate groups (not
      rows) — e.g. 3 rows that are all copies of each other is 1 group.
    - ``examples``: a bounded (`max_examples`, default 20) sample of the
      actual duplicate groups, largest group first, each with the shared
      key values, how many times the group repeats (``occurrences``), and
      the original row indices (``row_indices``) so a caller can look the
      rows up. ``examples_truncated`` is True when more groups exist than
      were sampled.

    Raises `ToolExecutionError` if: `max_examples` is negative; the
    dataframe is empty after `filters`; `subset_columns` is an empty list;
    or any name in `subset_columns` does not exist in `df`.
    """
    if max_examples < 0:
        raise ToolExecutionError("max_examples must be >= 0.")

    working = apply_filters(df, filters) if filters else df
    if working.empty:
        raise ToolExecutionError("No rows match the given filters.")

    if subset_columns is not None:
        if len(subset_columns) == 0:
            raise ToolExecutionError("subset_columns, if given, must contain at least one column name.")
        missing = [c for c in subset_columns if c not in working.columns]
        if missing:
            raise ToolExecutionError(f"Unknown column(s): {', '.join(missing)}")
        dup_cols = list(subset_columns)
        mode = "subset_columns"
    else:
        dup_cols = list(working.columns)
        mode = "full_row"

    total_rows = int(len(working))
    dup_mask = working.duplicated(subset=dup_cols, keep=False)
    duplicate_row_count = int(dup_mask.sum())
    duplicate_pct = round(duplicate_row_count / total_rows * 100, 2) if total_rows else 0.0

    base_result = {
        "mode": mode,
        "subset_columns": dup_cols if mode == "subset_columns" else None,
        "total_rows": total_rows,
        "duplicate_row_count": duplicate_row_count,
        "duplicate_pct": duplicate_pct,
        "max_examples": max_examples,
    }

    if duplicate_row_count == 0:
        return {
            **base_result,
            "duplicate_group_count": 0,
            "examples": [],
            "examples_truncated": False,
        }

    dup_df = working[dup_mask]
    # dropna=False so an entirely-missing key still forms a valid group
    # (matches df.duplicated()'s treatment of NaN as equal to NaN for
    # duplicate purposes); sort=False preserves first-appearance order so
    # the subsequent stable sort-by-size ties break deterministically on
    # original row order.
    # pandas can raise ``Categorical categories cannot be null`` when grouping a
    # single all-null duplicate key. Build collision-free in-memory tuple keys
    # with a private sentinel, while retaining the original values for output.
    null_sentinel = object()
    normalized = dup_df[dup_cols].astype(object).where(dup_df[dup_cols].notna(), null_sentinel)
    group_keys = normalized.apply(lambda row: tuple(row.tolist()), axis=1)
    grouped = list(dup_df.groupby(group_keys, sort=False))
    grouped.sort(key=lambda kv: -len(kv[1]))  # largest group first; stable for ties

    duplicate_group_count = len(grouped)
    sample = grouped[:max_examples] if max_examples > 0 else []

    key_records: list[dict] = []
    occurrences: list[int] = []
    row_indices: list[list[object]] = []
    for _key, sub_df in sample:
        key_dict = sub_df.iloc[0][dup_cols].to_dict()
        key_records.append(key_dict)
        occurrences.append(int(len(sub_df)))
        row_indices.append([_row_index_to_json_safe(i) for i in sub_df.index.tolist()])

    safe_keys = dataframe_to_records(pd.DataFrame(key_records, columns=dup_cols)) if key_records else []

    examples = [
        {"key_values": safe_keys[i], "occurrences": occurrences[i], "row_indices": row_indices[i]}
        for i in range(len(sample))
    ]

    return {
        **base_result,
        "duplicate_group_count": duplicate_group_count,
        "examples": examples,
        "examples_truncated": duplicate_group_count > max_examples,
    }


def _mixed_type_check(df: pd.DataFrame) -> list[dict]:
    """Flag object-dtype columns that look numeric but contain non-numeric
    string values mixed in (e.g. "1,234", "N/A" alongside real numbers).

    A column is only flagged when a majority (>= `MIXED_TYPE_MIN_SUCCESS_FRACTION`)
    of its non-null values coerce cleanly via `pd.to_numeric` AND at least
    one value does not — distinct from `eda.py`'s cardinality/entropy
    checks, which never attempt numeric coercion at all.
    """
    results: list[dict] = []
    for col in df.columns:
        series = df[col]
        if not pd.api.types.is_object_dtype(series):
            continue
        non_null = series.dropna()
        n = int(len(non_null))
        if n < _MIXED_TYPE_MIN_NON_NULL:
            continue

        coerced = pd.to_numeric(non_null, errors="coerce")
        numeric_mask = coerced.notna()
        succeeded = int(numeric_mask.sum())
        failed = n - succeeded
        frac_success = succeeded / n
        frac_fail = failed / n

        if failed == 0 or frac_success < MIXED_TYPE_MIN_SUCCESS_FRACTION:
            continue

        example_values = [str(v) for v in non_null[~numeric_mask].unique().tolist()[:_MIXED_TYPE_EXAMPLE_VALUES]]

        if frac_fail > MIXED_TYPE_HIGH_FRACTION:
            severity = "high"
        elif frac_fail > MIXED_TYPE_MEDIUM_FRACTION:
            severity = "medium"
        else:
            severity = "low"

        examples_str = ", ".join(f"'{v}'" for v in example_values[:3]) or "n/a"
        results.append(
            {
                "column": str(col),
                "non_null_count": n,
                "numeric_coercible_count": succeeded,
                "non_numeric_count": failed,
                "non_numeric_fraction": round(frac_fail, 4),
                "example_non_numeric_values": example_values,
                "severity": severity,
                "message": (
                    f"Column '{col}' looks numeric ({round(frac_success * 100, 1)}% of values coerce cleanly) "
                    f"but {failed} value(s) ({round(frac_fail * 100, 1)}%) are stored as non-numeric text "
                    f"(e.g. {examples_str}) — likely formatting artifacts (thousands separators, 'N/A', stray "
                    f"text) mixed into what should be a numeric column."
                ),
            }
        )

    results.sort(key=lambda r: -r["non_numeric_fraction"])
    return results


def _missingness_cooccurrence(df: pd.DataFrame) -> list[dict]:
    """For column pairs that both have a non-trivial missing rate
    (>= `MISSINGNESS_COOCCURRENCE_MIN_RATE`), compute the Pearson
    correlation between their 0/1 missingness indicators and report pairs
    whose correlation clears `MISSINGNESS_COOCCURRENCE_CORR_THRESHOLD`.

    Only non-trivially-missing columns are considered at all, specifically
    to avoid noise: with a near-zero missing rate, any correlation reading
    is essentially random and not actionable.
    """
    n = len(df)
    if n == 0:
        return []

    candidates: list[str] = []
    rates: dict[str, float] = {}
    indicators: dict[str, pd.Series] = {}
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        rate = missing_count / n
        if missing_count > 0 and rate >= MISSINGNESS_COOCCURRENCE_MIN_RATE:
            candidates.append(str(col))
            rates[str(col)] = rate
            indicators[str(col)] = df[col].isna().astype(int)

    flagged: list[dict] = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            corr = indicators[a].corr(indicators[b])
            if corr is None or (isinstance(corr, float) and math.isnan(corr)):
                continue
            corr = float(corr)
            if abs(corr) < MISSINGNESS_COOCCURRENCE_CORR_THRESHOLD:
                continue
            severity = "high" if abs(corr) >= MISSINGNESS_COOCCURRENCE_HIGH_CORR else "medium"
            flagged.append(
                {
                    "column_a": a,
                    "column_b": b,
                    "missing_rate_a_pct": round(rates[a] * 100, 2),
                    "missing_rate_b_pct": round(rates[b] * 100, 2),
                    "correlation": round(corr, 4),
                    "severity": severity,
                    "message": (
                        f"Missingness in '{a}' and '{b}' is correlated (r={round(corr, 2)}) — rows missing one "
                        f"are disproportionately likely to also be missing the other. Investigate a shared "
                        f"upstream cause (same source/process/join) rather than treating the two gaps as "
                        f"independent."
                    ),
                }
            )

    flagged.sort(key=lambda x: -abs(x["correlation"]))
    return flagged


def _verdict_for_score(score: int) -> str:
    if score >= 90:
        return "high trust - no material data-quality concerns"
    if score >= 70:
        return "moderate trust - review flagged issues before relying on aggregate results"
    if score >= 50:
        return "low trust - significant data-quality issues should be resolved first"
    return "very low trust - do not rely on aggregate results without remediation"


def data_quality_report(df: pd.DataFrame, filters: list[dict] | None = None) -> dict:
    """Composite "can I trust this data" report, distinct from
    `eda.automated_eda`'s broader exploratory summary.

    Composes:
    - `profile_dataset` for the missingness/dtype baseline (not recomputed).
    - This module's own `duplicate_analysis` (full-row mode) for detailed
      duplicate-row detection (not just `profile_dataset`'s basic count).
    - `_mixed_type_check` (new): object columns that look numeric but
      contain non-numeric string values mixed in.
    - `_missingness_cooccurrence` (new): column pairs whose missingness is
      correlated.

    All four are synthesized into one severity-ranked `quality_issues` list
    (same high/medium/low convention as `eda.py`'s `potential_problems`)
    plus a single ``quality_score``.

    **quality_score formula (exact, deterministic, documented so it is
    reproducible and testable — not a vibe):**

        score = max(0, 100 - sum(WEIGHT[issue.severity] for issue in all_issues))

        WEIGHT = {"high": 15, "medium": 7, "low": 2}

    `all_issues` is the FULL synthesized issue set (missingness columns +
    duplicate-row summary + mixed-type columns + missingness-cooccurrence
    pairs) — the score is computed before the returned `quality_issues`
    list is capped at `_MAX_ISSUES` for payload size, so the score always
    reflects every issue found even if the printed list is truncated.
    A dataset with zero issues scores 100. Per-category severity
    assignment:
    - missingness: per column with missing_pct > 0 — high if > 20%,
      medium if > 5%, else low (`MISSING_HIGH_PCT`/`MISSING_MEDIUM_PCT`).
    - duplicates: one issue if any full-row duplicates exist — high if
      duplicate_pct > 5% (`DUPLICATE_HIGH_PCT`), else medium.
    - mixed_type: one issue per flagged column — high if non-numeric
      fraction > 20%, medium if > 5%, else low
      (`MIXED_TYPE_HIGH_FRACTION`/`MIXED_TYPE_MEDIUM_FRACTION`).
    - missingness_cooccurrence: one issue per flagged pair — high if
      |correlation| >= 0.8 (`MISSINGNESS_COOCCURRENCE_HIGH_CORR`), else
      medium (pairs are only flagged at all once |correlation| >= 0.5).

    Bounded payload: duplicate examples are capped inside
    `duplicate_analysis` (`_REPORT_DUPLICATE_EXAMPLES`, tighter than that
    function's own default since this is a composite report), and
    `quality_issues` is capped at `_MAX_ISSUES`.
    """
    working = apply_filters(df, filters) if filters else df
    if working.empty:
        raise ToolExecutionError("No rows match the given filters.")

    profile = profile_dataset(working)
    dup_report = duplicate_analysis(working, max_examples=_REPORT_DUPLICATE_EXAMPLES)
    mixed_type_columns = _mixed_type_check(working)
    missingness_cooccurrence = _missingness_cooccurrence(working)

    issues: list[dict] = []

    for c in profile["column_info"]:
        pct = c["missing_pct"]
        if pct <= 0:
            continue
        if pct > MISSING_HIGH_PCT:
            severity = "high"
        elif pct > MISSING_MEDIUM_PCT:
            severity = "medium"
        else:
            severity = "low"
        issues.append(
            {
                "severity": severity,
                "category": "missingness",
                "message": f"Column '{c['name']}' is missing {pct}% of values ({c['missing_count']} rows).",
            }
        )

    if dup_report["duplicate_row_count"] > 0:
        dup_pct = dup_report["duplicate_pct"]
        severity = "high" if dup_pct > DUPLICATE_HIGH_PCT else "medium"
        issues.append(
            {
                "severity": severity,
                "category": "duplicates",
                "message": (
                    f"{dup_report['duplicate_row_count']} rows ({dup_pct}%) are part of "
                    f"{dup_report['duplicate_group_count']} exact full-row duplicate group(s) — consider "
                    f"de-duplicating before aggregate analysis, or re-run duplicate_analysis with "
                    f"subset_columns to check for repeated natural keys with differing values."
                ),
            }
        )

    for item in mixed_type_columns:
        issues.append({"severity": item["severity"], "category": "mixed_type", "message": item["message"]})

    for item in missingness_cooccurrence:
        issues.append(
            {"severity": item["severity"], "category": "missingness_cooccurrence", "message": item["message"]}
        )

    issues.sort(key=lambda i: _SEVERITY_RANK.get(i["severity"], 3))

    total_penalty = sum(_SEVERITY_SCORE_WEIGHTS.get(i["severity"], 0) for i in issues)
    quality_score = max(0, 100 - total_penalty)

    issues_truncated = len(issues) > _MAX_ISSUES
    bounded_issues = issues[:_MAX_ISSUES]

    return {
        "row_count": profile["rows"],
        "column_count": profile["columns"],
        "quality_score": quality_score,
        "quality_verdict": _verdict_for_score(quality_score),
        "quality_issues": bounded_issues,
        "issues_truncated": issues_truncated,
        "missingness": {
            "missing_total": profile["missing_total"],
            "columns_with_missing": [c["name"] for c in profile["column_info"] if c["missing_count"] > 0],
        },
        "duplicates": dup_report,
        "mixed_type_columns": mixed_type_columns,
        "missingness_cooccurrence": missingness_cooccurrence,
    }
