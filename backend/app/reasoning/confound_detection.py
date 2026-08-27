"""Deterministic confounding-variable detection (Phase 5; upgraded to "Confound
Engine 2.0" for the v2 reliability mission).

Motivated by a REAL failure confirmed live against the configured provider, not a
scripted benchmark artifact (see `.agent/hard_realworld_real_llm_spotcheck.md`): asked
"North region has a $55 higher average basket size than South -- is North just a
better-performing region?" against a dataset where North is 90% large-format stores
and South is 90% small-format stores (a classic Simpson's-paradox-style confound), the
real model ran a plain `t_test` on the raw regional comparison and never checked
whether another variable explained the gap. The `region_size_confound` fixture was
built specifically to test this pattern; it had only ever been exercised through a
scripted MockProvider response until this live run confirmed the real model misses it
too, with nothing in the architecture catching it.

Per the standing instruction to fix the reasoning architecture rather than merely the
prompt: this module does not rely on the model remembering to check for confounds. It
runs unconditionally, deterministically, after any group-comparison tool call
(`t_test`, `group_and_aggregate`, `anova_test`), using the real dataset the comparison
ran against -- zero new LLM calls, following the same "read already-gathered evidence
plus the dataset itself" discipline as `verifier.py`'s other deterministic checks.

## v2 upgrade: three independent signal sources, not one

1. **Categorical distributional imbalance** (the original mechanism): a candidate
   categorical column whose distribution differs sharply between the compared groups.
2. **Numeric confound** (new): a candidate NUMERIC column whose mean differs sharply
   (standardized, Cohen's-d style) between the compared groups -- e.g. "customer
   tenure" confounding a region comparison. The original module was explicitly scoped
   to categorical-only; this closes that gap with the same standardized-effect-size
   discipline `app/tools/hypothesis.py::effect_size` already uses, not a new metric.
3. **Missingness imbalance** (new): a candidate column's NULL RATE differs sharply
   between the compared groups -- a real, distinct data-quality-flavored confound
   signal (e.g. a tracking gap that disproportionately affects one group) separate
   from either of the above.

## v2 upgrade: an actual stratified-effect check, not just a distributional proxy

The original module only asked "does the candidate's own distribution differ between
groups?" -- necessary but NOT sufficient for a real Simpson's-paradox reversal (the
comparison could still point the same direction within every stratum, just with
different weights). `_stratified_effect_check` now directly computes the compared
metric's mean for each group WITHIN each stratum of the candidate column, and checks
whether the direction (a real reversal) or magnitude (a real, substantial shrink)
changes relative to the unstratified (aggregate) comparison -- the actual analytical
question a human analyst controlling for a variable would ask, not a proxy for it.
When this direct check can run (strata have enough rows in both groups -- often NOT
the case for the most extreme confounds, where one group barely appears in one
stratum at all, e.g. this module's own motivating 18-vs-2 fixture), its result
overrides the proportion-gap-based severity: a genuine within-stratum reversal is
stronger evidence than a distributional gap alone (escalates to `blocks_conclusion`
regardless of the gap's own severity), and a confirmed ABSENT reversal (every checked
stratum agrees with the aggregate direction) downgrades severity, since direct
evidence the conclusion survives stratification is more informative than the raw gap.
When it cannot run (too little per-stratum-per-group data), the original
proportion-gap-based severity logic is the fallback -- unchanged from before this
upgrade.

## v2 upgrade: an explicit classification, not just accept/reject

Every candidate column search this module runs now resolves to one of:
`"confounder"` (a real signal, reported), `"nested"` (skipped -- see
`_is_nested_not_confounded`'s docstring), `"identifier"` (skipped -- near-unique,
would never plausibly be a real confound, just a row identifier), or `"irrelevant"`
(skipped -- distribution/mean/missingness do not differ enough to matter). A
`"possible_mediator_or_confound"` note is appended to the reported text (never a
silent reclassification) when a stratified reversal check succeeds and finds NO
reversal but the aggregate distributional/mean gap is still large -- this shape (the
candidate correlates with both the group and the outcome, but doesn't change the
within-stratum conclusion) is equally consistent with a true confound OR a mediator
(X -> candidate -> Y) sitting on the causal path, and cross-sectional data alone
cannot distinguish the two without temporal ordering or domain knowledge neither of
which this module has access to. This module never claims causal certainty either
way -- see the module-level rule this whole reasoning layer follows
(`causation_guard.py`).

**Still deliberately out of scope**: multi-way (2+ simultaneous candidate) stratification,
and any attempt to infer causal direction from column names or metadata.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from app.reasoning.contracts import Evidence, Limitation

_MIN_GROUP_SIZE = 5  # each compared group needs at least this many rows to trust a proportion
_MIN_CATEGORY_LEVELS = 2
_MAX_CATEGORY_LEVELS = 20  # avoid scanning near-unique/ID-like columns as "categorical"
_CONFOUND_PROPORTION_GAP_THRESHOLD = 0.40  # a 40-point swing in a category's share is not noise
_MIN_SHARED_VALUE_FRACTION = 0.3  # see _is_nested_not_confounded's docstring
# A gap this large (e.g. a 90%-vs-10% split, like the real region_size_confound
# fixture) is severe enough that a comparison along `group_col` is barely
# distinguishable from a comparison along `other_col` -- treated as `blocks_conclusion`
# severity (no confidence claim is justified at all, see recommendation_grounding.py's
# global blocks_conclusion override) rather than merely `reduces_confidence`. A more
# moderate gap between this and `_CONFOUND_PROPORTION_GAP_THRESHOLD` is still real and
# worth flagging, but not severe enough to say no conclusion is possible at all.
_SEVERE_CONFOUND_PROPORTION_GAP_THRESHOLD = 0.70
# A value must make up at least this FRACTION of a group's rows to count as "really
# present" there (not a stray/noise row) -- a fraction, not an absolute count, because
# an absolute-count floor breaks down at the extremes: it correctly separates an
# 18-vs-2-split confound (region_size_confound) from a truly nested relationship
# (product/category, 0% overlap) only by accident of matching group sizes. A more
# extreme but still genuine confound (e.g. an 18/1/1 three-way split) has a minority
# presence of just 1 row per group, which falls below any count-based floor above 1 --
# found as a real bug via direct testing (see test_confound_detection.py's 3-group
# regression test) before this fraction-based version replaced it.
_MIN_PRESENCE_FRACTION = 0.03
# A candidate column where more than this fraction of rows are distinct values is
# treated as an identifier (a customer/transaction/order ID, a free-text field, a
# near-unique code) rather than a real confound candidate -- these can technically
# pass the cardinality/gap checks (e.g. a 15-distinct-value ID column in a 20-row
# comparison) but are never a plausible confounding VARIABLE, just row-level noise.
_IDENTIFIER_UNIQUENESS_RATIO = 0.9
# Cohen's-d-style standardized mean difference threshold for a NUMERIC candidate to
# count as a real confound signal -- reuses the exact "medium/large" magnitude
# boundary app/tools/hypothesis.py::effect_size already uses (_MEANINGFUL_MAGNITUDES),
# so this module and that tool never disagree about what counts as a meaningful
# effect size.
_NUMERIC_CONFOUND_EFFECT_SIZE_THRESHOLD = 0.5
# A 20-point gap in NULL rate between the compared groups is a real, worth-flagging
# data-quality-flavored confound signal (e.g. a tracking gap disproportionately
# affecting one group) -- same order of magnitude as the categorical proportion-gap
# threshold above, chosen for the same "not noise" reason.
_MISSINGNESS_GAP_THRESHOLD = 0.20
# Per-stratum-per-group minimum to trust a stratified mean comparison -- same floor as
# _MIN_GROUP_SIZE, applied WITHIN each stratum rather than to the whole group.
_STRATIFIED_MIN_GROUP_SIZE = 5

ConfoundClassification = Literal["confounder", "nested", "identifier", "irrelevant"]


def _extract_comparison(result_summary: dict) -> tuple[str | None, list]:
    """Reads which column and which specific values were compared directly off a
    tool's own real result shape -- never re-derived or guessed. Three distinct real
    shapes are recognized (confirmed against app/tools/hypothesis.py and
    app/tools/aggregation.py, not guessed):

    - `t_test`: "group_column" + "group_a"/"group_b" dicts each with a "label".
    - `group_and_aggregate`: "group_by" + a "groups" LIST of {"group": ..., ...} dicts.
    - `anova_test`: "group_column" + a "groups" DICT keyed directly by group name
      (found missing entirely in this function's first version -- a 3+-group ANOVA
      comparison got zero confound-checking coverage until this branch was added)."""
    group_col = result_summary.get("group_column")
    if group_col:
        values = []
        for key in ("group_a", "group_b"):
            g = result_summary.get(key)
            if isinstance(g, dict) and "label" in g:
                values.append(g["label"])
        if len(values) >= 2:
            return group_col, values

        groups_dict = result_summary.get("groups")
        if isinstance(groups_dict, dict) and len(groups_dict) >= 2:
            return group_col, list(groups_dict.keys())

    group_by = result_summary.get("group_by")
    groups_list = result_summary.get("groups")
    if group_by and isinstance(groups_list, list):
        values = [g["group"] for g in groups_list if isinstance(g, dict) and "group" in g]
        if len(values) >= 2:
            return group_by, values

    return None, []


def _is_candidate_categorical(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return False
    n_unique = series.nunique(dropna=True)
    if n_unique < _MIN_CATEGORY_LEVELS or n_unique > _MAX_CATEGORY_LEVELS:
        return False
    return not _is_identifier(series)


def _is_candidate_numeric(series: pd.Series) -> bool:
    """Deliberately does NOT apply `_is_identifier`'s uniqueness-ratio check -- a
    continuous numeric measurement (revenue, tenure, age) is EXPECTED to be
    near-unique across rows (that is just what a continuous distribution looks
    like), unlike a categorical/string identifier where near-uniqueness really
    does mean "this is a row-level code, not a variable". Found as a real bug while
    testing this exact case: a synthetic 30-row tenure_months column (all distinct
    by construction, being drawn from a continuous normal) was being excluded as an
    "identifier" before it ever reached the standardized-mean-gap check."""
    if not pd.api.types.is_numeric_dtype(series):
        return False
    return not pd.api.types.is_bool_dtype(series)


def _is_identifier(series: pd.Series) -> bool:
    """A near-unique CATEGORICAL/string column (a customer/order/transaction ID, a
    free-text field, a row-level code) is never a plausible confounding VARIABLE --
    it can technically survive the cardinality/gap checks by coincidence in a small
    comparison, but flagging it would be noise, not a real analytical signal. Only
    ever applied to the categorical candidate path -- see `_is_candidate_numeric`'s
    docstring for why numeric columns are exempt."""
    n = len(series.dropna())
    if n == 0:
        return False
    return (series.nunique(dropna=True) / n) >= _IDENTIFIER_UNIQUENESS_RATIO


def _is_nested_not_confounded(df: pd.DataFrame, group_col: str, group_values: list, other_col: str) -> bool:
    """Distinguishes a genuine confound (both compared groups contain the SAME set
    of `other_col` categories, just at different rates -- e.g. both regions have
    both store formats) from a nested/hierarchical relationship (each `other_col`
    value belongs almost entirely to ONE group -- e.g. each product belongs to
    exactly one category, so 'product' trivially 'differs' across 'category' groups
    without being an independent confounding variable at all).

    Found as a real false positive while verifying this detector against the
    project's own primary dataset: `product` was flagged as confounding a
    `category` comparison, when category IS product's own grouping -- every one of
    the 10 products belongs to exactly 1 of the 4 categories (confirmed by direct
    computation), the textbook nested-not-confounded case this guard exists for.

    Returns True (skip -- not a real confound) when fewer than
    `_MIN_SHARED_VALUE_FRACTION` of `other_col`'s distinct values present in this
    comparison actually appear (at real presence -- at least `_MIN_PRESENCE_FRACTION`
    of that group's rows, not a stray/noise row) in more than one of the compared
    groups.

    Compares `group_col` as strings throughout (rather than `.isin`/`==` against the
    column's native dtype) -- a tool's own reported group labels (e.g.
    `group_and_aggregate`'s `groups[i]["group"]`) are always plain strings even when
    `group_col` itself is a datetime/numeric column, so a native-dtype comparison
    would silently match nothing (or, for datetime columns specifically, raise a
    pandas `FutureWarning` on `.isin` with mismatched types)."""
    group_col_as_str = df[group_col].astype(str)
    group_values_str = {str(v) for v in group_values}
    subset = df[group_col_as_str.isin(group_values_str)]
    subset_group_str = group_col_as_str[group_col_as_str.isin(group_values_str)]
    presence: dict = {}
    for gv in group_values_str:
        group_rows = subset.loc[subset_group_str == gv, other_col]
        group_size = len(group_rows)
        if group_size == 0:
            continue
        for value, count in group_rows.value_counts().items():
            if (count / group_size) >= _MIN_PRESENCE_FRACTION:
                presence.setdefault(value, set()).add(gv)

    if not presence:
        return True
    shared = sum(1 for groups_seen in presence.values() if len(groups_seen) > 1)
    return (shared / len(presence)) < _MIN_SHARED_VALUE_FRACTION


def _max_proportion_gap(df: pd.DataFrame, group_col: str, group_values: list, other_col: str) -> float | None:
    """String-compares `group_col` for the same reason `_is_nested_not_confounded`
    does -- a tool's own reported group labels are plain strings even when
    `group_col` is a datetime/numeric column."""
    group_col_as_str = df[group_col].astype(str)
    subsets: dict = {}
    for gv in group_values:
        subset = df[group_col_as_str == str(gv)]
        if len(subset) < _MIN_GROUP_SIZE:
            return None
        subsets[gv] = subset[other_col].value_counts(normalize=True)

    other_values: set = set()
    for props in subsets.values():
        other_values.update(props.index)

    max_gap = 0.0
    for ov in other_values:
        proportions = [subsets[gv].get(ov, 0.0) for gv in group_values]
        max_gap = max(max_gap, max(proportions) - min(proportions))
    return max_gap


def _numeric_standardized_gap(df: pd.DataFrame, group_col: str, group_values: list, other_col: str) -> float | None:
    """Cohen's-d-style standardized mean difference of a NUMERIC candidate column
    between the two compared groups -- the same pooled-standard-deviation formula
    `app/tools/hypothesis.py::effect_size` uses, applied here to the CANDIDATE
    confound column rather than the analysis metric. Only defined for exactly 2
    groups (a 3+-group numeric standardized gap needs a different, ANOVA-shaped
    formulation not built here -- an honest scope limit, not an oversight)."""
    if len(group_values) != 2:
        return None
    group_col_as_str = df[group_col].astype(str)
    a_values = df.loc[group_col_as_str == str(group_values[0]), other_col].dropna()
    b_values = df.loc[group_col_as_str == str(group_values[1]), other_col].dropna()
    if len(a_values) < _MIN_GROUP_SIZE or len(b_values) < _MIN_GROUP_SIZE:
        return None

    n1, n2 = len(a_values), len(b_values)
    var1, var2 = a_values.var(ddof=1), b_values.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if not np.isfinite(pooled_std) or pooled_std == 0:
        return None
    return float(abs(a_values.mean() - b_values.mean()) / pooled_std)


def _missingness_gap(df: pd.DataFrame, group_col: str, group_values: list, other_col: str) -> float | None:
    """Max gap in `other_col`'s null rate across the compared groups."""
    group_col_as_str = df[group_col].astype(str)
    rates = []
    for gv in group_values:
        subset = df.loc[group_col_as_str == str(gv), other_col]
        if len(subset) < _MIN_GROUP_SIZE:
            return None
        rates.append(subset.isna().mean())
    return float(max(rates) - min(rates)) if rates else None


def _stratified_effect_check(
    df: pd.DataFrame, group_col: str, group_values: list, other_col: str, metric_col: str | None
) -> dict | None:
    """The actual Simpson's-paradox question: does `metric_col`'s group comparison
    hold the same direction WITHIN each stratum of `other_col` as it does overall?
    Only defined for the classic 2-group case with a real numeric metric and enough
    per-stratum-per-group data to trust a stratified mean -- returns None (caller
    falls back to the distributional-gap-based severity) whenever it cannot be
    computed reliably, which is common for the MOST extreme confounds (a stratum
    where one group barely appears at all has too little data to check within that
    stratum, even though the aggregate distributional gap is obvious)."""
    if len(group_values) != 2 or not metric_col or metric_col not in df.columns:
        return None
    if not pd.api.types.is_numeric_dtype(df[metric_col]):
        return None

    group_col_as_str = df[group_col].astype(str)
    a, b = str(group_values[0]), str(group_values[1])
    overall_a = df.loc[group_col_as_str == a, metric_col].mean()
    overall_b = df.loc[group_col_as_str == b, metric_col].mean()
    if pd.isna(overall_a) or pd.isna(overall_b) or overall_a == overall_b:
        return None
    overall_diff = overall_a - overall_b

    checked, reversals, shrink_ratios = 0, 0, []
    for stratum in df[other_col].dropna().unique():
        stratum_mask = df[other_col] == stratum
        sub_a = df.loc[stratum_mask & (group_col_as_str == a), metric_col].dropna()
        sub_b = df.loc[stratum_mask & (group_col_as_str == b), metric_col].dropna()
        if len(sub_a) < _STRATIFIED_MIN_GROUP_SIZE or len(sub_b) < _STRATIFIED_MIN_GROUP_SIZE:
            continue
        stratum_diff = sub_a.mean() - sub_b.mean()
        checked += 1
        if (stratum_diff > 0) != (overall_diff > 0):
            reversals += 1
        elif abs(overall_diff) > 1e-9:
            shrink_ratios.append(abs(stratum_diff) / abs(overall_diff))

    if checked == 0:
        return None
    return {
        "checked_strata": checked,
        "reversals": reversals,
        "median_shrink_ratio": float(np.median(shrink_ratios)) if shrink_ratios else None,
    }


def _classify_and_signal(
    df: pd.DataFrame, group_col: str, group_values: list, other_col: str, metric_col: str | None
) -> tuple[ConfoundClassification, str | None, float | None]:
    """Runs the full candidate pipeline for one column and returns
    (classification, signal_description, magnitude). `signal_description` /
    `magnitude` are only meaningful when classification == "confounder".

    Missingness is checked FIRST and independent of the categorical/numeric/
    identifier shape checks below -- a column's null-rate pattern is a real,
    general data-quality signal regardless of whether the column itself is
    categorical, numeric, high-cardinality, or identifier-shaped (a free-text
    "email" column, for instance, is neither a categorical nor a numeric
    candidate by this module's other rules, but its missingness rate can still
    differ meaningfully between the compared groups -- found as a real bug: an
    early return for "neither categorical nor numeric" was silently skipping the
    missingness check entirely for exactly this kind of column)."""
    missing_gap = _missingness_gap(df, group_col, group_values, other_col)
    if missing_gap is not None and missing_gap >= _MISSINGNESS_GAP_THRESHOLD:
        return "confounder", "missingness", missing_gap

    series = df[other_col]
    is_categorical = _is_candidate_categorical(series)
    is_numeric = _is_candidate_numeric(series)
    if not is_categorical and not is_numeric:
        return "irrelevant", None, None
    if is_categorical and _is_identifier(series):
        return "identifier", None, None

    if is_categorical:
        if _is_nested_not_confounded(df, group_col, group_values, other_col):
            return "nested", None, None
        gap = _max_proportion_gap(df, group_col, group_values, other_col)
        if gap is not None and gap >= _CONFOUND_PROPORTION_GAP_THRESHOLD:
            return "confounder", "distribution", gap

    if is_numeric and other_col != metric_col:
        d = _numeric_standardized_gap(df, group_col, group_values, other_col)
        if d is not None and d >= _NUMERIC_CONFOUND_EFFECT_SIZE_THRESHOLD:
            return "confounder", "mean", d

    return "irrelevant", None, None


def _build_limitation(
    df: pd.DataFrame, group_col: str, group_values: list, other_col: str, metric_col: str | None,
    signal: str, magnitude: float, finding_index: int,
) -> Limitation:
    group_desc = ", ".join(str(v) for v in group_values)
    if signal == "distribution":
        base = (
            f"'{other_col}' has a very different mix across the compared '{group_col}' "
            f"groups ({group_desc}) -- this comparison may reflect a difference in "
            f"'{other_col}' rather than a true '{group_col}' effect (a possible "
            "confounding variable)."
        )
        proportion_severe = magnitude >= _SEVERE_CONFOUND_PROPORTION_GAP_THRESHOLD
        gap_note = (
            f" This split is severe (a {round(magnitude * 100)}-point gap) -- the two "
            f"variables are barely distinguishable in this data, so no confident "
            f"conclusion about '{group_col}' alone is possible here."
            if proportion_severe else ""
        )
    elif signal == "mean":
        base = (
            f"'{other_col}' (numeric) differs substantially on average between the "
            f"compared '{group_col}' groups ({group_desc}) -- standardized effect size "
            f"{round(magnitude, 2)} -- this comparison may reflect a difference in "
            f"'{other_col}' rather than a true '{group_col}' effect (a possible "
            "confounding variable)."
        )
        proportion_severe = False
        gap_note = ""
    else:  # missingness
        base = (
            f"'{other_col}' is missing at a very different rate across the compared "
            f"'{group_col}' groups ({group_desc}, a {round(magnitude * 100)}-point gap) -- "
            f"this comparison may partly reflect which group's data is more complete, "
            f"not just a true '{group_col}' effect."
        )
        proportion_severe = magnitude >= _SEVERE_CONFOUND_PROPORTION_GAP_THRESHOLD
        gap_note = ""

    severity: str = "blocks_conclusion" if proportion_severe else "reduces_confidence"
    stratified_note = ""

    if signal in ("distribution", "mean"):
        stratified = _stratified_effect_check(df, group_col, group_values, other_col, metric_col)
        if stratified is not None:
            if stratified["reversals"] > 0:
                severity = "blocks_conclusion"
                stratified_note = (
                    f" Confirmed: within at least one '{other_col}' subgroup with enough data "
                    f"to check, the '{group_col}' comparison actually REVERSES direction -- this "
                    "is a genuine Simpson's-paradox-style reversal, not just a distributional "
                    "imbalance."
                )
            else:
                severity = "reduces_confidence"
                ratio = stratified["median_shrink_ratio"]
                if ratio is not None and ratio < 0.5:
                    stratified_note = (
                        f" Checked directly: within every '{other_col}' subgroup with enough data "
                        f"to check ({stratified['checked_strata']}), the '{group_col}' comparison "
                        "still points the same direction, though noticeably smaller in magnitude -- "
                        f"'{other_col}' is correlated with both '{group_col}' and the outcome, but "
                        "does not fully explain the aggregate gap. This shape is also consistent with "
                        f"'{other_col}' being a downstream consequence of '{group_col}' rather than an "
                        "independent cause (a possible mediator) -- this data alone cannot tell the two "
                        "apart."
                    )
                else:
                    stratified_note = (
                        f" Checked directly: within every '{other_col}' subgroup with enough data to "
                        f"check ({stratified['checked_strata']}), the '{group_col}' comparison still "
                        "points the same direction and a similar magnitude -- this candidate does not "
                        "appear to change the substantive conclusion, though the underlying imbalance "
                        "is still worth being aware of."
                    )

    return Limitation(
        category="methodological",
        text=base + gap_note + stratified_note,
        severity=severity,
        affected_findings=[f"finding_{finding_index}"],
    )


def detect_confounds(df: pd.DataFrame, evidence: list[Evidence]) -> list[Limitation]:
    """Runs against already-gathered evidence and the real dataset -- no new tool
    call, no new LLM call. Deduplicates so the same (group_column, other_column,
    group_values) combination is never flagged twice even if multiple tools compared
    the same groups. Bounded resource use: only columns that pass the cheap
    cardinality/identifier checks are ever fully scored (the same discipline the v1
    module already used), and every candidate resolves to exactly one classification
    (confounder/nested/identifier/irrelevant) so nothing is scanned twice."""
    limitations: list[Limitation] = []
    seen: set[tuple] = set()

    for i, ev in enumerate(evidence):
        r = ev.result_summary
        if not isinstance(r, dict):
            continue
        group_col, group_values = _extract_comparison(r)
        if not group_col or group_col not in df.columns or len(group_values) < 2:
            continue

        for other_col in df.columns:
            if other_col == group_col:
                continue

            classification, signal, magnitude = _classify_and_signal(df, group_col, group_values, other_col, ev.metric)
            if classification != "confounder" or signal is None or magnitude is None:
                continue

            signature = (group_col, other_col, signal, tuple(sorted(str(v) for v in group_values)))
            if signature in seen:
                continue
            seen.add(signature)

            limitations.append(_build_limitation(df, group_col, group_values, other_col, ev.metric, signal, magnitude, i))
    return limitations
