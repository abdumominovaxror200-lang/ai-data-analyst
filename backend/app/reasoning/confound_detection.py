"""Deterministic confounding-variable detection (Phase 5).

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
(`t_test`, `group_and_aggregate`), using the real dataset the comparison ran against --
zero new LLM calls, following the same "read already-gathered evidence plus the
dataset itself" discipline as `verifier.py`'s other deterministic checks.

**What it checks**: for any evidence that compared 2+ groups on some column (the
comparison is read directly off the tool's own real result shape -- `group_column`/
`group_a`/`group_b` for `t_test`, `group_by`/`groups` for `group_and_aggregate` --
confirmed against `app/tools/hypothesis.py` and `app/tools/aggregation.py`, not
guessed), this scans every OTHER low-cardinality categorical column in the dataset for
whether its distribution differs sharply between the compared groups. A large gap
(one category makes up a very different share of each group) is a real, mechanically
detectable confound signal -- exactly the store-format pattern that caused the live
failure above.

**Deliberately scoped to categorical-vs-categorical confounds only** (not continuous
numeric confounds, which would need a proper group-mean-difference test per candidate
column -- a larger undertaking not justified by the evidence gathered so far).
"""

from __future__ import annotations

import pandas as pd

from app.reasoning.contracts import Evidence, Limitation

_MIN_GROUP_SIZE = 5  # each compared group needs at least this many rows to trust a proportion
_MIN_CATEGORY_LEVELS = 2
_MAX_CATEGORY_LEVELS = 20  # avoid scanning near-unique/ID-like columns as "categorical"
_CONFOUND_PROPORTION_GAP_THRESHOLD = 0.40  # a 40-point swing in a category's share is not noise
_MIN_SHARED_VALUE_FRACTION = 0.3  # see _is_nested_not_confounded's docstring
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


def _extract_comparison(result_summary: dict) -> tuple[str | None, list]:
    """Reads which column and which specific values were compared directly off a
    tool's own real result shape -- never re-derived or guessed."""
    group_col = result_summary.get("group_column")
    if group_col:
        values = []
        for key in ("group_a", "group_b"):
            g = result_summary.get(key)
            if isinstance(g, dict) and "label" in g:
                values.append(g["label"])
        if len(values) >= 2:
            return group_col, values

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
    return _MIN_CATEGORY_LEVELS <= n_unique <= _MAX_CATEGORY_LEVELS


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


def detect_confounds(df: pd.DataFrame, evidence: list[Evidence]) -> list[Limitation]:
    """Runs against already-gathered evidence and the real dataset -- no new tool
    call, no new LLM call. Deduplicates so the same (group_column, other_column,
    group_values) combination is never flagged twice even if multiple tools compared
    the same groups."""
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
            if other_col == group_col or other_col == ev.metric:
                continue
            if not _is_candidate_categorical(df[other_col]):
                continue
            if _is_nested_not_confounded(df, group_col, group_values, other_col):
                continue

            gap = _max_proportion_gap(df, group_col, group_values, other_col)
            if gap is None or gap < _CONFOUND_PROPORTION_GAP_THRESHOLD:
                continue

            signature = (group_col, other_col, tuple(sorted(str(v) for v in group_values)))
            if signature in seen:
                continue
            seen.add(signature)

            limitations.append(
                Limitation(
                    category="methodological",
                    text=(
                        f"'{other_col}' is distributed very differently across the compared "
                        f"'{group_col}' groups ({', '.join(str(v) for v in group_values)}) -- "
                        f"this comparison may reflect a difference in '{other_col}' rather than "
                        f"a true '{group_col}' effect (a possible confounding variable)."
                    ),
                    severity="reduces_confidence",
                    affected_findings=[f"finding_{i}"],
                )
            )
    return limitations
