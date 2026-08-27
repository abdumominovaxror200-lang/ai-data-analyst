"""Unit tests for app.reasoning.confound_detection -- the deterministic confounding-
variable detector (Phase 5).

Built in direct response to a REAL failure confirmed live against the configured
provider (see .agent/hard_realworld_real_llm_spotcheck.md and the module's own
docstring): asked to compare North vs. South region on average basket size, the real
model ran a plain t_test and never noticed North is 90% large-format stores while
South is 90% small-format stores -- a textbook confound. These tests verify the
detector catches exactly that pattern from the tool's own real result shape, and does
not fire on legitimate, unconfounded comparisons or on noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.reasoning.confound_detection import detect_confounds
from app.reasoning.contracts import Evidence


def _ev(id_, tool, metric, result_summary):
    return Evidence(
        id=id_, source_tool=tool, evidence_type="STATISTICAL_RESULT", metric=metric,
        result_summary=result_summary, tool_call_ref=f"tool_call[{id_}]",
    )


def _confounded_df() -> pd.DataFrame:
    """Mirrors the real region_size_confound fixture's shape: region strongly
    predicts format (North=90% large, South=90% small)."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(18):
        rows.append({"region": "North", "format": "large", "avg_basket": rng.normal(150, 5)})
    for i in range(2):
        rows.append({"region": "North", "format": "small", "avg_basket": rng.normal(70, 5)})
    for i in range(2):
        rows.append({"region": "South", "format": "large", "avg_basket": rng.normal(148, 5)})
    for i in range(18):
        rows.append({"region": "South", "format": "small", "avg_basket": rng.normal(80, 5)})
    return pd.DataFrame(rows)


def _unconfounded_df() -> pd.DataFrame:
    """Format is split roughly evenly within both regions -- no confound."""
    rng = np.random.default_rng(2)
    regions = ["North"] * 20 + ["South"] * 20
    formats = (["large"] * 10 + ["small"] * 10) * 2
    return pd.DataFrame({"region": regions, "format": formats, "avg_basket": rng.normal(120, 10, 40)})


# --- t_test-shaped evidence ------------------------------------------------------------


def test_flags_a_real_confound_from_a_t_test_comparison():
    df = _confounded_df()
    evidence = [
        _ev("ev_0", "t_test", "avg_basket", {
            "group_column": "region", "group_a": {"label": "North", "n": 20}, "group_b": {"label": "South", "n": 20},
        }),
    ]
    limitations = detect_confounds(df, evidence)
    assert any("format" in l.text and "region" in l.text for l in limitations)
    assert limitations[0].category == "methodological"


def test_no_confound_flagged_when_the_other_variable_is_evenly_split():
    df = _unconfounded_df()
    evidence = [
        _ev("ev_0", "t_test", "avg_basket", {
            "group_column": "region", "group_a": {"label": "North", "n": 20}, "group_b": {"label": "South", "n": 20},
        }),
    ]
    assert not detect_confounds(df, evidence)


# --- group_and_aggregate-shaped evidence ------------------------------------------------


def test_flags_a_real_confound_from_a_group_and_aggregate_comparison():
    df = _confounded_df()
    evidence = [
        _ev("ev_0", "group_and_aggregate", "avg_basket", {
            "group_by": "region", "groups": [{"group": "North", "value": 143.0}, {"group": "South", "value": 88.0}],
        }),
    ]
    limitations = detect_confounds(df, evidence)
    assert any("format" in l.text for l in limitations)


# --- guardrails: no false positives -----------------------------------------------------


def test_no_evidence_at_all_produces_no_limitations():
    assert not detect_confounds(_confounded_df(), [])


def test_evidence_with_no_group_comparison_is_ignored():
    df = _confounded_df()
    evidence = [_ev("ev_0", "describe_data", "avg_basket", {"columns": {}})]
    assert not detect_confounds(df, evidence)


def test_small_group_size_is_not_flagged_even_if_skewed():
    """Fewer than _MIN_GROUP_SIZE rows in a compared group -- too little data to
    trust a proportion, must not fire."""
    df = pd.DataFrame({
        "region": ["North"] * 3 + ["South"] * 3,
        "format": ["large", "large", "large", "small", "small", "small"],
        "avg_basket": [150.0, 148.0, 152.0, 80.0, 78.0, 82.0],
    })
    evidence = [_ev("ev_0", "t_test", "avg_basket", {
        "group_column": "region", "group_a": {"label": "North"}, "group_b": {"label": "South"},
    })]
    assert not detect_confounds(df, evidence)


def test_high_cardinality_columns_are_not_scanned_as_categorical_confounds():
    """A near-unique ID-like column must never be treated as a categorical confound
    candidate -- it would trivially 'differ' between any two groups."""
    df = _confounded_df()
    df["store_id"] = [f"S{i:03d}" for i in range(len(df))]
    evidence = [_ev("ev_0", "t_test", "avg_basket", {
        "group_column": "region", "group_a": {"label": "North"}, "group_b": {"label": "South"},
    })]
    limitations = detect_confounds(df, evidence)
    assert not any("store_id" in l.text for l in limitations)


def test_the_compared_metric_column_itself_is_never_flagged_as_its_own_confound():
    df = _confounded_df()
    evidence = [_ev("ev_0", "t_test", "avg_basket", {
        "group_column": "region", "group_a": {"label": "North"}, "group_b": {"label": "South"},
    })]
    limitations = detect_confounds(df, evidence)
    assert not any(l.affected_findings and "avg_basket" in l.text.split("'")[1] for l in limitations if "'avg_basket'" in l.text[:20])


def test_a_nested_hierarchical_column_is_not_flagged_as_a_confound():
    """Real false positive found while verifying this detector against the actual
    primary dataset: 'product' was flagged as confounding a 'category' comparison,
    when category IS product's own grouping -- every product belongs to exactly one
    category. This is a nested/hierarchical relationship, not an independent
    confound, and must never be flagged."""
    df = pd.DataFrame({
        "category": ["Electronics"] * 10 + ["Apparel"] * 10,
        "product": (["Laptop"] * 5 + ["Phone"] * 5) + (["Shirt"] * 5 + ["Pants"] * 5),
        "revenue": np.random.default_rng(3).normal(100, 10, 20),
    })
    evidence = [_ev("ev_0", "group_and_aggregate", "revenue", {
        "group_by": "category", "groups": [{"group": "Electronics", "value": 500.0}, {"group": "Apparel", "value": 480.0}],
    })]
    limitations = detect_confounds(df, evidence)
    assert not any("product" in l.text for l in limitations)


def test_a_genuine_confound_with_partial_overlap_is_still_flagged():
    """A confound doesn't require every value to appear in every group -- only that
    a meaningful share of other_col's values are genuinely shared, not almost
    entirely partitioned by group (the nested case above)."""
    rng = np.random.default_rng(4)
    rows = []
    for _ in range(15):
        rows.append({"region": "North", "tier": "premium", "revenue": rng.normal(200, 10)})
    for _ in range(5):
        rows.append({"region": "North", "tier": "standard", "revenue": rng.normal(100, 10)})
    for _ in range(5):
        rows.append({"region": "South", "tier": "premium", "revenue": rng.normal(200, 10)})
    for _ in range(15):
        rows.append({"region": "South", "tier": "standard", "revenue": rng.normal(100, 10)})
    df = pd.DataFrame(rows)
    evidence = [_ev("ev_0", "t_test", "revenue", {
        "group_column": "region", "group_a": {"label": "North"}, "group_b": {"label": "South"},
    })]
    limitations = detect_confounds(df, evidence)
    assert any("tier" in l.text for l in limitations)


def test_datetime_group_column_does_not_raise_or_warn():
    """Real edge case found via pytest's warning capture: a group_and_aggregate call
    grouped by a real datetime64 column (e.g. group_by='date') reports string group
    labels in its 'groups' list -- comparing a datetime64 Series against those
    strings via a bare `.isin`/`==` either raises a pandas FutureWarning or silently
    matches nothing. Must work correctly (and warning-free) either way."""
    import warnings

    df = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01"] * 10 + ["2025-02-01"] * 10),
        "channel": (["web"] * 8 + ["app"] * 2) + (["web"] * 2 + ["app"] * 8),
        "revenue": np.random.default_rng(5).normal(100, 10, 20),
    })
    evidence = [_ev("ev_0", "group_and_aggregate", "revenue", {
        "group_by": "date", "groups": [{"group": "2025-01-01", "value": 1000.0}, {"group": "2025-02-01", "value": 900.0}],
    })]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        limitations = detect_confounds(df, evidence)
    assert any("channel" in l.text for l in limitations)


def test_deduplicates_repeated_comparisons_of_the_same_groups():
    df = _confounded_df()
    evidence = [
        _ev("ev_0", "t_test", "avg_basket", {"group_column": "region", "group_a": {"label": "North"}, "group_b": {"label": "South"}}),
        _ev("ev_1", "group_and_aggregate", "avg_basket", {"group_by": "region", "groups": [{"group": "North", "value": 143.0}, {"group": "South", "value": 88.0}]}),
    ]
    limitations = detect_confounds(df, evidence)
    signatures = {(l.text) for l in limitations}
    assert len(limitations) == len(signatures)  # no duplicate text entries for the same finding
