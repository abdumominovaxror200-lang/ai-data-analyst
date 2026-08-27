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


def test_severe_confound_gets_blocks_conclusion_severity():
    """_confounded_df() has an 80-point gap (90% large in North, 10% large in South)
    -- above _SEVERE_CONFOUND_PROPORTION_GAP_THRESHOLD (0.70). Severity must escalate
    to blocks_conclusion so recommendation_grounding.py's global override actually
    applies (see test_blocks_conclusion_enforcement.py for the full end-to-end
    verification this unit-level check backs up)."""
    df = _confounded_df()
    evidence = [_ev("ev_0", "t_test", "avg_basket", {
        "group_column": "region", "group_a": {"label": "North", "n": 20}, "group_b": {"label": "South", "n": 20},
    })]
    limitations = detect_confounds(df, evidence)
    format_limitation = next(l for l in limitations if "format" in l.text)
    assert format_limitation.severity == "blocks_conclusion"
    assert "severe" in format_limitation.text.lower()


def test_moderate_confound_stays_at_reduces_confidence_severity():
    """A real but more moderate gap (below the severe threshold, above the base
    detection threshold), where the v2 stratified-effect check (both strata have
    >= 5 rows per group here, so it CAN run) directly confirms the region comparison
    points the same direction within every stratum -- should NOT be escalated to
    blocks_conclusion. 75%-vs-25% is a 50-point distributional gap, real and
    flaggable, but not the near-total 70+ point split the severe escalation is
    reserved for, and the stratified check finds no reversal to escalate on either.

    (This scenario deliberately differs from an earlier version of this test that
    used group sizes producing a genuine within-stratum reversal in the "small"
    stratum -- that was found, while building the v2 stratified-effect check, to
    correctly escalate to blocks_conclusion, which is the MORE correct behavior: a
    real reversal is stronger evidence than a distributional gap alone. That
    scenario now lives in
    test_stratified_reversal_escalates_to_blocks_conclusion_even_with_a_moderate_gap
    below.)"""
    rows = []
    for _ in range(15):
        rows.append({"region": "North", "format": "large", "avg_basket": 150.0})
    for _ in range(5):
        rows.append({"region": "North", "format": "small", "avg_basket": 140.0})
    for _ in range(5):
        rows.append({"region": "South", "format": "large", "avg_basket": 100.0})
    for _ in range(15):
        rows.append({"region": "South", "format": "small", "avg_basket": 90.0})
    df = pd.DataFrame(rows)
    evidence = [_ev("ev_0", "t_test", "avg_basket", {
        "group_column": "region", "group_a": {"label": "North", "n": 20}, "group_b": {"label": "South", "n": 20},
    })]
    limitations = detect_confounds(df, evidence)
    format_limitation = next(l for l in limitations if "format" in l.text)
    assert format_limitation.severity == "reduces_confidence"
    assert "severe" not in format_limitation.text.lower()
    assert "same direction" in format_limitation.text.lower()


def test_stratified_reversal_escalates_to_blocks_conclusion_even_with_a_moderate_gap():
    """v2 confound engine: a genuine within-stratum REVERSAL (North looks WORSE than
    South specifically within the well-powered "small" stratum, even though North
    looks much better in aggregate) is stronger evidence than the raw distributional
    gap alone, and escalates to blocks_conclusion even though the gap itself (a
    50-point split: North 55%/45% large/small vs South 15%/85%) is only "moderate",
    below _SEVERE_CONFOUND_PROPORTION_GAP_THRESHOLD. Real numbers verified by direct
    computation: aggregate North=114.0 > South=90.2, but within "small"
    (North n=9, South n=17, both >= the stratified minimum) North=70.0 < South=80.0
    -- a real reversal, not a rounding artifact."""
    rows = []
    for _ in range(11):
        rows.append({"region": "North", "format": "large", "avg_basket": 150.0})
    for _ in range(9):
        rows.append({"region": "North", "format": "small", "avg_basket": 70.0})
    for _ in range(3):
        rows.append({"region": "South", "format": "large", "avg_basket": 148.0})
    for _ in range(17):
        rows.append({"region": "South", "format": "small", "avg_basket": 80.0})
    df = pd.DataFrame(rows)
    evidence = [_ev("ev_0", "t_test", "avg_basket", {
        "group_column": "region", "group_a": {"label": "North", "n": 20}, "group_b": {"label": "South", "n": 20},
    })]
    limitations = detect_confounds(df, evidence)
    format_limitation = next(l for l in limitations if "format" in l.text)
    assert format_limitation.severity == "blocks_conclusion"
    assert "reverses direction" in format_limitation.text.lower()


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


# --- anova_test-shaped evidence (found missing entirely, added as its own branch) ------


def test_flags_a_real_confound_from_an_anova_test_comparison():
    """Real gap found via direct testing: anova_test's real result shape
    ("group_column" + a "groups" DICT keyed by group name, not a list and not
    group_a/group_b) matched neither of _extract_comparison's original two
    branches, so a 3+-group ANOVA comparison got zero confound-checking coverage at
    all until this branch was added."""
    df = _confounded_df()
    evidence = [
        _ev("ev_0", "anova_test", "avg_basket", {
            "test": "one_way_anova", "value_column": "avg_basket", "group_column": "region",
            "groups": {"North": {"n": 20, "mean": 143.0}, "South": {"n": 20, "mean": 88.0}}, "statistic": 12.0,
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


def test_a_three_way_confound_with_extreme_per_group_skew_is_still_flagged():
    """Real bug found via direct testing: an 18/1/1-style three-way split (each
    format value overwhelmingly concentrated in ONE of three regions, with only 1
    minority row in each of the other two) has a minority presence of just 1 row per
    group -- below any absolute-count presence floor greater than 1, which
    previously caused this genuine confound to be misclassified as 'nested' and
    silently skipped. A fraction-based presence check (>=3% of a group's rows)
    correctly recognizes the minority rows as real presence, not noise, and still
    flags the confound."""
    rng = np.random.default_rng(7)
    rows = []
    for region, fmt in [("North", "large"), ("South", "small"), ("East", "medium")]:
        for _ in range(18):
            rows.append({"region": region, "format": fmt, "revenue": rng.normal(150, 10)})
        for other_fmt in [f for f in ("large", "small", "medium") if f != fmt]:
            rows.append({"region": region, "format": other_fmt, "revenue": rng.normal(100, 10)})
    df = pd.DataFrame(rows)
    evidence = [_ev("ev_0", "group_and_aggregate", "revenue", {
        "group_by": "region", "groups": [{"group": "North", "value": 1}, {"group": "South", "value": 2}, {"group": "East", "value": 3}],
    })]
    limitations = detect_confounds(df, evidence)
    assert any("format" in l.text for l in limitations)


def test_deduplicates_repeated_comparisons_of_the_same_groups():
    df = _confounded_df()
    evidence = [
        _ev("ev_0", "t_test", "avg_basket", {"group_column": "region", "group_a": {"label": "North"}, "group_b": {"label": "South"}}),
        _ev("ev_1", "group_and_aggregate", "avg_basket", {"group_by": "region", "groups": [{"group": "North", "value": 143.0}, {"group": "South", "value": 88.0}]}),
    ]
    limitations = detect_confounds(df, evidence)
    signatures = {(l.text) for l in limitations}
    assert len(limitations) == len(signatures)  # no duplicate text entries for the same finding


# --- v2 confound engine: numeric confounds -----------------------------------------------


def test_a_numeric_candidate_with_a_large_standardized_mean_gap_is_flagged():
    """A NUMERIC confound (customer tenure) -- the v1 module was explicitly scoped to
    categorical-only. North customers are, on average, much longer-tenured than South
    customers, and tenure itself plausibly drives basket size."""
    rng = np.random.default_rng(3)
    n = 15
    df = pd.DataFrame({
        "region": ["North"] * n + ["South"] * n,
        "tenure_months": np.concatenate([rng.normal(48, 4, n), rng.normal(6, 4, n)]),
        "avg_basket": np.concatenate([rng.normal(150, 5, n), rng.normal(90, 5, n)]),
    })
    evidence = [_ev("ev_0", "t_test", "avg_basket", {
        "group_column": "region", "group_a": {"label": "North", "n": n}, "group_b": {"label": "South", "n": n},
    })]
    limitations = detect_confounds(df, evidence)
    tenure_limitations = [l for l in limitations if "tenure_months" in l.text]
    assert tenure_limitations, f"no numeric confound was detected: {[l.text for l in limitations]}"
    assert "standardized effect size" in tenure_limitations[0].text


def test_a_numeric_candidate_with_similar_means_is_not_flagged():
    rng = np.random.default_rng(4)
    n = 15
    df = pd.DataFrame({
        "region": ["North"] * n + ["South"] * n,
        "tenure_months": np.concatenate([rng.normal(24, 4, n), rng.normal(23, 4, n)]),
        "avg_basket": np.concatenate([rng.normal(150, 5, n), rng.normal(90, 5, n)]),
    })
    evidence = [_ev("ev_0", "t_test", "avg_basket", {
        "group_column": "region", "group_a": {"label": "North", "n": n}, "group_b": {"label": "South", "n": n},
    })]
    limitations = detect_confounds(df, evidence)
    assert not any("tenure_months" in l.text for l in limitations)


def test_the_metric_column_itself_is_never_flagged_as_a_numeric_confound_of_itself():
    """avg_basket obviously 'differs' between the groups (that's the comparison
    being made) -- it must never be flagged as a numeric confound OF ITSELF."""
    df = _confounded_df()
    evidence = [_ev("ev_0", "t_test", "avg_basket", {
        "group_column": "region", "group_a": {"label": "North"}, "group_b": {"label": "South"},
    })]
    limitations = detect_confounds(df, evidence)
    assert not any(l.text.startswith("'avg_basket' (numeric)") for l in limitations)


# --- v2 confound engine: missingness confounds -------------------------------------------


def test_a_large_missingness_gap_is_flagged():
    rng = np.random.default_rng(5)
    n = 20
    email_north = [f"n{i}@x.com" for i in range(n)]
    for i in range(1):  # 5% missing in North
        email_north[i] = None
    email_south = [f"s{i}@x.com" for i in range(n)]
    for i in range(13):  # 65% missing in South
        email_south[i] = None
    df = pd.DataFrame({
        "region": ["North"] * n + ["South"] * n,
        "email": email_north + email_south,
        "avg_basket": rng.normal(120, 10, 2 * n),
    })
    evidence = [_ev("ev_0", "t_test", "avg_basket", {
        "group_column": "region", "group_a": {"label": "North", "n": n}, "group_b": {"label": "South", "n": n},
    })]
    limitations = detect_confounds(df, evidence)
    email_limitations = [l for l in limitations if "email" in l.text]
    assert email_limitations, f"no missingness confound was detected: {[l.text for l in limitations]}"
    assert "missing" in email_limitations[0].text.lower()


def test_similar_missingness_rates_are_not_flagged():
    rng = np.random.default_rng(6)
    n = 20
    df = pd.DataFrame({
        "region": ["North"] * n + ["South"] * n,
        "email": [f"n{i}@x.com" if i % 10 else None for i in range(n)] + [f"s{i}@x.com" if i % 10 else None for i in range(n)],
        "avg_basket": rng.normal(120, 10, 2 * n),
    })
    evidence = [_ev("ev_0", "t_test", "avg_basket", {
        "group_column": "region", "group_a": {"label": "North", "n": n}, "group_b": {"label": "South", "n": n},
    })]
    limitations = detect_confounds(df, evidence)
    assert not any("email" in l.text and "missing" in l.text.lower() for l in limitations)


# --- v2 confound engine: identifier exclusion ---------------------------------------------


def test_a_near_unique_identifier_column_is_never_flagged_even_if_skewed():
    """A per-row customer ID is never a plausible confounding VARIABLE -- even
    though every value trivially 'differs' between groups (every ID is unique), it
    must be excluded as an identifier, not reported as a confound."""
    rng = np.random.default_rng(7)
    n = 15
    df = pd.DataFrame({
        "region": ["North"] * n + ["South"] * n,
        "customer_id": [f"CUST-{i:05d}" for i in range(2 * n)],
        "avg_basket": np.concatenate([rng.normal(150, 5, n), rng.normal(90, 5, n)]),
    })
    evidence = [_ev("ev_0", "t_test", "avg_basket", {
        "group_column": "region", "group_a": {"label": "North", "n": n}, "group_b": {"label": "South", "n": n},
    })]
    limitations = detect_confounds(df, evidence)
    assert not any("customer_id" in l.text for l in limitations)


# --- v2 confound engine: stratified check downgrades + mediator caveat --------------------


def test_consistent_within_strata_but_much_smaller_magnitude_adds_a_possible_mediator_note():
    """A candidate whose stratified check finds the SAME direction in every checked
    stratum, but a much SMALLER magnitude than the aggregate gap (most of the
    aggregate gap comes from compositional imbalance, not the within-stratum effect)
    -- this shape is equally consistent with a true confound or with the candidate
    being a downstream mediator, and the text must say so honestly rather than pick
    one causal story."""
    rows = []
    for _ in range(15):
        rows.append({"region": "North", "format": "large", "avg_basket": 102.0})
    for _ in range(5):
        rows.append({"region": "North", "format": "small", "avg_basket": 52.0})
    for _ in range(5):
        rows.append({"region": "South", "format": "large", "avg_basket": 100.0})
    for _ in range(15):
        rows.append({"region": "South", "format": "small", "avg_basket": 50.0})
    df = pd.DataFrame(rows)
    evidence = [_ev("ev_0", "t_test", "avg_basket", {
        "group_column": "region", "group_a": {"label": "North", "n": 20}, "group_b": {"label": "South", "n": 20},
    })]
    limitations = detect_confounds(df, evidence)
    format_limitation = next(l for l in limitations if "format" in l.text)
    assert format_limitation.severity == "reduces_confidence"
    assert "mediator" in format_limitation.text.lower()
    assert "reverses direction" not in format_limitation.text.lower()
