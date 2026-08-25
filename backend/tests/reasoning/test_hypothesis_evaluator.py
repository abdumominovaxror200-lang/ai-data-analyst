"""Tests for Phase 4 P1's evidence-driven hypothesis status
(`app.reasoning.hypothesis_evaluator.update_hypothesis_status`).

Covers all six `HypothesisStatus` values (including "inconclusive", new this phase),
the `evidence_for`/`evidence_against` population rule, and the non-mutation guarantee.
"""

from __future__ import annotations

import copy

from app.reasoning.contracts import Evidence, Hypothesis
from app.reasoning.hypothesis_evaluator import update_hypothesis_status


def _hyp(hyp_id: str, description: str, is_causal: bool = False) -> Hypothesis:
    return Hypothesis(id=hyp_id, description=description, is_causal=is_causal)


def _evidence(
    ev_id: str,
    source_tool: str,
    metric: str | None,
    result_summary: dict,
    evidence_type: str = "STATISTICAL_RESULT",
) -> Evidence:
    return Evidence(
        id=ev_id,
        source_tool=source_tool,
        evidence_type=evidence_type,
        metric=metric,
        result_summary=result_summary,
        tool_call_ref=f"tool_call[{ev_id}]",
    )


# --- supported ----------------------------------------------------------------------


def test_supported_via_significant_one_sample_t_test_matching_direction():
    hyp = _hyp("h1", "Revenue declined in the South region")
    ev = _evidence(
        "ev_0",
        "t_test",
        "revenue",
        {
            "test": "one_sample_t_test",
            "mean": 450.0,
            "popmean": 600.0,
            "statistic": -3.2,
            "p_value": 0.01,
            "alpha": 0.05,
            "significant": True,
        },
    )
    [updated] = update_hypothesis_status([hyp], [ev], [])
    assert updated.status == "supported"
    assert updated.evidence_for == ["ev_0"]
    assert updated.evidence_against == []


def test_supported_when_hypothesis_direction_is_unclaimed_and_evidence_is_significant():
    # No direction keyword in the description -- significance alone is enough per
    # the "ambiguity never manufactures a false contradiction" rule.
    hyp = _hyp("h1", "Revenue differs meaningfully between regions")
    ev = _evidence(
        "ev_0",
        "t_test",
        "revenue",
        {
            "test": "two_sample_t_test",
            "group_a": {"label": "North", "n": 50, "mean": 620.0},
            "group_b": {"label": "South", "n": 40, "mean": 480.0},
            "statistic": 4.1,
            "p_value": 0.001,
            "alpha": 0.05,
            "significant": True,
        },
    )
    [updated] = update_hypothesis_status([hyp], [ev], [])
    assert updated.status == "supported"
    assert updated.evidence_for == ["ev_0"]


# --- weakly_supported -----------------------------------------------------------


def test_weakly_supported_via_suggestive_calculated_aggregate_no_significance_test():
    hyp = _hyp("h2", "Revenue declined in Q2 due to a pricing change")
    ev = _evidence(
        "ev_0",
        "compare_periods",
        "revenue",
        {
            "current_period": {"start": "2024-04-01", "end": "2024-06-30", "value": 8500.0, "n": 91},
            "previous_period": {"start": "2024-01-01", "end": "2024-03-31", "value": 10000.0, "n": 91},
            "delta": -1500.0,
            "pct_change": -15.0,
            "agg_func": "sum",
        },
        evidence_type="CALCULATED_RESULT",
    )
    [updated] = update_hypothesis_status([hyp], [ev], [])
    assert updated.status == "weakly_supported"
    assert updated.evidence_for == ["ev_0"]
    assert updated.evidence_against == []


def test_weakly_supported_via_significant_result_with_small_linked_effect_size():
    hyp = _hyp("h3", "Revenue differs by region")
    ev_ttest = _evidence(
        "ev_0",
        "t_test",
        "revenue",
        {
            "test": "two_sample_t_test",
            "group_a": {"label": "North", "n": 50, "mean": 610.0},
            "group_b": {"label": "South", "n": 48, "mean": 600.0},
            "statistic": 2.1,
            "p_value": 0.04,
            "alpha": 0.05,
            "significant": True,
        },
    )
    ev_effect = _evidence(
        "ev_1",
        "effect_size",
        "revenue",
        {
            "column": "revenue",
            "group_column": "region",
            "group_a": {"label": "North", "n": 50, "mean": 610.0},
            "group_b": {"label": "South", "n": 48, "mean": 600.0},
            "cohens_d": 0.12,
            "magnitude": "negligible",
        },
    )
    [updated] = update_hypothesis_status([hyp], [ev_ttest, ev_effect], [])
    assert updated.status == "weakly_supported"
    assert set(updated.evidence_for) == {"ev_0", "ev_1"}
    assert updated.evidence_against == []


# --- contradicted -----------------------------------------------------------------


def test_contradicted_via_significant_result_in_opposite_direction():
    hyp = _hyp("h4", "Revenue increased due to the new pricing model")
    ev = _evidence(
        "ev_0",
        "t_test",
        "revenue",
        {
            "test": "one_sample_t_test",
            "mean": 450.0,
            "popmean": 600.0,
            "statistic": -3.2,
            "p_value": 0.01,
            "alpha": 0.05,
            "significant": True,
        },
    )
    [updated] = update_hypothesis_status([hyp], [ev], [])
    assert updated.status == "contradicted"
    assert updated.evidence_against == ["ev_0"]
    assert updated.evidence_for == []


# --- inconclusive (new this phase) ------------------------------------------------


def test_inconclusive_via_non_significant_test():
    hyp = _hyp("h5", "Revenue declined due to unusually cold weather")
    ev = _evidence(
        "ev_0",
        "t_test",
        "revenue",
        {
            "test": "one_sample_t_test",
            "mean": 580.0,
            "popmean": 600.0,
            "statistic": -0.4,
            "p_value": 0.42,
            "alpha": 0.05,
            "significant": False,
        },
    )
    [updated] = update_hypothesis_status([hyp], [ev], [])
    assert updated.status == "inconclusive"
    assert updated.evidence_for == []
    assert updated.evidence_against == []


# --- unsupported -------------------------------------------------------------------


def test_unsupported_via_confident_null_calculated_result():
    hyp = _hyp("h6", "Revenue declined due to reduced marketing spend")
    ev = _evidence(
        "ev_0",
        "compare_periods",
        "revenue",
        {
            "current_period": {"start": "2024-04-01", "end": "2024-06-30", "value": 9998.0, "n": 91},
            "previous_period": {"start": "2024-01-01", "end": "2024-03-31", "value": 10000.0, "n": 91},
            "delta": -2.0,
            "pct_change": -0.3,
            "agg_func": "sum",
        },
        evidence_type="CALCULATED_RESULT",
    )
    [updated] = update_hypothesis_status([hyp], [ev], [])
    assert updated.status == "unsupported"
    assert updated.evidence_for == []
    assert updated.evidence_against == []


# --- untested ------------------------------------------------------------------------


def test_untested_when_no_evidence_links_to_the_hypothesis():
    hyp = _hyp("h7", "Sales dropped due to a website outage")
    ev = _evidence(
        "ev_0",
        "profile_dataset",
        None,
        {"rows": 100, "columns": 5},
        evidence_type="FACT",
    )
    [updated] = update_hypothesis_status([hyp], [ev], [])
    assert updated.status == "untested"
    assert updated.evidence_for == []
    assert updated.evidence_against == []


def test_untested_when_evidence_list_is_empty():
    hyp = _hyp("h8", "Sales dropped due to a website outage")
    [updated] = update_hypothesis_status([hyp], [], [])
    assert updated.status == "untested"


# --- non-mutation guarantee ----------------------------------------------------------


def test_does_not_mutate_input_hypotheses_or_evidence():
    hyp = _hyp("h1", "Revenue declined in the South region")
    ev = _evidence(
        "ev_0",
        "t_test",
        "revenue",
        {
            "test": "one_sample_t_test",
            "mean": 450.0,
            "popmean": 600.0,
            "statistic": -3.2,
            "p_value": 0.01,
            "alpha": 0.05,
            "significant": True,
        },
    )
    hyp_snapshot = copy.deepcopy(hyp.model_dump())
    ev_snapshot = copy.deepcopy(ev.model_dump())
    hypotheses_in = [hyp]
    evidence_in = [ev]

    updated = update_hypothesis_status(hypotheses_in, evidence_in, [])

    # The original list objects are untouched (still contain the same original
    # object references, same length).
    assert hypotheses_in == [hyp]
    assert evidence_in == [ev]
    # The original Hypothesis's own field values are unchanged.
    assert hyp.model_dump() == hyp_snapshot
    assert hyp.status == "untested"
    assert hyp.evidence_for == []
    # The original Evidence is unchanged too.
    assert ev.model_dump() == ev_snapshot
    # The returned hypothesis is a genuinely different object, not the same instance.
    assert updated[0] is not hyp
    assert updated[0].status == "supported"


def test_multiple_hypotheses_are_scored_independently():
    hyp_supported = _hyp("h1", "Revenue declined in the South region")
    hyp_untested = _hyp("h2", "Customer satisfaction dropped due to slower shipping")
    ev = _evidence(
        "ev_0",
        "t_test",
        "revenue",
        {
            "test": "one_sample_t_test",
            "mean": 450.0,
            "popmean": 600.0,
            "statistic": -3.2,
            "p_value": 0.01,
            "alpha": 0.05,
            "significant": True,
        },
    )
    updated = update_hypothesis_status([hyp_supported, hyp_untested], [ev], [])
    assert len(updated) == 2
    assert updated[0].status == "supported"
    assert updated[1].status == "untested"
