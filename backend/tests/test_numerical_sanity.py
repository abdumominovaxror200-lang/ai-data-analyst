"""Unit tests for app.reasoning.numerical_sanity -- the general-purpose deterministic
numerical/denominator/units sanity checker (Phase 5).

Built in response to a gap flagged independently across three separate benchmark
waves this project has run: no deterministic mechanism previously caught an
impossible or badly-scaled numeric value in a tool's own output before it reached the
user (see the module's own docstring for the full evidence trail). Every test here
checks a concrete, mechanically well-defined property of a tool's real output shape,
not a guess about business semantics.
"""

from __future__ import annotations

from app.reasoning.contracts import Evidence
from app.reasoning.numerical_sanity import check_numerical_sanity
from app.reasoning.verifier import build_findings


def _ev(id_, tool, metric, result_summary, sample_size=None):
    return Evidence(
        id=id_, source_tool=tool, evidence_type="CALCULATED_RESULT", metric=metric,
        result_summary=result_summary, sample_size=sample_size, tool_call_ref=f"tool_call[{id_}]",
    )


# --- impossible percentages ------------------------------------------------------------


def test_a_percentage_field_over_100_is_flagged():
    evidence = [_ev("ev_0", "group_and_aggregate", "conversion", {"match_pct": 800.0})]
    limitations = check_numerical_sanity(evidence)
    assert any("match_pct" in l.text and l.severity == "blocks_conclusion" for l in limitations)


def test_a_negative_percentage_field_is_flagged():
    evidence = [_ev("ev_0", "detect_anomalies", "revenue", {"anomaly_pct": -5.0})]
    limitations = check_numerical_sanity(evidence)
    assert any("anomaly_pct" in l.text for l in limitations)


def test_a_normal_percentage_field_is_not_flagged():
    evidence = [_ev("ev_0", "duplicate_analysis", "revenue", {"duplicate_pct": 6.81})]
    limitations = check_numerical_sanity(evidence)
    assert not limitations


def test_boundary_values_zero_and_100_are_not_flagged():
    evidence = [_ev("ev_0", "duplicate_analysis", "revenue", {"duplicate_pct": 0.0})]
    assert not check_numerical_sanity(evidence)
    evidence2 = [_ev("ev_0", "duplicate_analysis", "revenue", {"duplicate_pct": 100.0})]
    assert not check_numerical_sanity(evidence2)


def test_mape_pct_is_exempt_from_the_100_ceiling():
    """MAPE (mean absolute percentage error) can legitimately exceed 100% -- must
    never be flagged as 'impossible'."""
    evidence = [_ev("ev_0", "forecast", "revenue", {"mape_pct": 340.0})]
    assert not check_numerical_sanity(evidence)


# --- population/denominator mismatch ---------------------------------------------------


def test_wildly_different_sample_sizes_for_the_same_metric_are_flagged():
    evidence = [
        _ev("ev_0", "describe_data", "revenue", {"mean": 380.0}, sample_size=4000),
        _ev("ev_1", "t_test", "revenue", {"p_value": 0.03}, sample_size=6),
    ]
    limitations = check_numerical_sanity(evidence)
    assert any("population" in l.text.lower() for l in limitations)


def test_similar_sample_sizes_are_not_flagged():
    evidence = [
        _ev("ev_0", "describe_data", "revenue", {"mean": 380.0}, sample_size=4000),
        _ev("ev_1", "t_test", "revenue", {"p_value": 0.03}, sample_size=3800),
    ]
    assert not check_numerical_sanity(evidence)


def test_a_single_tool_alone_is_never_flagged_for_population_mismatch():
    evidence = [_ev("ev_0", "describe_data", "revenue", {"mean": 380.0}, sample_size=4000)]
    assert not check_numerical_sanity(evidence)


# --- within-evidence magnitude outliers -------------------------------------------------


def test_a_group_value_wildly_larger_than_the_rest_is_flagged():
    """The finance_units_mismatch trap shape: same metric, one group ~100x the rest."""
    groups = [{"group": "item_0", "value": 40.0}, {"group": "item_1", "value": 45.0}, {"group": "item_2", "value": 4200.0}]
    evidence = [_ev("ev_0", "group_and_aggregate", "amount", {"groups": groups})]
    limitations = check_numerical_sanity(evidence)
    assert any("units" in l.text.lower() or "magnitude" in l.text.lower() or "outlier" in l.text.lower() for l in limitations)


def test_group_values_of_similar_magnitude_are_not_flagged():
    groups = [{"group": "North", "value": 297276.0}, {"group": "South", "value": 302679.0}, {"group": "East", "value": 303489.0}]
    evidence = [_ev("ev_0", "group_and_aggregate", "revenue", {"groups": groups})]
    assert not check_numerical_sanity(evidence)


def test_fewer_than_three_groups_is_not_flagged_regardless_of_magnitude():
    """Not enough groups to distinguish 'one real outlier' from 'this is just what
    two categories look like' -- requires >=3 to compute a meaningful median."""
    groups = [{"group": "A", "value": 10.0}, {"group": "B", "value": 5000.0}]
    evidence = [_ev("ev_0", "group_and_aggregate", "amount", {"groups": groups})]
    assert not check_numerical_sanity(evidence)


# --- integration: wired into build_findings ---------------------------------------------


def test_check_numerical_sanity_is_wired_into_build_findings():
    evidence = [_ev("ev_0", "group_and_aggregate", "conversion", {"match_pct": 800.0})]
    _, limitations = build_findings(evidence)
    assert any("match_pct" in l.text for l in limitations)
