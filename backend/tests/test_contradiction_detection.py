"""Unit tests for app.reasoning.contradiction_detection -- the deterministic
mean-vs-median (or any two differing aggregations) ranking-contradiction detector
(final professional-analyst stress-test mission, Phase 5).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from app.agent.providers import MockProvider, ProviderResponse, ToolCall
from app.datasets.storage import DatasetRecord
from app.reasoning.contracts import Evidence
from app.reasoning.contradiction_detection import (
    detect_data_quality_contradictions,
    detect_overall_vs_subgroup_contradiction,
    detect_ranking_contradictions,
)
from app.reasoning.orchestrator import ReasoningOrchestrator


def _ev(id_, result_summary):
    return Evidence(
        id=id_, source_tool="group_and_aggregate", evidence_type="CALCULATED_RESULT",
        metric=result_summary.get("agg_column"), result_summary=result_summary, tool_call_ref=f"tool_call[{id_}]",
    )


def _mean_result(groups):
    return {"group_by": "region", "agg_column": "revenue", "agg_func": "mean", "groups": groups, "group_count": len(groups)}


def _median_result(groups):
    return {"group_by": "region", "agg_column": "revenue", "agg_func": "median", "groups": groups, "group_count": len(groups)}


def test_mean_and_median_disagreeing_on_the_top_group_is_flagged():
    """The exact Phase-5 named scenario: mean says A > B, median says B > A --
    one region has a few extreme high-revenue outliers pulling its mean up, but its
    typical (median) performance is actually lower."""
    mean_groups = [{"group": "North", "value": 50000.0}, {"group": "South", "value": 40000.0}]
    median_groups = [{"group": "North", "value": 8000.0}, {"group": "South", "value": 12000.0}]
    evidence = [_ev("ev_0", _mean_result(mean_groups)), _ev("ev_1", _median_result(median_groups))]

    limitations = detect_ranking_contradictions(evidence)
    assert len(limitations) == 1
    assert limitations[0].category == "methodological"
    assert limitations[0].severity == "reduces_confidence"
    assert "North" in limitations[0].text and "South" in limitations[0].text
    assert "mean" in limitations[0].text and "median" in limitations[0].text


def test_mean_and_median_agreeing_on_the_top_group_is_not_flagged():
    mean_groups = [{"group": "North", "value": 50000.0}, {"group": "South", "value": 40000.0}]
    median_groups = [{"group": "North", "value": 12000.0}, {"group": "South", "value": 8000.0}]
    evidence = [_ev("ev_0", _mean_result(mean_groups)), _ev("ev_1", _median_result(median_groups))]
    assert detect_ranking_contradictions(evidence) == []


def test_a_single_aggregation_alone_is_never_flagged():
    mean_groups = [{"group": "North", "value": 50000.0}, {"group": "South", "value": 40000.0}]
    evidence = [_ev("ev_0", _mean_result(mean_groups))]
    assert detect_ranking_contradictions(evidence) == []


def test_different_group_by_or_agg_column_are_never_compared_against_each_other():
    """Two unrelated group_and_aggregate calls (different group_by/agg_column) must
    never be compared as if they were the same underlying comparison."""
    r1 = {"group_by": "region", "agg_column": "revenue", "agg_func": "mean", "groups": [{"group": "North", "value": 1.0}, {"group": "South", "value": 2.0}], "group_count": 2}
    r2 = {"group_by": "product", "agg_column": "quantity", "agg_func": "median", "groups": [{"group": "Widget", "value": 5.0}, {"group": "Gadget", "value": 1.0}], "group_count": 2}
    evidence = [_ev("ev_0", r1), _ev("ev_1", r2)]
    assert detect_ranking_contradictions(evidence) == []


def test_evidence_from_a_different_tool_is_ignored():
    from app.reasoning.contracts import Evidence as _E
    ev = _E(id="ev_0", source_tool="describe_data", evidence_type="CALCULATED_RESULT", metric="revenue",
             result_summary={"group_by": "region", "agg_column": "revenue", "agg_func": "mean", "groups": [{"group": "North", "value": 1.0}, {"group": "South", "value": 2.0}]},
             tool_call_ref="tool_call[ev_0]")
    assert detect_ranking_contradictions([ev]) == []


def test_duplicate_pairs_are_not_flagged_twice():
    mean_groups = [{"group": "North", "value": 50000.0}, {"group": "South", "value": 40000.0}]
    median_groups = [{"group": "North", "value": 8000.0}, {"group": "South", "value": 12000.0}]
    evidence = [
        _ev("ev_0", _mean_result(mean_groups)),
        _ev("ev_1", _median_result(median_groups)),
        _ev("ev_2", _mean_result(mean_groups)),
    ]
    assert len(detect_ranking_contradictions(evidence)) == 1


# --- end-to-end: real ReasoningOrchestrator, real group_and_aggregate calls ------------


def test_ranking_contradiction_reaches_the_final_result_end_to_end():
    """North has one massive outlier deal pulling its mean up; South is more
    consistently mid-sized. mean(North) > mean(South) but median(North) < median(South)
    -- a real, constructed reversal, driven through two real group_and_aggregate calls
    (not synthetic Evidence), proving the check fires on genuine tool output and the
    resulting Limitation survives into the final AnalysisResult."""
    rows = []
    for v in [100.0, 105.0, 95.0, 110.0, 90.0, 10000.0]:
        rows.append({"region": "North", "deal_size": v})
    for v in [500.0, 520.0, 480.0, 510.0, 490.0, 505.0]:
        rows.append({"region": "South", "deal_size": v})
    df = pd.DataFrame(rows)
    record = DatasetRecord(id="x", original_filename="deals.csv", extension=".csv", uploaded_at=pd.Timestamp.utcnow(), df=df, stored_path="u")

    parsed_question = {
        "intent": "comparative", "requested_metrics": ["deal_size"], "requested_dimensions": ["region"],
        "requested_time_range": None, "requested_population": None, "explicit_constraints": [],
        "required_confidence": None, "language": "en", "claims": [],
    }
    plan = {
        "objective": "Compare regions", "capability_categories": ["GENERAL_ANALYSIS"], "steps": [],
        "tools_required": ["group_and_aggregate"], "expected_outputs": [], "validation_steps": [],
        "stopping_conditions": ["done"], "hypotheses": [],
    }
    responses = [
        ProviderResponse(content=json.dumps(parsed_question)),
        ProviderResponse(content=json.dumps(plan)),
        ProviderResponse(content=None, tool_calls=[ToolCall(id="call_1", name="group_and_aggregate", arguments={"group_by": "region", "agg_column": "deal_size", "agg_func": "mean"})]),
        ProviderResponse(content=None, tool_calls=[ToolCall(id="call_2", name="group_and_aggregate", arguments={"group_by": "region", "agg_column": "deal_size", "agg_func": "median"})]),
        ProviderResponse(content="evidence gathered"),
        ProviderResponse(content=json.dumps({"final_answer_text": "North has the higher average deal size.", "recommendation": None})),
    ]
    orchestrator = ReasoningOrchestrator(MockProvider(responses))
    result = orchestrator.analyze(record, "Which region has bigger deals, North or South?")

    contradiction_limitations = [l for l in result.limitations if "ranks" in l.text and "region" in l.text]
    assert contradiction_limitations, f"no ranking contradiction was detected: {[l.text for l in result.limitations]}"
    assert "North" in contradiction_limitations[0].text and "South" in contradiction_limitations[0].text


# --- v2 contradiction engine: overall vs subgroup ---------------------------------------


def _period_ev(id_, pct_change, population=None):
    return Evidence(
        id=id_, source_tool="compare_periods", evidence_type="CALCULATED_RESULT", metric="revenue",
        result_summary={
            "current_period": {"start": "2025-04-01", "end": "2025-04-30", "value": 100.0, "n": 30},
            "previous_period": {"start": "2025-03-01", "end": "2025-03-31", "value": 90.0, "n": 31},
            "delta": 10.0, "pct_change": pct_change, "agg_func": "sum",
        },
        population=population, tool_call_ref=f"tool_call[{id_}]",
    )


def test_overall_increase_with_every_subgroup_decreasing_is_flagged():
    """The mission's own flagship example, verbatim: revenue increased overall but
    every examined subgroup decreased."""
    evidence = [
        _period_ev("ev_0", 15.0, population=None),
        _period_ev("ev_1", -8.0, population="segment == Enterprise"),
        _period_ev("ev_2", -12.0, population="segment == SMB"),
    ]
    limitations = detect_overall_vs_subgroup_contradiction(evidence)
    assert len(limitations) == 1
    assert limitations[0].severity == "blocks_conclusion"
    assert "increased overall" in limitations[0].text
    assert "every examined subgroup" in limitations[0].text


def test_overall_and_subgroups_agreeing_is_not_flagged():
    evidence = [
        _period_ev("ev_0", 15.0, population=None),
        _period_ev("ev_1", 10.0, population="segment == Enterprise"),
        _period_ev("ev_2", 20.0, population="segment == SMB"),
    ]
    assert detect_overall_vs_subgroup_contradiction(evidence) == []


def test_only_one_subgroup_disagreeing_is_not_flagged():
    """A single dissenting subgroup is normal, not a contradiction -- only a
    unanimous reversal across every examined subgroup counts."""
    evidence = [
        _period_ev("ev_0", 15.0, population=None),
        _period_ev("ev_1", -8.0, population="segment == Enterprise"),
        _period_ev("ev_2", 20.0, population="segment == SMB"),
    ]
    assert detect_overall_vs_subgroup_contradiction(evidence) == []


def test_no_overall_call_present_is_not_flagged():
    evidence = [
        _period_ev("ev_1", -8.0, population="segment == Enterprise"),
        _period_ev("ev_2", -12.0, population="segment == SMB"),
    ]
    assert detect_overall_vs_subgroup_contradiction(evidence) == []


def test_fewer_than_two_subgroups_is_not_flagged():
    evidence = [
        _period_ev("ev_0", 15.0, population=None),
        _period_ev("ev_1", -8.0, population="segment == Enterprise"),
    ]
    assert detect_overall_vs_subgroup_contradiction(evidence) == []


def test_overall_vs_subgroup_reaches_the_final_result_end_to_end():
    """Real end-to-end proof: three real compare_periods tool calls (one
    unfiltered, two filtered) against a constructed dataset where this pattern
    genuinely exists -- a real mix-shift Simpson's paradox on AVERAGE order value
    (mean, not sum: a straightforward sum can never produce this pattern when
    segments fully partition the data, since segment sums must add up to the
    overall sum by simple arithmetic -- verified by direct computation before
    writing this test that every segment's own mean genuinely decreases while more
    order volume shifts toward the high-value Enterprise segment, pulling the
    OVERALL mean up: overall March mean ~145, April mean ~588; Enterprise March
    ~1002, April ~949 (down); SMB March ~50, April ~48 (down))."""
    rng = np.random.default_rng(1)
    rows = []
    for _ in range(10):
        rows.append({"date": pd.Timestamp("2025-03-15"), "segment": "Enterprise", "order_value": 1000.0 + rng.normal(0, 10)})
    for _ in range(90):
        rows.append({"date": pd.Timestamp("2025-03-15"), "segment": "SMB", "order_value": 50.0 + rng.normal(0, 2)})
    for _ in range(60):
        rows.append({"date": pd.Timestamp("2025-04-15"), "segment": "Enterprise", "order_value": 950.0 + rng.normal(0, 10)})
    for _ in range(40):
        rows.append({"date": pd.Timestamp("2025-04-15"), "segment": "SMB", "order_value": 48.0 + rng.normal(0, 2)})
    df = pd.DataFrame(rows)
    record = DatasetRecord(id="x", original_filename="x.csv", extension=".csv", uploaded_at=pd.Timestamp.utcnow(), df=df, stored_path="u")

    parsed_question = {
        "intent": "diagnostic", "requested_metrics": ["order_value"], "requested_dimensions": ["segment"],
        "requested_time_range": None, "requested_population": None, "explicit_constraints": [],
        "required_confidence": None, "language": "en", "claims": [],
    }
    plan = {
        "objective": "Check average order value trend by segment", "capability_categories": ["GENERAL_ANALYSIS"], "steps": [],
        "tools_required": ["compare_periods"], "expected_outputs": [], "validation_steps": [],
        "stopping_conditions": ["done"], "hypotheses": [],
    }
    common_args = {
        "date_column": "date", "value_column": "order_value", "agg_func": "mean",
        "current_start": "2025-04-01", "current_end": "2025-04-30",
        "previous_start": "2025-03-01", "previous_end": "2025-03-31",
    }
    responses = [
        ProviderResponse(content=json.dumps(parsed_question)),
        ProviderResponse(content=json.dumps(plan)),
        ProviderResponse(content=None, tool_calls=[ToolCall(id="c1", name="compare_periods", arguments=dict(common_args))]),
        ProviderResponse(content=None, tool_calls=[ToolCall(id="c2", name="compare_periods", arguments={**common_args, "filters": [{"column": "segment", "op": "==", "value": "Enterprise"}]})]),
        ProviderResponse(content=None, tool_calls=[ToolCall(id="c3", name="compare_periods", arguments={**common_args, "filters": [{"column": "segment", "op": "==", "value": "SMB"}]})]),
        ProviderResponse(content="evidence gathered"),
        ProviderResponse(content=json.dumps({"final_answer_text": "Average order value is summarized above.", "recommendation": None})),
    ]
    orchestrator = ReasoningOrchestrator(MockProvider(responses))
    result = orchestrator.analyze(record, "How did average order value change month over month by segment?")

    contradiction_limitations = [l for l in result.limitations if "overall" in l.text and "subgroup" in l.text]
    assert contradiction_limitations, f"no overall-vs-subgroup contradiction detected: {[l.text for l in result.limitations]}"
    assert contradiction_limitations[0].severity == "blocks_conclusion"


# --- v2 contradiction engine: conflicting data-quality signals ---------------------------


def _quality_ev(id_, tool, result_summary, population=None):
    return Evidence(
        id=id_, source_tool=tool, evidence_type="CALCULATED_RESULT", metric=None,
        result_summary=result_summary, population=population, tool_call_ref=f"tool_call[{id_}]",
    )


def test_clean_report_vs_dirty_anomaly_scan_is_flagged():
    evidence = [
        _quality_ev("ev_0", "data_quality_report", {"quality_issues": []}),
        _quality_ev("ev_1", "detect_anomalies", {"anomaly_pct": 42.0}),
    ]
    limitations = detect_data_quality_contradictions(evidence)
    assert len(limitations) == 1
    assert limitations[0].severity == "reduces_confidence"
    assert "conflicting data-quality signals" in limitations[0].text


def test_both_tools_agreeing_clean_is_not_flagged():
    evidence = [
        _quality_ev("ev_0", "data_quality_report", {"quality_issues": []}),
        _quality_ev("ev_1", "detect_anomalies", {"anomaly_pct": 2.0}),
    ]
    assert detect_data_quality_contradictions(evidence) == []


def test_both_tools_agreeing_dirty_is_not_flagged():
    evidence = [
        _quality_ev("ev_0", "data_quality_report", {"quality_issues": ["dup rows"]}),
        _quality_ev("ev_1", "detect_anomalies", {"anomaly_pct": 42.0}),
    ]
    assert detect_data_quality_contradictions(evidence) == []


def test_different_populations_are_not_compared_against_each_other():
    """Two verification tools examining DIFFERENT scopes disagreeing is not a real
    contradiction -- it may just mean the subsets genuinely differ."""
    evidence = [
        _quality_ev("ev_0", "data_quality_report", {"quality_issues": []}, population="region == North"),
        _quality_ev("ev_1", "detect_anomalies", {"anomaly_pct": 42.0}, population="region == South"),
    ]
    assert detect_data_quality_contradictions(evidence) == []


def test_a_single_verification_tool_alone_is_never_flagged():
    evidence = [_quality_ev("ev_0", "detect_anomalies", {"anomaly_pct": 42.0})]
    assert detect_data_quality_contradictions(evidence) == []
