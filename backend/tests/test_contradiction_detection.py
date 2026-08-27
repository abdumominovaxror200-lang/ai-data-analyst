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
from app.reasoning.contradiction_detection import detect_ranking_contradictions
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
