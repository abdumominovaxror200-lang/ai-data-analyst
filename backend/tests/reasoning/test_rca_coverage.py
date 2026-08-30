"""Deterministic RCA coverage tests. No live provider is used."""

from __future__ import annotations

from app.agent.providers import MockProvider
from app.reasoning import orchestrator as orchestrator_module
from app.reasoning.contracts import AnalysisPlan, AnalyticalQuestion, Evidence, Finding, Recommendation
from app.reasoning.coverage import assess_coverage
from app.reasoning.orchestrator import ReasoningOrchestrator


def _question(*, dimensions=None, time_range=None):
    return AnalyticalQuestion(original_question="Why did revenue change?", intent="diagnostic",
                              requested_metrics=["revenue"], requested_dimensions=dimensions or [],
                              requested_time_range=time_range)


def _plan(*tools, categories=None):
    return AnalysisPlan(objective="Explain the revenue change",
                        capability_categories=categories or ["GENERAL_ANALYSIS"],
                        tools_required=list(tools))


def _evidence(tool, index, *, group_by=None, evidence_type="CALCULATED_RESULT"):
    summary = {"value": index + 1}
    if group_by:
        summary["group_by"] = group_by
    return Evidence(id=f"ev_{index}", source_tool=tool, evidence_type=evidence_type,
                    result_summary=summary, tool_call_ref=f"tool_call[{index}]")


def test_temporal_grouping_cannot_satisfy_segment_coverage():
    assessment = assess_coverage(
        _question(dimensions=["region"], time_range="2024 vs 2025"),
        _plan("group_and_aggregate"), [_evidence("group_and_aggregate", 0, group_by="date")],
        date_columns=["date"], executed_tools=["group_and_aggregate"], recovery_finished=True)
    temporal = next(item for item in assessment.requirements if item.kind == "temporal")
    segment = next(item for item in assessment.requirements if item.kind == "segment")
    assert temporal.supported is True
    assert segment.supported is False
    assert segment.dimension == "region"


def test_full_tool_lifecycle_transitions_to_evidenced():
    assessment = assess_coverage(
        _question(), _plan("t_test", categories=["STATISTICS"]),
        [_evidence("t_test", 0, evidence_type="STATISTICAL_RESULT")],
        date_columns=["date"], executed_tools=["t_test"], recovery_finished=True)
    tool = assessment.tools[0]
    assert tool.stage == "evidenced"
    assert tool.transitions == ["planned", "selected", "executed", "evidenced"]
    assert tool.planned and tool.selected and tool.executed and tool.evidenced


def test_exact_missing_tool_is_the_only_recovery_target():
    assessment = assess_coverage(
        _question(), _plan("describe_data", "group_and_aggregate"), [_evidence("describe_data", 0)],
        date_columns=["date"], executed_tools=["describe_data"], recovery_finished=False)
    assert assessment.recovery_targets == ["group_and_aggregate"]
    assert [(item.tool_name, item.stage) for item in assessment.tools] == [
        ("describe_data", "evidenced"), ("group_and_aggregate", "selected")]


def test_missing_evidence_becomes_typed_unavailable_after_recovery():
    assessment = assess_coverage(
        _question(), _plan("group_and_aggregate"), [], date_columns=["date"],
        executed_tools=["group_and_aggregate"], recovery_finished=True)
    tool = assessment.tools[0]
    assert tool.stage == "unavailable"
    assert tool.unavailable and tool.executed
    assert tool.transitions == ["planned", "selected", "executed", "unavailable"]
    assert "one bounded recovery pass" in tool.reason


def _run_stubbed(monkeypatch, sales_record, *, initial_evidence, recovery_evidence, expected_targets):
    question = _question(dimensions=["region"], time_range="2024 vs 2025")
    plan = _plan("compare_periods", "group_and_aggregate")
    monkeypatch.setattr(orchestrator_module.question_parser, "parse_question", lambda *_: (question, []))
    monkeypatch.setattr(orchestrator_module, "validate_question",
                        lambda *_: ([], [], {"date_columns": ["date"]}))
    monkeypatch.setattr(orchestrator_module.planner, "plan_analysis", lambda *_: plan)
    monkeypatch.setattr(orchestrator_module.executor, "execute_plan",
                        lambda *_: (initial_evidence, "", [e.source_tool for e in initial_evidence]))
    recovery_calls = []

    def recover(_provider, _record, _text, missing_tools, _router, *, evidence_offset):
        recovery_calls.append(missing_tools.copy())
        assert evidence_offset == len(initial_evidence)
        return recovery_evidence, expected_targets.copy()

    monkeypatch.setattr(orchestrator_module.executor, "execute_recovery", recover)
    monkeypatch.setattr(orchestrator_module.verifier, "build_findings", lambda evidence: ([
        Finding(id=f"finding_{i}", statement=f"Evidence from {e.source_tool}",
                classification=e.evidence_type, supporting_evidence=[e.id])
        for i, e in enumerate(evidence)], []))
    monkeypatch.setattr(orchestrator_module.confound_detection, "detect_confounds", lambda *_: [])
    monkeypatch.setattr(orchestrator_module.contradiction_detection,
                        "detect_ranking_contradictions", lambda *_: [])
    monkeypatch.setattr(orchestrator_module.contradiction_detection,
                        "detect_overall_vs_subgroup_contradiction", lambda *_: [])
    monkeypatch.setattr(orchestrator_module.contradiction_detection,
                        "detect_data_quality_contradictions", lambda *_: [])
    monkeypatch.setattr(orchestrator_module.hypothesis_evaluator,
                        "update_hypothesis_status", lambda *_: [])
    monkeypatch.setattr(orchestrator_module.epistemic_checks, "check_all", lambda *_: [])
    monkeypatch.setattr(orchestrator_module, "synthesize", lambda *_: (
        "Deterministic result.", Recommendation(recommendation="Act on the supported driver."),
        False, []))
    instance = ReasoningOrchestrator(MockProvider([]))
    return instance.analyze(sales_record, question.original_question), instance, recovery_calls


def test_recovery_targets_exact_gap_and_removes_recommendation(monkeypatch, sales_record):
    result, instance, calls = _run_stubbed(
        monkeypatch, sales_record, initial_evidence=[_evidence("compare_periods", 0)],
        recovery_evidence=[], expected_targets=["group_and_aggregate"])
    assert calls == [["group_and_aggregate"]]
    assert instance.last_coverage.unresolved_tools == ["group_and_aggregate"]
    assert result.recommendation is None
    blocker = next(item for item in result.limitations if item.severity == "blocks_conclusion")
    assert "group_and_aggregate" in blocker.text
    assert "segment(region)" in blocker.text


def test_recommendation_is_preserved_when_all_coverage_is_evidenced(monkeypatch, sales_record):
    result, instance, calls = _run_stubbed(
        monkeypatch, sales_record, initial_evidence=[_evidence("compare_periods", 0)],
        recovery_evidence=[_evidence("group_and_aggregate", 1, group_by="region")],
        expected_targets=["group_and_aggregate"])
    assert calls == [["group_and_aggregate"]]
    assert instance.last_coverage.complete
    assert result.recommendation is not None
    assert not any(item.severity == "blocks_conclusion" for item in result.limitations)


def test_existing_non_diagnostic_flow_does_not_trigger_rca_recovery(monkeypatch, sales_record):
    question = AnalyticalQuestion(original_question="Summarize revenue", intent="descriptive")
    plan, evidence = _plan("describe_data"), [_evidence("describe_data", 0)]
    monkeypatch.setattr(orchestrator_module.question_parser, "parse_question", lambda *_: (question, []))
    monkeypatch.setattr(orchestrator_module, "validate_question",
                        lambda *_: ([], [], {"date_columns": ["date"]}))
    monkeypatch.setattr(orchestrator_module.planner, "plan_analysis", lambda *_: plan)
    monkeypatch.setattr(orchestrator_module.executor, "execute_plan",
                        lambda *_: (evidence, "", ["describe_data"]))
    monkeypatch.setattr(orchestrator_module.executor, "execute_recovery",
                        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("unexpected recovery")))
    monkeypatch.setattr(orchestrator_module.verifier, "build_findings", lambda *_: ([], []))
    monkeypatch.setattr(orchestrator_module.confound_detection, "detect_confounds", lambda *_: [])
    monkeypatch.setattr(orchestrator_module.contradiction_detection,
                        "detect_ranking_contradictions", lambda *_: [])
    monkeypatch.setattr(orchestrator_module.contradiction_detection,
                        "detect_overall_vs_subgroup_contradiction", lambda *_: [])
    monkeypatch.setattr(orchestrator_module.contradiction_detection,
                        "detect_data_quality_contradictions", lambda *_: [])
    monkeypatch.setattr(orchestrator_module.hypothesis_evaluator,
                        "update_hypothesis_status", lambda *_: [])
    monkeypatch.setattr(orchestrator_module.epistemic_checks, "check_all", lambda *_: [])
    monkeypatch.setattr(orchestrator_module, "synthesize", lambda *_: ("Summary.", None, False, []))
    result = ReasoningOrchestrator(MockProvider([])).analyze(sales_record, question.original_question)
    assert result.final_answer_text == "Summary."
    assert result.evidence == evidence
