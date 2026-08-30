"""Mix Decomposition Engine Phase 2: deterministic RCA coverage integration
tests for app.reasoning.coverage. No live provider is used anywhere in this
file. Independent, freshly-authored fixtures only.
"""

from __future__ import annotations

from app.agent.providers import MockProvider
from app.reasoning import orchestrator as orchestrator_module
from app.reasoning.conclusion_guard import enforce_conclusion_guard
from app.reasoning.contracts import AnalysisPlan, AnalyticalQuestion, Evidence, Finding, Recommendation
from app.reasoning.coverage import assess_coverage
from app.reasoning.orchestrator import ReasoningOrchestrator


def _question(text: str, *, dimensions=None, time_range=None):
    return AnalyticalQuestion(original_question=text, intent="diagnostic",
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


_MIX_QUESTION = "Did the change come from a shift in customer mix, or from a within-segment performance change?"
_MARKETING_MIX_QUESTION = "How did our marketing mix affect overall performance this quarter?"
_WITHIN_EACH_SEGMENT_QUESTION = "What happened within each segment last quarter?"


# --- 1. true-positive detection -------------------------------------------------------


def test_mix_and_within_segment_language_together_creates_a_required_requirement():
    assessment = assess_coverage(
        _question(_MIX_QUESTION), _plan(), [],
        date_columns=["date"], executed_tools=[], recovery_finished=False)
    mix_requirement = next((item for item in assessment.requirements if item.kind == "mix_decomposition"), None)
    assert mix_requirement is not None
    assert mix_requirement.supported is False
    assert mix_requirement.required_tools == ["mix_decomposition"]


def test_true_positive_also_appears_as_an_unresolved_requirement_when_unsupported():
    assessment = assess_coverage(
        _question(_MIX_QUESTION), _plan(), [],
        date_columns=["date"], executed_tools=[], recovery_finished=True)
    assert any(item.startswith("mix_decomposition") for item in assessment.unresolved_requirements)
    assert not assessment.complete


# --- 2. two false-positive guards ------------------------------------------------------


def test_marketing_mix_alone_does_not_trigger_the_requirement():
    assessment = assess_coverage(
        _question(_MARKETING_MIX_QUESTION), _plan(), [],
        date_columns=["date"], executed_tools=[], recovery_finished=False)
    assert not any(item.kind == "mix_decomposition" for item in assessment.requirements)


def test_within_each_segment_alone_does_not_trigger_the_requirement():
    assessment = assess_coverage(
        _question(_WITHIN_EACH_SEGMENT_QUESTION), _plan(), [],
        date_columns=["date"], executed_tools=[], recovery_finished=False)
    assert not any(item.kind == "mix_decomposition" for item in assessment.requirements)


def test_neither_false_positive_guard_leaves_other_requirement_kinds_broken():
    """Sanity check: the guards above are specific to mix_decomposition, not an
    accidental over-broad exclusion that also swallows segment detection."""
    assessment = assess_coverage(
        _question(_WITHIN_EACH_SEGMENT_QUESTION, dimensions=["region"]), _plan("group_and_aggregate"),
        [_evidence("group_and_aggregate", 0, group_by="region")],
        date_columns=["date"], executed_tools=["group_and_aggregate"], recovery_finished=True)
    segment_requirement = next(item for item in assessment.requirements if item.kind == "segment")
    assert segment_requirement.supported is True


# --- 3. tool evidence satisfies the requirement ----------------------------------------


def test_evidenced_mix_decomposition_call_satisfies_the_requirement():
    assessment = assess_coverage(
        _question(_MIX_QUESTION), _plan("mix_decomposition"),
        [_evidence("mix_decomposition", 0)],
        date_columns=["date"], executed_tools=["mix_decomposition"], recovery_finished=True)
    mix_requirement = next(item for item in assessment.requirements if item.kind == "mix_decomposition")
    assert mix_requirement.supported is True
    assert mix_requirement.supporting_evidence == ["ev_0"]
    assert assessment.complete


# --- 4. other tools do NOT satisfy it ---------------------------------------------------


def test_group_and_aggregate_does_not_satisfy_mix_decomposition_requirement():
    assessment = assess_coverage(
        _question(_MIX_QUESTION), _plan("group_and_aggregate"),
        [_evidence("group_and_aggregate", 0, group_by="region")],
        date_columns=["date"], executed_tools=["group_and_aggregate"], recovery_finished=True)
    mix_requirement = next(item for item in assessment.requirements if item.kind == "mix_decomposition")
    assert mix_requirement.supported is False


def test_contribution_analysis_does_not_satisfy_mix_decomposition_requirement():
    assessment = assess_coverage(
        _question(_MIX_QUESTION), _plan("contribution_analysis"),
        [_evidence("contribution_analysis", 0)],
        date_columns=["date"], executed_tools=["contribution_analysis"], recovery_finished=True)
    mix_requirement = next(item for item in assessment.requirements if item.kind == "mix_decomposition")
    assert mix_requirement.supported is False


def test_generate_chart_does_not_satisfy_mix_decomposition_requirement():
    assessment = assess_coverage(
        _question(_MIX_QUESTION), _plan("generate_chart"),
        [_evidence("generate_chart", 0)],
        date_columns=["date"], executed_tools=["generate_chart"], recovery_finished=True)
    mix_requirement = next(item for item in assessment.requirements if item.kind == "mix_decomposition")
    assert mix_requirement.supported is False


def test_temporal_grouping_does_not_satisfy_mix_decomposition_requirement():
    assessment = assess_coverage(
        _question(_MIX_QUESTION), _plan("compare_periods"),
        [_evidence("compare_periods", 0)],
        date_columns=["date"], executed_tools=["compare_periods"], recovery_finished=True)
    mix_requirement = next(item for item in assessment.requirements if item.kind == "mix_decomposition")
    assert mix_requirement.supported is False


# --- deterministically required even if the planner never listed it -------------------


def test_mix_decomposition_is_required_even_when_not_in_plan_tools_required():
    """Zero-trust in the planner: the requirement is derived from the QUESTION,
    not from whatever the planner happened to list."""
    assessment = assess_coverage(
        _question(_MIX_QUESTION), _plan("describe_data"), [_evidence("describe_data", 0)],
        date_columns=["date"], executed_tools=["describe_data"], recovery_finished=False)
    assert "mix_decomposition" in assessment.recovery_targets
    tool_names = {item.tool_name for item in assessment.tools}
    assert "mix_decomposition" in tool_names


# --- 5. bounded recovery targets exactly mix_decomposition + 6/7 orchestrator flow -----


def _run_stubbed(monkeypatch, sales_record, *, question_text, plan, initial_evidence, recovery_evidence, expected_targets):
    question = _question(question_text)
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

    def _stub_synthesize(_provider, _question, _claims, _plan, _evidence, findings, _hypotheses, limitations):
        # Mirrors the real synthesizer.synthesize()'s own blocking/guard logic
        # (minus the LLM call itself), so this integration test genuinely
        # exercises requirement #5: the EXISTING, generic conclusion guard must
        # still catch unsafe definitive-driver/recommendation prose for this new
        # insufficient_coverage/mix_decomposition limitation, with zero changes
        # to conclusion_guard.py.
        blocking = any(item.severity == "blocks_conclusion" for item in limitations)
        raw_text = "The Basic tier's average order value increased, and a primary driver was pricing."
        recommendation = None if blocking else Recommendation(recommendation="Act on the supported driver.")
        caveated_text, _caveat_added = enforce_conclusion_guard(raw_text, limitations, findings)
        return caveated_text, recommendation, False, []

    monkeypatch.setattr(orchestrator_module, "synthesize", _stub_synthesize)
    instance = ReasoningOrchestrator(MockProvider([]))
    return instance.analyze(sales_record, question.original_question), instance, recovery_calls


def test_bounded_recovery_targets_exactly_mix_decomposition(monkeypatch, sales_record):
    plan = _plan("describe_data")
    result, instance, calls = _run_stubbed(
        monkeypatch, sales_record, question_text=_MIX_QUESTION, plan=plan,
        initial_evidence=[_evidence("describe_data", 0)],
        recovery_evidence=[], expected_targets=["mix_decomposition"])
    assert calls == [["mix_decomposition"]]
    assert instance.last_coverage.unresolved_tools == ["mix_decomposition"]


# --- 6. unresolved requirement blocks conclusion ---------------------------------------


def test_unresolved_mix_requirement_blocks_conclusion_and_withholds_recommendation(monkeypatch, sales_record):
    plan = _plan("describe_data")
    result, instance, calls = _run_stubbed(
        monkeypatch, sales_record, question_text=_MIX_QUESTION, plan=plan,
        initial_evidence=[_evidence("describe_data", 0)],
        recovery_evidence=[], expected_targets=["mix_decomposition"])
    assert not instance.last_coverage.complete
    assert result.recommendation is None
    blocker = next(item for item in result.limitations if item.severity == "blocks_conclusion")
    assert "mix_decomposition" in blocker.text
    # the existing conclusion guard must still fire on unsafe definitive-driver /
    # recommendation prose -- the stubbed synthesis text above deliberately
    # contains both "a primary driver" and "act on the supported driver"
    assert "primary driver" not in result.final_answer_text
    assert "cannot be made from the available evidence" in result.final_answer_text


# --- 7. fully evidenced flow remains allowed --------------------------------------------


def test_fully_evidenced_mix_decomposition_flow_allows_recommendation(monkeypatch, sales_record):
    plan = _plan("describe_data")
    result, instance, calls = _run_stubbed(
        monkeypatch, sales_record, question_text=_MIX_QUESTION, plan=plan,
        initial_evidence=[_evidence("describe_data", 0)],
        recovery_evidence=[_evidence("mix_decomposition", 1)],
        expected_targets=["mix_decomposition"])
    assert calls == [["mix_decomposition"]]
    assert instance.last_coverage.complete
    assert result.recommendation is not None
    assert not any(item.severity == "blocks_conclusion" for item in result.limitations)
