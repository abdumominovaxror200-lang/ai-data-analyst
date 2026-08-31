"""Generic Phase A regressions; no benchmark fixtures or provider calls."""

from __future__ import annotations

from app.reasoning.conclusion_guard import blocked_narrative_violations, sanitize_blocked_hypotheses
from app.reasoning.contracts import (
    AnalysisPlan,
    AnalyticalQuestion,
    Evidence,
    EvidenceScope,
    Hypothesis,
    Limitation,
    TemporalEvidenceScope,
)
from app.reasoning.coverage import assess_coverage
from app.reasoning.executor import _to_evidence
from app.reasoning.hypothesis_evaluator import update_hypothesis_status
from app.reasoning.verifier import build_findings
from app.schemas import ToolCallRecord


def _plan(*tools: str) -> AnalysisPlan:
    return AnalysisPlan(
        objective="Compare two calendar periods and explain the change",
        capability_categories=["GENERAL_ANALYSIS", "STATISTICS"],
        tools_required=list(tools),
    )


def _evidence(identifier: str, tool: str, *, scope: EvidenceScope | None = None, mean: float | None = None) -> Evidence:
    summary = {} if mean is None else {"mean": mean}
    return Evidence(
        id=identifier,
        source_tool=tool,
        evidence_type="STATISTICAL_RESULT" if tool in {"confidence_interval", "t_test"} else "CALCULATED_RESULT",
        metric="response_time",
        result_summary=summary,
        scope=scope,
        tool_call_ref=f"tool_call[{identifier}]",
    )


def test_numeric_variables_never_create_segment_obligations():
    question = AnalyticalQuestion(
        original_question="Explain the change",
        intent="diagnostic",
        requested_dimensions=["backlog", "cost", "region"],
    )
    result = assess_coverage(
        question, _plan(), [], date_columns=["date"],
        categorical_columns=["region", "channel"], executed_tools=[], recovery_finished=True,
    )
    segments = [item.dimension for item in result.requirements if item.kind == "segment"]
    assert segments == ["region"]


def test_true_categorical_dimension_still_requires_segment_evidence():
    question = AnalyticalQuestion(original_question="Explain by channel", intent="diagnostic", requested_dimensions=["channel"])
    result = assess_coverage(
        question, _plan(), [], date_columns=["date"], categorical_columns=["channel"],
        executed_tools=[], recovery_finished=True,
    )
    requirement = next(item for item in result.requirements if item.kind == "segment")
    assert requirement.dimension == "channel" and requirement.supported is False


def test_h2_evidence_cannot_satisfy_a_different_period_requirement():
    requested = AnalyticalQuestion(
        original_question="Compare H2 2025 with H2 2024", intent="diagnostic",
        requested_time_range="H2 2025 versus H2 2024",
    )
    wrong_scope = EvidenceScope(temporal=TemporalEvidenceScope(
        current_start="2025-01-01", current_end="2025-06-30",
        previous_start="2024-01-01", previous_end="2024-06-30",
    ))
    result = assess_coverage(
        requested, _plan("compare_periods"), [_evidence("ev_1", "compare_periods", scope=wrong_scope)],
        date_columns=["date"], categorical_columns=[], executed_tools=["compare_periods"], recovery_finished=True,
    )
    temporal = next(item for item in result.requirements if item.kind == "temporal")
    assert temporal.supported is False


def test_exact_two_period_scope_satisfies_temporal_requirement():
    question = AnalyticalQuestion(
        original_question="Compare H2 periods", intent="diagnostic",
        requested_time_range="H2 2025 versus H2 2024",
    )
    exact = EvidenceScope(temporal=TemporalEvidenceScope(
        current_start="2025-07-01", current_end="2025-12-31",
        previous_start="2024-07-01", previous_end="2024-12-31",
    ))
    result = assess_coverage(
        question, _plan("compare_periods"), [_evidence("ev_1", "compare_periods", scope=exact)],
        date_columns=["date"], categorical_columns=[], executed_tools=["compare_periods"], recovery_finished=True,
    )
    assert next(item for item in result.requirements if item.kind == "temporal").supported is True


def test_mismatched_ci_and_test_scopes_are_not_cross_checked():
    first = EvidenceScope(filters=[{"column": "period", "op": "==", "value": "A"}])
    second = EvidenceScope(filters=[{"column": "period", "op": "==", "value": "B"}])
    findings, limitations = build_findings([
        _evidence("ev_1", "confidence_interval", scope=first, mean=10.0),
        _evidence("ev_2", "t_test", scope=second, mean=10.0),
    ])
    assert not any(item.cross_checked for item in findings)
    assert not any("disagree" in item.text.lower() for item in limitations)


def test_executor_records_typed_filter_population_and_period_scope():
    call = ToolCallRecord(
        tool="compare_periods",
        params={
            "current_start": "2026-01-01", "current_end": "2026-03-31",
            "previous_start": "2025-01-01", "previous_end": "2025-03-31",
            "filters": [{"column": "region", "op": "==", "value": "West"}],
        },
        result={"current_value": 12.0, "previous_value": 10.0},
    )
    evidence = _to_evidence(0, call)
    assert evidence.scope is not None
    assert evidence.scope.population == "region == West"
    assert evidence.scope.filters == [{"column": "region", "op": "==", "value": "West"}]
    assert evidence.scope.temporal.current_start == "2026-01-01"
    assert evidence.scope.temporal.previous_end == "2025-03-31"


def test_executor_records_comparison_groups_and_dual_filter_scopes():
    call = ToolCallRecord(
        tool="contribution_analysis",
        params={
            "group_column": "channel", "group_a": "Direct", "group_b": "Partner",
            "current_filters": [{"column": "year", "op": "==", "value": 2026}],
            "baseline_filters": [{"column": "year", "op": "==", "value": 2025}],
        },
        result={"groups": []},
    )
    scope = _to_evidence(0, call).scope
    assert scope.comparison_groups == {"group_column": "channel", "group_a": "Direct", "group_b": "Partner"}
    assert scope.current_filters[0]["value"] == 2026
    assert scope.previous_filters[0]["value"] == 2025


def test_blocks_conclusion_sanitizes_every_hypothesis_status_and_text():
    hypotheses = [Hypothesis(
        id="h1", description="The primary driver was caused by staffing changes.",
        is_causal=True, status="supported", evidence_for=["ev_1"],
    )]
    blocker = [Limitation(category="insufficient_coverage", text="Required evidence is missing.", severity="blocks_conclusion")]
    sanitized = sanitize_blocked_hypotheses(hypotheses, blocker)
    assert sanitized[0].status == "inconclusive"
    assert sanitized[0].is_causal is False
    assert blocked_narrative_violations(sanitized[0].description) == []
    assert "additional evidence" in sanitized[0].description.lower()


def test_nonblocked_hypotheses_remain_byte_for_byte_unchanged():
    hypotheses = [Hypothesis(id="h1", description="A supported descriptive pattern.", is_causal=False, status="supported")]
    assert sanitize_blocked_hypotheses(hypotheses, []) == hypotheses


def test_one_sample_evidence_cannot_satisfy_explicit_two_period_inference():
    question = AnalyticalQuestion(
        original_question="Compare uncertainty between H2 2026 and H2 2025",
        intent="diagnostic", requested_time_range="H2 2026 versus H2 2025",
        required_confidence="95% confidence interval",
    )
    exact = EvidenceScope(temporal=TemporalEvidenceScope(
        current_start="2026-07-01", current_end="2026-12-31",
        previous_start="2025-07-01", previous_end="2025-12-31",
    ))
    result = assess_coverage(
        question, _plan("t_test"), [_evidence("ev_1", "t_test", scope=exact)],
        date_columns=["date"], categorical_columns=[], executed_tools=["t_test"], recovery_finished=True,
    )
    requirement = next(item for item in result.requirements if item.kind == "two_period_inference")
    assert requirement.supported is False
    assert "compare_periods_inference" in result.unresolved_tools


def test_exact_scope_phase_b_evidence_satisfies_only_matching_request():
    question = AnalyticalQuestion(
        original_question="Test uncertainty and robustness by segment in H2 2026 versus H2 2025",
        intent="diagnostic", requested_time_range="H2 2026 versus H2 2025",
        required_confidence="95%", requested_dimensions=["channel"],
    )
    exact = EvidenceScope(temporal=TemporalEvidenceScope(
        current_start="2026-07-01", current_end="2026-12-31",
        previous_start="2025-07-01", previous_end="2025-12-31",
    ))
    evidence = _evidence("ev_1", "compare_periods_inference", scope=exact)
    evidence.evidence_type = "STATISTICAL_RESULT"
    result = assess_coverage(
        question, _plan("compare_periods_inference"), [evidence], date_columns=["date"],
        categorical_columns=["channel"], executed_tools=["compare_periods_inference"], recovery_finished=True,
    )
    assert next(item for item in result.requirements if item.kind == "two_period_inference").supported


def test_two_period_evidence_with_mismatched_population_filter_is_rejected():
    question = AnalyticalQuestion(
        original_question="Compare uncertainty for region North in H2 2026 versus H2 2025",
        intent="diagnostic", requested_time_range="H2 2026 versus H2 2025",
        requested_population="region == North", required_confidence="95%",
    )
    wrong = EvidenceScope(
        population="region == South",
        filters=[{"column": "region", "op": "==", "value": "South"}],
        temporal=TemporalEvidenceScope(
            current_start="2026-07-01", current_end="2026-12-31",
            previous_start="2025-07-01", previous_end="2025-12-31",
        ),
    )
    evidence = _evidence("ev_1", "compare_periods_inference", scope=wrong)
    evidence.evidence_type = "STATISTICAL_RESULT"
    result = assess_coverage(
        question, _plan("compare_periods_inference"), [evidence], date_columns=["date"],
        categorical_columns=["region"], executed_tools=["compare_periods_inference"], recovery_finished=True,
    )
    assert not next(item for item in result.requirements if item.kind == "two_period_inference").supported


def test_post_outcome_field_is_descriptive_and_cannot_support_causal_hypothesis():
    call = ToolCallRecord(
        tool="correlation_analysis",
        params={"columns": ["duration", "post_resolution_rating"]},
        result={"correlations": [{"column_a": "duration", "column_b": "post_resolution_rating", "correlation": .9}]},
    )
    evidence = _to_evidence(0, call)
    assert evidence.causal_eligible is False
    hypothesis = Hypothesis(
        id="h1", description="Post resolution rating caused duration changes.", is_causal=True,
    )
    [updated] = update_hypothesis_status([hypothesis], [evidence], [])
    assert updated.status == "untested"
    assert updated.evidence_for == []
