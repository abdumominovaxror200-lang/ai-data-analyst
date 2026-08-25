from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.reasoning.contracts import (
    AnalysisPlan,
    AnalysisResult,
    AnalyticalQuestion,
    Claim,
    Evidence,
    Finding,
    Hypothesis,
    Limitation,
    Recommendation,
    Uncertainty,
)


def test_analytical_question_requires_a_valid_intent():
    AnalyticalQuestion(original_question="q", intent="diagnostic")
    with pytest.raises(ValidationError):
        AnalyticalQuestion(original_question="q", intent="not_a_real_intent")


def test_claim_source_is_constrained_to_user_or_system():
    Claim(text="x", source="user_asserted")
    Claim(text="x", source="system_inferred")
    with pytest.raises(ValidationError):
        Claim(text="x", source="made_up_source")


def test_hypothesis_is_causal_is_a_required_explicit_field():
    with pytest.raises(ValidationError):
        Hypothesis(id="h1", description="X causes Y", status="supported")  # missing is_causal
    h = Hypothesis(id="h1", description="X causes Y", is_causal=True, status="supported")
    assert h.is_causal is True


def test_hypothesis_status_has_five_way_scale():
    for status in ("untested", "supported", "weakly_supported", "unsupported", "contradicted"):
        Hypothesis(id="h", description="d", is_causal=False, status=status)
    with pytest.raises(ValidationError):
        Hypothesis(id="h", description="d", is_causal=False, status="probably")


def test_finding_classification_is_exactly_the_six_way_enum():
    valid = {"FACT", "CALCULATED_RESULT", "STATISTICAL_RESULT", "HYPOTHESIS", "ASSUMPTION", "UNKNOWN"}
    for classification in valid:
        Finding(id="f1", statement="s", classification=classification)
    with pytest.raises(ValidationError):
        Finding(id="f1", statement="s", classification="MAYBE_TRUE")


def test_uncertainty_level_is_the_four_way_categorical_scale():
    for level in ("known", "estimated", "uncertain", "unavailable"):
        Uncertainty(level=level)
    with pytest.raises(ValidationError):
        Uncertainty(level="very_sure")


def test_limitation_has_seven_categories():
    valid = {
        "missing_data",
        "insufficient_coverage",
        "sample_size",
        "unavailable_capability",
        "methodological",
        "resource_limit",
        "other",
    }
    for category in valid:
        Limitation(category=category, text="x")
    with pytest.raises(ValidationError):
        Limitation(category="not_a_category", text="x")


def test_recommendation_confidence_is_nullable_and_never_forced():
    rec = Recommendation(recommendation="Do X")
    assert rec.confidence is None  # evidence didn't justify a confidence claim -- must be a valid state
    rec2 = Recommendation(recommendation="Do X", confidence="high")
    assert rec2.confidence == "high"
    with pytest.raises(ValidationError):
        Recommendation(recommendation="Do X", confidence="extremely_certain")


def test_analysis_plan_has_all_required_fields_with_safe_defaults():
    plan = AnalysisPlan(objective="test")
    assert plan.steps == []
    assert plan.tools_required == []
    assert plan.hypotheses == []


def test_analysis_result_is_the_full_traceable_pipeline_output():
    question = AnalyticalQuestion(original_question="q", intent="descriptive")
    evidence = Evidence(id="ev_0", source_tool="describe_data", evidence_type="CALCULATED_RESULT", tool_call_ref="tool_call[0]")
    finding = Finding(id="f0", statement="s", classification="CALCULATED_RESULT", supporting_evidence=["ev_0"])
    result = AnalysisResult(
        question=question,
        evidence=[evidence],
        findings=[finding],
        final_answer_text="answer",
    )
    assert result.findings[0].supporting_evidence[0] == result.evidence[0].id
