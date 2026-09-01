from __future__ import annotations

from app.agent.providers import MockProvider, ProviderResponse, ToolCall
from app.reasoning.contracts import (
    AnalysisPlan,
    AnalyticalQuestion,
    Evidence,
    EvidenceScope,
    TemporalEvidenceScope,
)
from app.reasoning.inferential_safety import assess_multiple_comparisons, insufficient_data_limitation
from app.reasoning.orchestrator import ReasoningOrchestrator
from tests.reasoning.conftest import (
    json_response,
    no_more_tools_response,
    parsed_question_payload,
    plan_payload,
    synthesis_payload,
    tool_call_response,
)


def _question(text="Estimate statistical uncertainty"):
    return AnalyticalQuestion(original_question=text, intent="comparative")


def _plan():
    return AnalysisPlan(objective="Estimate statistical uncertainty", capability_categories=["STATISTICS"])


def _evidence(identifier, summary, *, scope=None, evidence_type="STATISTICAL_RESULT"):
    return Evidence(
        id=identifier,
        source_tool="t_test",
        evidence_type=evidence_type,
        metric="cycle_time",
        result_summary=summary,
        scope=scope,
        sample_size=40,
        tool_call_ref=f"tool_call[{identifier}]",
    )


def test_two_unadjusted_tests_create_typed_multiple_comparisons_warning():
    assessment = assess_multiple_comparisons([
        _evidence("e1", {"p_value": 0.02, "alpha": 0.05}),
        _evidence("e2", {"p_value": 0.04, "alpha": 0.05}),
    ])
    assert assessment.comparison_count == 2
    assert assessment.adjusted is False
    assert assessment.limitation is not None
    assert assessment.limitation.category == "multiple_comparisons"
    assert assessment.limitation.severity == "reduces_confidence"
    assert "Holm" in assessment.limitation.text


def test_single_test_does_not_create_multiplicity_warning():
    assessment = assess_multiple_comparisons([_evidence("e1", {"p_value": 0.02})])
    assert assessment.comparison_count == 1
    assert assessment.limitation is None


def test_documented_adjustment_prevents_false_warning():
    assessment = assess_multiple_comparisons([
        _evidence("e1", {
            "p_value": 0.02,
            "adjusted_p_values": [0.04, 0.08],
            "multiple_comparison_adjustment": "holm",
        }),
        _evidence("e2", {
            "p_value": 0.04,
            "adjusted_p_values": [0.04, 0.08],
            "multiple_comparison_adjustment": "holm",
        }),
    ])
    assert assessment.adjusted is True
    assert assessment.limitation is None


def test_regression_coefficient_p_values_are_counted_but_diagnostics_are_not():
    result = {
        "coefficients": {
            "volume": {"p_value": 0.01},
            "tenure": {"p_value": 0.03},
        },
        "residual_normality": {"p_value": 0.2},
        "heteroscedasticity": {"p_value": 0.4},
    }
    assessment = assess_multiple_comparisons([_evidence("e1", result)])
    assert assessment.comparison_count == 2
    assert assessment.limitation is not None


def test_no_evidence_always_returns_typed_blocking_refusal():
    limitation = insufficient_data_limitation(_question("Summarize the metric"), _plan(), [])
    assert limitation is not None
    assert limitation.category == "missing_data"
    assert limitation.severity == "blocks_conclusion"
    assert limitation.text.startswith("Insufficient data:")


def test_descriptive_evidence_remains_sufficient_for_non_inferential_question():
    question = AnalyticalQuestion(original_question="Summarize cycle time", intent="descriptive")
    plan = AnalysisPlan(objective="Summarize cycle time", capability_categories=["GENERAL_ANALYSIS"])
    evidence = [_evidence("e1", {"mean": 12.0}, evidence_type="CALCULATED_RESULT")]
    assert insufficient_data_limitation(question, plan, evidence) is None


def test_descriptive_evidence_cannot_satisfy_explicit_statistical_request():
    evidence = [_evidence("e1", {"mean": 12.0}, evidence_type="CALCULATED_RESULT")]
    limitation = insufficient_data_limitation(_question(), _plan(), evidence)
    assert limitation is not None and limitation.severity == "blocks_conclusion"


def test_unscoped_test_cannot_satisfy_requested_two_period_inference():
    question = AnalyticalQuestion(
        original_question="Compare statistical uncertainty in 2025 versus 2024",
        intent="comparative",
        requested_time_range="2025 versus 2024",
    )
    limitation = insufficient_data_limitation(question, _plan(), [_evidence("e1", {"p_value": 0.01})])
    assert limitation is not None
    assert "two-period scope" in limitation.text


def test_exact_two_period_scope_satisfies_requested_inference():
    question = AnalyticalQuestion(
        original_question="Compare statistical uncertainty in 2025 versus 2024",
        intent="comparative",
        requested_time_range="2025 versus 2024",
    )
    scope = EvidenceScope(temporal=TemporalEvidenceScope(
        current_start="2025-01-01", current_end="2025-12-31",
        previous_start="2024-01-01", previous_end="2024-12-31",
    ))
    assert insufficient_data_limitation(question, _plan(), [_evidence("e1", {"p_value": 0.01}, scope=scope)]) is None


def test_orchestrator_refuses_inferential_conclusion_when_only_descriptive_evidence_exists(sales_record):
    parsed = parsed_question_payload(intent="comparative", requested_metrics=["revenue"])
    parsed["required_confidence"] = "95%"
    script = [
        json_response(parsed),
        json_response(plan_payload(
            capability_categories=["GENERAL_ANALYSIS"], tools_required=["describe_data"]
        )),
        tool_call_response("describe_data", {"columns": ["revenue"]}),
        no_more_tools_response(),
        json_response(synthesis_payload("Revenue is definitively higher, so prioritize expansion.")),
    ]
    result = ReasoningOrchestrator(MockProvider(script)).analyze(
        sales_record, "Is revenue statistically higher with 95% confidence?"
    )
    blocker = next(item for item in result.limitations if item.category == "insufficient_data")
    assert blocker.severity == "blocks_conclusion"
    assert result.recommendation is None
    assert all(item.status == "inconclusive" for item in result.hypotheses)
    assert "A definitive conclusion cannot be made" in result.final_answer_text
    assert "prioritize expansion" not in result.final_answer_text.lower()


def test_orchestrator_surfaces_multiple_comparisons_warning(sales_record):
    parsed = parsed_question_payload(intent="comparative", requested_metrics=["revenue", "quantity"])
    parsed["required_confidence"] = "95%"
    script = [
        json_response(parsed),
        json_response(plan_payload(capability_categories=["STATISTICS"], tools_required=["t_test"])),
        ProviderResponse(content=None, tool_calls=[
            ToolCall(id="revenue-test", name="t_test", arguments={"column": "revenue", "popmean": 900}),
            ToolCall(id="quantity-test", name="t_test", arguments={"column": "quantity", "popmean": 20}),
        ]),
        no_more_tools_response(),
        json_response(synthesis_payload("The two exploratory tests are summarized.")),
    ]
    result = ReasoningOrchestrator(MockProvider(script)).analyze(
        sales_record, "Test revenue and quantity with statistical confidence."
    )
    warning = next(item for item in result.limitations if item.category == "multiple_comparisons")
    assert warning.severity == "reduces_confidence"
    assert "2 unadjusted inferential comparisons" in warning.text
    assert "Statistical caution:" in result.final_answer_text
    assert warning.text in result.final_answer_text
    assert not any(item.severity == "blocks_conclusion" for item in result.limitations)
    assert any("multiple-comparisons warning" in step for step in result.reasoning_trace)
