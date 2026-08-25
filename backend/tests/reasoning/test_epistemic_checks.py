"""Tests for `app.reasoning.epistemic_checks` (Phase 4 P2).

For each of the 10 checks: at least one test proving it flags a genuine violation
(hand-constructed minimal contract objects) and at least one test proving it stays
silent on a compliant case. Plus aggregation tests for `check_all`, including safety
on a maximally-empty/default input.
"""

from __future__ import annotations

from app.reasoning import epistemic_checks as ec
from app.reasoning.contracts import (
    AnalyticalQuestion,
    Claim,
    Evidence,
    Finding,
    Hypothesis,
    Limitation,
    Recommendation,
    Uncertainty,
)


# --- 1. observation vs. interpretation -------------------------------------------


def test_observation_without_evidence_is_flagged():
    findings = [
        Finding(id="f0", statement="Revenue is $50,000.", classification="FACT", supporting_evidence=[]),
    ]
    violations = ec.check_observation_vs_interpretation(findings)
    assert len(violations) == 1
    assert "f0" in violations[0]


def test_observation_with_evidence_is_silent():
    findings = [
        Finding(id="f0", statement="Revenue is $50,000.", classification="FACT", supporting_evidence=["ev_0"]),
    ]
    assert ec.check_observation_vs_interpretation(findings) == []


def test_observation_check_safe_on_none_and_empty():
    assert ec.check_observation_vs_interpretation(None) == []
    assert ec.check_observation_vs_interpretation([]) == []


# --- 2. correlation vs. causation --------------------------------------------------


def test_unhedged_causal_text_without_justifying_hypothesis_is_flagged():
    text = "The marketing campaign caused the revenue increase."
    violations = ec.check_correlation_vs_causation(text, [])
    assert len(violations) == 1
    assert "causal" in violations[0]


def test_causal_text_with_justifying_hypothesis_is_silent():
    text = "The marketing campaign caused the revenue increase."
    hyps = [
        Hypothesis(id="h0", description="Campaign caused revenue increase", is_causal=True, status="supported")
    ]
    assert ec.check_correlation_vs_causation(text, hyps) == []


def test_correlation_check_silent_when_no_causal_phrases_present():
    text = "Revenue increased by 5% and is associated with the campaign."
    assert ec.check_correlation_vs_causation(text, []) == []


def test_correlation_check_safe_on_none():
    assert ec.check_correlation_vs_causation(None, None) == []


# --- 3. evidence vs. assumption ----------------------------------------------------


def test_assumption_finding_cited_undisclosed_is_flagged():
    findings = [
        Finding(id="f0", statement="Assume seasonality is constant.", classification="ASSUMPTION"),
    ]
    rec = Recommendation(recommendation="Do X", supporting_findings=["f0"], assumptions=[])
    violations = ec.check_evidence_vs_assumption(findings, rec)
    assert len(violations) == 1
    assert "f0" in violations[0]


def test_assumption_finding_cited_and_disclosed_is_silent():
    findings = [
        Finding(id="f0", statement="Assume seasonality is constant.", classification="ASSUMPTION"),
    ]
    rec = Recommendation(
        recommendation="Do X",
        supporting_findings=["f0"],
        assumptions=["Assume seasonality is constant."],
    )
    assert ec.check_evidence_vs_assumption(findings, rec) == []


def test_evidence_vs_assumption_safe_on_none_recommendation():
    assert ec.check_evidence_vs_assumption([], None) == []
    assert ec.check_evidence_vs_assumption(None, None) == []


# --- 4. uncertainty acknowledged ---------------------------------------------------


def test_statistical_result_without_uncertainty_is_flagged():
    findings = [
        Finding(id="f0", statement="p < 0.05", classification="STATISTICAL_RESULT", uncertainty=None),
    ]
    violations = ec.check_uncertainty_acknowledged(findings)
    assert len(violations) == 1
    assert "f0" in violations[0]


def test_statistical_result_with_uncertainty_is_silent():
    findings = [
        Finding(
            id="f0",
            statement="p < 0.05",
            classification="STATISTICAL_RESULT",
            uncertainty=Uncertainty(level="known", point_estimate=0.03),
        ),
    ]
    assert ec.check_uncertainty_acknowledged(findings) == []


def test_uncertainty_check_safe_on_none():
    assert ec.check_uncertainty_acknowledged(None) == []


# --- 5. no inference beyond data ---------------------------------------------------


def test_invented_number_not_in_evidence_is_flagged():
    rec = Recommendation(
        recommendation="Do X",
        expected_business_effect="This should increase revenue by 37%.",
    )
    evidence = [
        Evidence(
            id="ev_0",
            source_tool="describe_data",
            evidence_type="CALCULATED_RESULT",
            result_summary={"mean": 100.0},
            tool_call_ref="tool_call[0]",
        )
    ]
    violations = ec.check_no_inference_beyond_data(rec, evidence)
    assert len(violations) == 1
    assert "37" in violations[0]


def test_number_grounded_in_evidence_is_silent():
    rec = Recommendation(
        recommendation="Do X",
        expected_business_effect="This should increase revenue by 37%.",
    )
    evidence = [
        Evidence(
            id="ev_0",
            source_tool="describe_data",
            evidence_type="CALCULATED_RESULT",
            result_summary={"pct_change": 37.0},
            tool_call_ref="tool_call[0]",
        )
    ]
    assert ec.check_no_inference_beyond_data(rec, evidence) == []


def test_no_inference_beyond_data_safe_on_none():
    assert ec.check_no_inference_beyond_data(None, None) == []
    assert ec.check_no_inference_beyond_data(Recommendation(recommendation="Do X"), None) == []


# --- 6. alternative explanations considered -----------------------------------------


def test_diagnostic_question_with_fewer_than_two_hypotheses_is_flagged():
    violations = ec.check_alternative_explanations_considered(
        "diagnostic", [Hypothesis(id="h0", description="d", is_causal=False)]
    )
    assert len(violations) == 1


def test_diagnostic_question_with_two_hypotheses_is_silent():
    hyps = [
        Hypothesis(id="h0", description="d0", is_causal=False),
        Hypothesis(id="h1", description="d1", is_causal=False),
    ]
    assert ec.check_alternative_explanations_considered("diagnostic", hyps) == []


def test_non_diagnostic_question_is_never_flagged_regardless_of_hypothesis_count():
    assert ec.check_alternative_explanations_considered("descriptive", []) == []


def test_alternative_explanations_check_safe_on_none():
    assert ec.check_alternative_explanations_considered(None, None) == []


# --- 7. falsifiability / internal consistency ---------------------------------------


def test_supported_hypothesis_with_contradicting_evidence_is_flagged():
    hyps = [
        Hypothesis(
            id="h0",
            description="X causes Y",
            is_causal=True,
            status="supported",
            evidence_against=["ev_3"],
        )
    ]
    violations = ec.check_falsifiability(hyps)
    assert len(violations) == 1
    assert "h0" in violations[0]


def test_supported_hypothesis_without_contradicting_evidence_is_silent():
    hyps = [
        Hypothesis(id="h0", description="X causes Y", is_causal=True, status="supported", evidence_against=[])
    ]
    assert ec.check_falsifiability(hyps) == []


def test_falsifiability_check_safe_on_none():
    assert ec.check_falsifiability(None) == []


# --- 8. missing evidence identified ---------------------------------------------------


def test_falsified_claim_with_no_limitations_is_flagged():
    question = AnalyticalQuestion(original_question="q", intent="descriptive")
    claims = [Claim(text="Column 'conversion_rate' exists", source="system_inferred", status="verified_false")]
    violations = ec.check_missing_evidence_identified(question, [], claims)
    assert len(violations) == 1


def test_falsified_claim_with_a_limitation_present_is_silent():
    question = AnalyticalQuestion(original_question="q", intent="descriptive")
    claims = [Claim(text="Column 'conversion_rate' exists", source="system_inferred", status="verified_false")]
    limitations = [Limitation(category="missing_data", text="conversion_rate is not a column.")]
    assert ec.check_missing_evidence_identified(question, limitations, claims) == []


def test_missing_evidence_check_safe_on_none():
    assert ec.check_missing_evidence_identified(None, None, None) == []


# --- 9. no overconfidence -------------------------------------------------------------


def test_high_confidence_with_one_supporting_finding_is_flagged():
    rec = Recommendation(recommendation="Do X", confidence="high", supporting_findings=["f0"])
    violations = ec.check_no_overconfidence(rec)
    assert len(violations) == 1


def test_high_confidence_with_two_supporting_findings_is_silent():
    rec = Recommendation(recommendation="Do X", confidence="high", supporting_findings=["f0", "f1"])
    assert ec.check_no_overconfidence(rec) == []


def test_no_overconfidence_check_safe_on_none():
    assert ec.check_no_overconfidence(None) == []


# --- 10. no manufactured certainty ----------------------------------------------------


def test_known_uncertainty_without_point_estimate_is_flagged():
    findings = [
        Finding(
            id="f0",
            statement="The value is known.",
            classification="STATISTICAL_RESULT",
            uncertainty=Uncertainty(level="known", point_estimate=None),
        )
    ]
    violations = ec.check_no_manufactured_certainty(findings)
    assert len(violations) == 1
    assert "f0" in violations[0]


def test_known_uncertainty_with_point_estimate_is_silent():
    findings = [
        Finding(
            id="f0",
            statement="The value is known.",
            classification="STATISTICAL_RESULT",
            uncertainty=Uncertainty(level="known", point_estimate=42.0),
        )
    ]
    assert ec.check_no_manufactured_certainty(findings) == []


def test_no_manufactured_certainty_check_safe_on_none():
    assert ec.check_no_manufactured_certainty(None) == []


# --- check_all aggregation -------------------------------------------------------------


def test_check_all_aggregates_violations_from_multiple_checks():
    question = AnalyticalQuestion(original_question="Why did revenue drop?", intent="diagnostic")
    claims = [Claim(text="metric exists", source="system_inferred", status="verified_false")]
    findings = [
        Finding(id="f0", statement="Revenue dropped.", classification="FACT", supporting_evidence=[]),
        Finding(
            id="f1",
            statement="p < 0.05",
            classification="STATISTICAL_RESULT",
            uncertainty=Uncertainty(level="known", point_estimate=None),
        ),
    ]
    evidence: list[Evidence] = []
    hypotheses = [Hypothesis(id="h0", description="one hypothesis only", is_causal=False)]
    limitations: list[Limitation] = []
    recommendation = Recommendation(recommendation="Do X", confidence="high", supporting_findings=["f0"])
    final_answer_text = "Revenue dropped because of the price change."

    violations = ec.check_all(
        question, claims, findings, evidence, hypotheses, limitations, recommendation, final_answer_text
    )

    # Expect at least: unsupported observation (f0), no-uncertainty-point-estimate (f1),
    # unhedged causal phrase, <2 hypotheses for diagnostic question, missing-evidence
    # (verified_false claim with no limitation), overconfidence (high w/ 1 finding).
    assert len(violations) >= 6
    assert len(violations) == len(set(violations))  # deduplicated


def test_check_all_is_safe_on_maximally_empty_input():
    violations = ec.check_all(None, None, None, None, None, None, None, None)
    assert violations == []


def test_check_all_deduplicates_identical_violation_strings():
    # Two findings referencing the SAME violation text would only happen coincidentally
    # in practice, but the aggregator must not emit exact-duplicate strings regardless
    # of which underlying check(s) produced them.
    findings = [
        Finding(id="f0", statement="x", classification="FACT", supporting_evidence=[]),
    ]
    v1 = ec.check_observation_vs_interpretation(findings)
    combined = ec.check_all(None, None, findings, None, None, None, None, None)
    assert combined.count(v1[0]) == 1
