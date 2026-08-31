"""Tests for `app.reasoning.recommendation_grounding` (Phase 4 remediation of the
Phase 3C QA finding logged in `.agent/completed_tasks.md`: the benchmark scorer's
"recommendation grounding" check verified *some* finding backs a recommendation but
never that the finding is *strong enough* to justify the stated confidence).

All objects are hand-constructed `Finding`/`Evidence`/`Recommendation`/`Hypothesis`
instances -- no orchestrator, no mock LLM -- exercising the deterministic grounding
rule directly, tier by tier, plus the four required examples from the task spec.
"""

from __future__ import annotations

from app.reasoning.contracts import Evidence, Finding, Hypothesis, Limitation, Recommendation
from app.reasoning.recommendation_grounding import evaluate_recommendation_grounding


def test_post_outcome_evidence_cannot_ground_an_action():
    evidence = [Evidence(
        id="ev_post", source_tool="correlation_analysis", evidence_type="CALCULATED_RESULT",
        metric="post_outcome_survey", result_summary={"correlation": .8},
        causal_eligible=False, causal_restriction="Measured after the outcome.",
        tool_call_ref="tool_call[0]",
    )]
    findings = [Finding(
        id="finding_post", statement="A descriptive association was observed.",
        classification="CALCULATED_RESULT", supporting_evidence=["ev_post"],
    )]
    recommendation = Recommendation(
        recommendation="Change operations based on the survey.",
        supporting_findings=["finding_post"], confidence="medium",
    )
    report = evaluate_recommendation_grounding(recommendation, findings, evidence, hypotheses=[])
    assert any(item.startswith("post-outcome evidence") for item in report.violations)

# --- 1. Weak correlation backing a causal, high-confidence recommendation ----------


def test_weak_single_correlation_backing_high_confidence_causal_recommendation_is_flagged():
    evidence = [
        Evidence(
            id="ev_0",
            source_tool="correlation_analysis",
            evidence_type="CALCULATED_RESULT",  # no formal significance test at all
            metric="marketing_spend_vs_revenue",
            result_summary={"correlation": 0.42, "n": 120},
            sample_size=120,
            tool_call_ref="tool_call[0]",
        )
    ]
    findings = [
        Finding(
            id="finding_0",
            statement="Marketing spend and revenue are correlated (r=0.42).",
            classification="CALCULATED_RESULT",
            supporting_evidence=["ev_0"],
        )
    ]
    recommendation = Recommendation(
        recommendation="Increase marketing spend because it will improve revenue.",
        supporting_findings=["finding_0"],
        confidence="high",
    )

    report = evaluate_recommendation_grounding(recommendation, findings, evidence, hypotheses=[])

    assert report.evidence_strength == "weak"
    assert report.recommended_confidence_ceiling == "low"
    assert report.is_observational_only is True
    assert report.adjusted_confidence == "low"
    assert any("exceeds the evidence ceiling" in v for v in report.violations)
    assert any("only observational/correlational evidence" in v for v in report.violations)


def test_causal_language_not_flagged_when_a_justifying_hypothesis_exists():
    """Same shape as the above, but a supported causal Hypothesis is present -- the
    causal language is then legitimate, mirroring causation_guard.py's own
    'restrict only unsupported causal language' principle."""
    evidence = [
        Evidence(
            id="ev_0",
            source_tool="correlation_analysis",
            evidence_type="CALCULATED_RESULT",
            metric="marketing_spend_vs_revenue",
            result_summary={"correlation": 0.42},
            sample_size=120,
            tool_call_ref="tool_call[0]",
        )
    ]
    findings = [
        Finding(
            id="finding_0",
            statement="Marketing spend and revenue are correlated.",
            classification="CALCULATED_RESULT",
            supporting_evidence=["ev_0"],
        )
    ]
    recommendation = Recommendation(
        recommendation="Increase marketing spend because it will improve revenue.",
        supporting_findings=["finding_0"],
        confidence="low",
    )
    hypotheses = [
        Hypothesis(id="h1", description="Marketing spend drives revenue", is_causal=True, status="supported")
    ]

    report = evaluate_recommendation_grounding(recommendation, findings, evidence, hypotheses)

    assert report.is_observational_only is False


# --- 2. Tiny sample reduces the ceiling regardless of p-value ----------------------


def test_tiny_sample_statistical_result_reduces_ceiling_regardless_of_p_value():
    evidence = [
        Evidence(
            id="ev_0",
            source_tool="t_test",
            evidence_type="STATISTICAL_RESULT",
            metric="revenue",
            result_summary={"test": "one_sample_t_test", "p_value": 0.0001, "alpha": 0.05, "significant": True, "n": 4},
            sample_size=4,
            tool_call_ref="tool_call[0]",
        )
    ]
    findings = [
        Finding(
            id="finding_0",
            statement="t_test produced a result for 'revenue'.",
            classification="STATISTICAL_RESULT",
            supporting_evidence=["ev_0"],
        )
    ]
    recommendation = Recommendation(
        recommendation="Roll out the new pricing plan.",
        supporting_findings=["finding_0"],
        confidence="high",
    )

    report = evaluate_recommendation_grounding(recommendation, findings, evidence, hypotheses=[])

    assert report.evidence_strength == "weak"
    assert report.sample_size_adequate is False
    assert report.recommended_confidence_ceiling == "low"
    assert report.adjusted_confidence == "low"
    assert any("sample of only 4 observations" in v for v in report.violations)


# --- 3. Statistically significant but practically tiny effect ----------------------


def test_significant_but_negligible_effect_size_does_not_qualify_as_strong():
    evidence = [
        Evidence(
            id="ev_0",
            source_tool="effect_size",
            evidence_type="STATISTICAL_RESULT",
            metric="revenue",
            result_summary={"cohens_d": 0.05, "magnitude": "negligible"},
            sample_size=500,
            tool_call_ref="tool_call[0]",
        ),
        Evidence(
            id="ev_1",
            source_tool="t_test",
            evidence_type="STATISTICAL_RESULT",
            metric="revenue",
            result_summary={"p_value": 0.001, "alpha": 0.05, "significant": True},
            sample_size=500,
            tool_call_ref="tool_call[1]",
        ),
    ]
    findings = [
        Finding(id="finding_0", statement="effect size", classification="STATISTICAL_RESULT", supporting_evidence=["ev_0"]),
        Finding(id="finding_1", statement="t-test", classification="STATISTICAL_RESULT", supporting_evidence=["ev_1"]),
    ]
    recommendation = Recommendation(
        recommendation="Prioritize this variable across all campaigns.",
        supporting_findings=["finding_0", "finding_1"],
        confidence="high",
    )

    report = evaluate_recommendation_grounding(recommendation, findings, evidence, hypotheses=[])

    assert report.evidence_strength != "strong"
    assert report.evidence_strength == "moderate"
    assert report.recommended_confidence_ceiling == "medium"
    assert report.adjusted_confidence == "medium"


def test_significant_but_tiny_r_squared_does_not_qualify_as_strong():
    evidence = [
        Evidence(
            id="ev_0",
            source_tool="linear_regression",
            evidence_type="STATISTICAL_RESULT",
            metric="revenue",
            result_summary={
                "r_squared": 0.02,
                "f_p_value": 0.001,
                "significant_features": ["ad_spend"],
                "n_observations": 200,
            },
            sample_size=200,
            tool_call_ref="tool_call[0]",
        )
    ]
    findings = [
        Finding(id="finding_0", statement="regression", classification="STATISTICAL_RESULT", supporting_evidence=["ev_0"])
    ]
    recommendation = Recommendation(
        recommendation="Base the budget entirely on ad_spend's effect.",
        supporting_findings=["finding_0"],
        confidence="high",
    )

    report = evaluate_recommendation_grounding(recommendation, findings, evidence, hypotheses=[])

    assert report.evidence_strength == "moderate"
    assert report.effect_size_meaningful is False
    assert report.recommended_confidence_ceiling == "medium"
    assert report.adjusted_confidence == "medium"
    assert any("negligible/small effect size" in v for v in report.violations)


# --- 4. No supporting findings at all -----------------------------------------------


def test_no_supporting_findings_yields_none_strength_and_flags_invented_confidence():
    recommendation = Recommendation(
        recommendation="Expand into a new market segment.",
        supporting_findings=[],
        confidence="high",
    )

    report = evaluate_recommendation_grounding(recommendation, findings=[], evidence=[], hypotheses=[])

    assert report.evidence_strength == "none"
    assert report.recommended_confidence_ceiling is None
    assert report.adjusted_confidence is None
    assert any("invented certainty" in v for v in report.violations)


def test_no_supporting_findings_with_null_confidence_is_not_itself_flagged_as_invented():
    recommendation = Recommendation(
        recommendation="Consider expanding into a new market segment.",
        supporting_findings=[],
        confidence=None,
    )

    report = evaluate_recommendation_grounding(recommendation, findings=[], evidence=[], hypotheses=[])

    assert report.evidence_strength == "none"
    assert report.adjusted_confidence is None
    # still flagged as ungrounded in general, just not the "invented certainty" wording
    assert not any("invented certainty" in v for v in report.violations)


def test_dangling_finding_and_evidence_ids_resolve_gracefully_to_none():
    """A recommendation citing ids that don't exist in the passed-in findings/evidence
    lists must not raise -- it should behave the same as citing nothing real."""
    recommendation = Recommendation(
        recommendation="Do something.",
        supporting_findings=["finding_missing"],
        confidence="medium",
    )
    report = evaluate_recommendation_grounding(recommendation, findings=[], evidence=[], hypotheses=[])
    assert report.evidence_strength == "none"


# --- 5. Positive case: strong, well-powered, non-causal-worded evidence ------------


def test_strong_significant_well_powered_evidence_produces_no_violations():
    # `linear_regression` (app/tools/regression.py) is the one real tool whose single
    # result shape carries significance (`significant_features`/coefficient p-values),
    # effect size (`r_squared`), and sample size (`n_observations`) together -- the
    # "strong" tier requires all three on ONE Evidence object per the module
    # docstring's rule 4 (a t-test's p-value and a separate effect_size call's Cohen's
    # d are two different Evidence objects and, on their own, only ever reach
    # "moderate" -- see the effect-size/negligible tests above for that combination).
    evidence = [
        Evidence(
            id="ev_0",
            source_tool="linear_regression",
            evidence_type="STATISTICAL_RESULT",
            metric="revenue",
            result_summary={
                "target_column": "revenue",
                "feature_columns": ["ad_spend"],
                "n_observations": 150,
                "coefficients": {"ad_spend": {"coefficient": 2.1, "p_value": 0.0001, "significant": True}},
                "r_squared": 0.55,
                "f_p_value": 0.0001,
                "alpha": 0.05,
                "significant_features": ["ad_spend"],
            },
            sample_size=150,
            tool_call_ref="tool_call[0]",
        ),
    ]
    findings = [
        Finding(id="finding_0", statement="Ad spend is a significant, well-powered predictor of revenue.", classification="STATISTICAL_RESULT", supporting_evidence=["ev_0"]),
    ]
    recommendation = Recommendation(
        recommendation="Prioritize ad_spend in the next campaign; higher spend is associated with higher revenue.",
        supporting_findings=["finding_0"],
        confidence="high",
    )

    report = evaluate_recommendation_grounding(recommendation, findings, evidence, hypotheses=[])

    assert report.evidence_strength == "strong"
    assert report.statistically_significant is True
    assert report.effect_size_meaningful is True
    assert report.sample_size_adequate is True
    assert report.recommended_confidence_ceiling == "high"
    assert report.violations == []
    assert report.adjusted_confidence == recommendation.confidence == "high"


def test_adjusted_confidence_is_unchanged_when_original_confidence_is_already_within_ceiling():
    evidence = [
        Evidence(
            id="ev_0",
            source_tool="linear_regression",
            evidence_type="STATISTICAL_RESULT",
            metric="revenue",
            result_summary={
                "n_observations": 200,
                "r_squared": 0.5,
                "significant_features": ["ad_spend"],
                "f_p_value": 0.001,
                "alpha": 0.05,
            },
            sample_size=200,
            tool_call_ref="tool_call[0]",
        ),
    ]
    findings = [
        Finding(id="finding_0", statement="regression", classification="STATISTICAL_RESULT", supporting_evidence=["ev_0"]),
    ]
    recommendation = Recommendation(
        recommendation="Consider prioritizing this segment.",
        supporting_findings=["finding_0"],
        confidence="medium",  # already at/below the "high" ceiling this evidence would support
    )

    report = evaluate_recommendation_grounding(recommendation, findings, evidence, hypotheses=[])

    assert report.recommended_confidence_ceiling == "high"
    assert report.adjusted_confidence == "medium"  # unchanged, not bumped up
    assert report.violations == []


# --- blocks_conclusion limitations override even strong evidence -------------------
#
# Real gap found via a systematic professional-analyst-workflow gap audit:
# Limitation.severity == "blocks_conclusion" was being SET in four places
# (premise_validator.py, numerical_sanity.py, confound_detection.py,
# orchestrator.py's own capability-unavailable path) but never actually CHECKED
# anywhere -- it reached the synthesizer's prompt as plain text and relied entirely
# on the LLM noticing and hedging, with zero deterministic enforcement. These tests
# lock in the fix: a blocks_conclusion limitation must force adjusted_confidence to
# None regardless of how strong the resolved evidence otherwise is.


def _strong_evidence_and_recommendation():
    evidence = [
        Evidence(
            id="ev_0",
            source_tool="linear_regression",
            evidence_type="STATISTICAL_RESULT",
            metric="revenue",
            result_summary={
                "n_observations": 150,
                "r_squared": 0.55,
                "significant_features": ["ad_spend"],
                "f_p_value": 0.0001,
                "alpha": 0.05,
            },
            sample_size=150,
            tool_call_ref="tool_call[0]",
        ),
    ]
    findings = [
        Finding(id="finding_0", statement="Ad spend is a significant, well-powered predictor of revenue.", classification="STATISTICAL_RESULT", supporting_evidence=["ev_0"]),
    ]
    recommendation = Recommendation(
        recommendation="Prioritize ad_spend in the next campaign.",
        supporting_findings=["finding_0"],
        confidence="high",
    )
    return recommendation, findings, evidence


def test_blocks_conclusion_limitation_overrides_even_strong_evidence():
    recommendation, findings, evidence = _strong_evidence_and_recommendation()
    limitations = [
        Limitation(category="methodological", text="columns are too linearly dependent for a stable analysis", severity="blocks_conclusion"),
    ]

    report = evaluate_recommendation_grounding(recommendation, findings, evidence, hypotheses=[], limitations=limitations)

    assert report.evidence_strength == "strong"  # the evidence itself is unaffected
    assert report.adjusted_confidence is None  # but no confidence claim survives
    assert any("blocks_conclusion" in v for v in report.violations)


def test_reduces_confidence_severity_limitation_does_not_trigger_the_override():
    """Only 'blocks_conclusion' forces a null confidence -- 'reduces_confidence' and
    'minor' limitations are exactly what the existing evidence-strength ceiling
    already accounts for; this override must not be overly broad."""
    recommendation, findings, evidence = _strong_evidence_and_recommendation()
    limitations = [
        Limitation(category="sample_size", text="based on a modest sample", severity="reduces_confidence"),
    ]

    report = evaluate_recommendation_grounding(recommendation, findings, evidence, hypotheses=[], limitations=limitations)

    assert report.adjusted_confidence == "high"


def test_no_limitations_argument_at_all_preserves_prior_behavior():
    """Backward compatibility: every pre-existing call site (and the many tests above
    that predate this parameter) omits `limitations` entirely -- must behave exactly
    as before, never invent a violation from an absent argument."""
    recommendation, findings, evidence = _strong_evidence_and_recommendation()
    report = evaluate_recommendation_grounding(recommendation, findings, evidence, hypotheses=[])
    assert report.adjusted_confidence == "high"
    assert report.violations == []


def test_blocks_conclusion_limitation_with_no_affected_findings_link_still_applies():
    """Several real sources of a blocks_conclusion limitation (premise_validator's
    scale/time-range mismatches) fire before any Finding exists to attach
    affected_findings to -- the override must be global, not scoped to the
    recommendation's own cited findings, or it would silently miss exactly the cases
    it exists to catch."""
    recommendation, findings, evidence = _strong_evidence_and_recommendation()
    limitations = [
        Limitation(category="insufficient_coverage", text="claimed 10 million rows but the dataset has 4,000", severity="blocks_conclusion", affected_findings=[]),
    ]

    report = evaluate_recommendation_grounding(recommendation, findings, evidence, hypotheses=[], limitations=limitations)

    assert report.adjusted_confidence is None
