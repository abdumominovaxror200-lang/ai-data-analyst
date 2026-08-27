"""Unit tests for app.reasoning.analytical_audit -- the structured self-challenge
object (v2 reliability mission, Phase 3/5).

See tests/test_blocks_conclusion_enforcement.py and test_contradiction_detection.py
for end-to-end proof (real ReasoningOrchestrator) that conclusion_status reaches the
final AnalysisResult correctly; these are the isolated unit tests for the
classification/assembly logic itself.
"""

from __future__ import annotations

from app.reasoning.analytical_audit import build_analytical_audit
from app.reasoning.contracts import Limitation
from app.reasoning.recommendation_grounding import RecommendationGroundingReport


def _limitation(category="methodological", severity="reduces_confidence", text="issue"):
    return Limitation(category=category, text=text, severity=severity)


def _grounding(evidence_strength="strong", violations=None, adjusted_confidence="high"):
    return RecommendationGroundingReport(
        evidence_strength=evidence_strength,
        is_observational_only=False,
        recommended_confidence_ceiling=adjusted_confidence,
        violations=violations or [],
        adjusted_confidence=adjusted_confidence,
    )


# --- conclusion_status priority order ----------------------------------------------


def test_a_blocking_limitation_anywhere_produces_blocked_status():
    limitations = [_limitation(severity="blocks_conclusion", text="fatal")]
    audit = build_analytical_audit(limitations, [], [], [], _grounding(), False, [])
    assert audit.conclusion_status == "BLOCKED"


def test_blocked_takes_priority_over_a_contradiction():
    limitations = [_limitation(severity="blocks_conclusion", text="fatal")]
    contradictions = [_limitation(text="A vs B contradiction")]
    audit = build_analytical_audit(limitations + contradictions, [], contradictions, [], _grounding(), False, [])
    assert audit.conclusion_status == "BLOCKED"


def test_a_contradiction_without_blocking_produces_contradicted_status():
    contradictions = [_limitation(text="mean vs median disagree")]
    audit = build_analytical_audit(contradictions, [], contradictions, [], _grounding(), False, [])
    assert audit.conclusion_status == "CONTRADICTED"


def test_weak_evidence_with_no_contradiction_or_block_produces_uncertain():
    audit = build_analytical_audit([], [], [], [], _grounding(evidence_strength="weak", adjusted_confidence="low"), False, [])
    assert audit.conclusion_status == "UNCERTAIN"


def test_grounding_violations_alone_produce_uncertain_even_with_strong_evidence_tier():
    audit = build_analytical_audit([], [], [], [], _grounding(evidence_strength="strong", violations=["stated confidence exceeds ceiling"]), False, [])
    assert audit.conclusion_status == "UNCERTAIN"


def test_moderate_evidence_produces_weakly_supported():
    audit = build_analytical_audit([], [], [], [], _grounding(evidence_strength="moderate", adjusted_confidence="medium"), False, [])
    assert audit.conclusion_status == "WEAKLY_SUPPORTED"


def test_strong_evidence_with_no_issues_produces_supported():
    audit = build_analytical_audit([], [], [], [], _grounding(evidence_strength="strong"), False, [])
    assert audit.conclusion_status == "SUPPORTED"


def test_no_recommendation_attempted_at_all_produces_supported_by_default():
    """No grounding report means no recommendation was ever attempted -- nothing to
    downgrade, so the default status is SUPPORTED (still subject to any limitation
    already surfaced through blocking/contradiction, both checked first)."""
    audit = build_analytical_audit([], [], [], [], None, False, [])
    assert audit.conclusion_status == "SUPPORTED"


# --- field assembly ------------------------------------------------------------------


def test_confounds_and_contradictions_are_kept_as_separate_lists():
    confounds = [_limitation(text="format confounds region")]
    contradictions = [_limitation(text="mean vs median disagree")]
    audit = build_analytical_audit(confounds + contradictions, confounds, contradictions, [], None, False, [])
    assert audit.confounds == [l.text for l in confounds]
    assert audit.contradictions == [l.text for l in contradictions]


def test_data_quality_issues_is_a_subset_of_contradictions_matched_by_the_controlled_phrase():
    contradictions = [
        _limitation(text="mean vs median disagree"),
        _limitation(text="conflicting data-quality signals over the same scope"),
    ]
    audit = build_analytical_audit(contradictions, [], contradictions, [], None, False, [])
    assert audit.data_quality_issues == [contradictions[1].text]
    assert len(audit.contradictions) == 2


def test_unresolved_questions_pulls_from_missing_data_and_unavailable_capability_categories():
    limitations = [
        _limitation(category="missing_data", text="no evidence gathered"),
        _limitation(category="unavailable_capability", text="no capability applies"),
        _limitation(category="sample_size", text="tiny sample"),  # not unresolved-flavored
    ]
    audit = build_analytical_audit(limitations, [], [], [], None, False, [])
    assert set(audit.unresolved_questions) == {limitations[0].text, limitations[1].text}


def test_causal_hedging_is_passed_through_unchanged():
    audit = build_analytical_audit([], [], [], [], None, True, ["caused"])
    assert audit.causal_language_hedged is True
    assert audit.hedged_causal_phrases == ["caused"]


def test_final_confidence_and_evidence_strength_mirror_the_grounding_report():
    audit = build_analytical_audit([], [], [], [], _grounding(evidence_strength="moderate", adjusted_confidence="medium"), False, [])
    assert audit.evidence_strength == "moderate"
    assert audit.final_confidence == "medium"


def test_no_grounding_report_leaves_evidence_strength_and_confidence_none():
    audit = build_analytical_audit([], [], [], [], None, False, [])
    assert audit.evidence_strength is None
    assert audit.final_confidence is None
