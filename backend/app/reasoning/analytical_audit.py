"""Assembles the structured `AnalyticalAudit` object (v2 reliability mission,
Phase 3: "Deterministic Self-Challenge" / Phase 5: "Conclusion Safety").

Not a new source of truth -- every field here is read from objects
`orchestrator.py` already computes and already tracks as separate lists before
merging them into `AnalysisResult.limitations` (confound/contradiction limitations,
`verifier.py`'s combined cross-check/sample-size/outlier/numerical-sanity list,
`recommendation_grounding.py`'s own report, `causation_guard.py`'s hedging result).
This module's only job is to assemble those into ONE structured view and derive
`conclusion_status` from them deterministically -- see `contracts.py`'s
`AnalyticalAudit`/`ConclusionStatus` docstring for the exact priority order.

Deliberately does NOT introduce a new discriminating field on `Limitation` itself
(e.g. a "source module" tag) to achieve this separation -- that would require
touching every existing `Limitation(...)` construction site across
`confound_detection.py`/`contradiction_detection.py`/`numerical_sanity.py`/
`verifier.py`/`premise_validator.py` for a benefit fully achievable by having the
one caller that already tracks these lists separately (`orchestrator.py`) pass them
in directly.
"""

from __future__ import annotations

from app.reasoning.contracts import AnalyticalAudit, ConclusionStatus, Limitation
from app.reasoning.recommendation_grounding import RecommendationGroundingReport

_UNRESOLVED_CATEGORIES = {
    "missing_data", "insufficient_coverage", "insufficient_data", "unavailable_capability"
}


def _classify_conclusion(
    blocking_limitations: list[Limitation],
    contradiction_limitations: list[Limitation],
    grounding: RecommendationGroundingReport | None,
) -> ConclusionStatus:
    if blocking_limitations:
        return "BLOCKED"
    if contradiction_limitations:
        return "CONTRADICTED"
    if grounding is None:
        # No recommendation was attempted at all -- nothing here to downgrade;
        # whatever the analysis found stands on its own (still subject to any
        # limitation already surfaced elsewhere).
        return "SUPPORTED"
    if grounding.evidence_strength in ("weak", "none") or grounding.violations:
        return "UNCERTAIN"
    if grounding.evidence_strength == "moderate":
        return "WEAKLY_SUPPORTED"
    return "SUPPORTED"


def build_analytical_audit(
    all_limitations: list[Limitation],
    confound_limitations: list[Limitation],
    contradiction_limitations: list[Limitation],
    verifier_limitations: list[Limitation],
    grounding: RecommendationGroundingReport | None,
    causal_language_hedged: bool,
    hedged_causal_phrases: list[str],
) -> AnalyticalAudit:
    blocking = [l for l in all_limitations if l.severity == "blocks_conclusion"]
    data_quality = [l for l in contradiction_limitations if "data-quality" in l.text]
    unresolved = [l for l in all_limitations if l.category in _UNRESOLVED_CATEGORIES]

    return AnalyticalAudit(
        evidence_strength=grounding.evidence_strength if grounding else None,
        contradictions=[l.text for l in contradiction_limitations],
        confounds=[l.text for l in confound_limitations],
        numerical_issues=[l.text for l in verifier_limitations],
        data_quality_issues=[l.text for l in data_quality],
        causal_language_hedged=causal_language_hedged,
        hedged_causal_phrases=list(hedged_causal_phrases),
        recommendation_grounding_violations=list(grounding.violations) if grounding else [],
        unresolved_questions=[l.text for l in unresolved],
        conclusion_status=_classify_conclusion(blocking, contradiction_limitations, grounding),
        final_confidence=grounding.adjusted_confidence if grounding else None,
    )
