from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.agent.providers import LLMProviderError, build_provider_from_settings
from app.datasets.storage import DatasetNotFoundError, get_dataset_store
from app.reasoning.orchestrator import ReasoningOrchestrator
from app.schemas import (
    AnalyticalAuditOut,
    EvidenceOut,
    FindingOut,
    HypothesisOut,
    LimitationOut,
    ReasonRequest,
    ReasonResponse,
    RecommendationOut,
    UncertaintyOut,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/reason", response_model=ReasonResponse)
def reason(request: ReasonRequest) -> ReasonResponse:
    """Production entry point for the Phase 3B reasoning layer (Phase 3C Part A).

    Deliberately a NEW endpoint alongside the existing `/api/chat`, not a
    replacement -- `/api/chat` (DataAnalystAgent's direct tool-calling loop) is
    unchanged and remains fully available (Part B backward-compatibility
    requirement). This route is the bounded, evidence-classified path; `/api/chat`
    remains the faster, simpler path for callers that don't need it.

    Reuses the exact dataset-lookup and provider-construction pattern
    `routes_chat.chat` already uses -- no new dataset access or provider wiring
    logic invented here.
    """
    store = get_dataset_store()
    try:
        record = store.get(request.dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        provider = build_provider_from_settings(record.df)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f"AI provider not configured: {exc}") from exc

    orchestrator = ReasoningOrchestrator(provider)
    try:
        result = orchestrator.analyze(record, request.message)
    except LLMProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    logger.info(
        "reason dataset_id=%s intent=%s findings=%d evidence=%d",
        request.dataset_id,
        result.question.intent,
        len(result.findings),
        len(result.evidence),
    )

    return ReasonResponse(
        answer=result.final_answer_text,
        intent=result.question.intent,
        evidence=[
            EvidenceOut(
                id=e.id,
                source_tool=e.source_tool,
                evidence_type=e.evidence_type,
                metric=e.metric,
                result_summary=e.result_summary,
                population=e.population,
                scope=e.scope.model_dump(mode="json") if e.scope else None,
                sample_size=e.sample_size,
                causal_eligible=e.causal_eligible,
                causal_restriction=e.causal_restriction,
            )
            for e in result.evidence
        ],
        findings=[
            FindingOut(
                id=f.id,
                statement=f.statement,
                classification=f.classification,
                cross_checked=f.cross_checked,
                uncertainty=(
                    UncertaintyOut(
                        level=f.uncertainty.level,
                        point_estimate=f.uncertainty.point_estimate,
                        interval_low=f.uncertainty.interval_low,
                        interval_high=f.uncertainty.interval_high,
                        confidence_level=f.uncertainty.confidence_level,
                        method=f.uncertainty.method,
                    )
                    if f.uncertainty
                    else None
                ),
                supporting_evidence=f.supporting_evidence,
            )
            for f in result.findings
        ],
        limitations=[
            LimitationOut(category=l.category, text=l.text, severity=l.severity, affected_findings=l.affected_findings)
            for l in result.limitations
        ],
        data_caveats=result.data_caveats,
        hypotheses=[
            HypothesisOut(
                id=h.id,
                description=h.description,
                is_causal=h.is_causal,
                status=h.status,
                evidence_for=h.evidence_for,
                evidence_against=h.evidence_against,
            )
            for h in result.hypotheses
        ],
        recommendation=(
            RecommendationOut(
                recommendation=result.recommendation.recommendation,
                expected_business_effect=result.recommendation.expected_business_effect,
                confidence=result.recommendation.confidence,
                assumptions=result.recommendation.assumptions,
                risks=result.recommendation.risks,
                supporting_findings=result.recommendation.supporting_findings,
            )
            if result.recommendation
            else None
        ),
        tools_used=[e.source_tool for e in result.evidence],
        reasoning_trace=result.reasoning_trace,
        principle_violations=result.principle_violations,
        analytical_audit=(
            AnalyticalAuditOut(
                evidence_strength=result.analytical_audit.evidence_strength,
                contradictions=result.analytical_audit.contradictions,
                confounds=result.analytical_audit.confounds,
                numerical_issues=result.analytical_audit.numerical_issues,
                data_quality_issues=result.analytical_audit.data_quality_issues,
                causal_language_hedged=result.analytical_audit.causal_language_hedged,
                hedged_causal_phrases=result.analytical_audit.hedged_causal_phrases,
                recommendation_grounding_violations=result.analytical_audit.recommendation_grounding_violations,
                unresolved_questions=result.analytical_audit.unresolved_questions,
                conclusion_status=result.analytical_audit.conclusion_status,
                final_confidence=result.analytical_audit.final_confidence,
            )
            if result.analytical_audit
            else None
        ),
    )
