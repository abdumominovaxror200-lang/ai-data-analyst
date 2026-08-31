from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    role: Literal["numeric", "categorical", "datetime", "boolean", "text"]
    missing_count: int
    missing_pct: float
    unique_count: int
    min_date: str | None = None
    max_date: str | None = None


class DatasetProfile(BaseModel):
    dataset_id: str
    filename: str
    rows: int
    columns: int
    column_info: list[ColumnInfo]
    numeric_columns: list[str]
    categorical_columns: list[str]
    date_columns: list[str]
    boolean_columns: list[str]
    missing_total: int
    duplicate_rows: int
    date_ranges: dict[str, dict[str, str]] = Field(default_factory=dict)
    uploaded_at: datetime


class UploadResponse(BaseModel):
    dataset_id: str
    profile: DatasetProfile


class AnalysisRequest(BaseModel):
    dataset_id: str
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)


class AnalysisResponse(BaseModel):
    tool: str
    result: dict[str, Any]
    elapsed_ms: float


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    dataset_id: str
    message: str
    history: list[ChatMessage] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    tool: str
    params: dict[str, Any]
    result: dict[str, Any]


class ChatResponse(BaseModel):
    answer: str
    tool_calls: list[ToolCallRecord]
    charts: list[dict[str, Any]] = Field(default_factory=list)


class ReportRequest(BaseModel):
    dataset_id: str


class ReportResponse(BaseModel):
    dataset_id: str
    generated_at: datetime
    report: dict[str, Any]


class ErrorResponse(BaseModel):
    detail: str


# --- Phase 3C: reasoning-layer API response shapes (additive, append-only per the
# schemas.py convention -- structure mirrors app.reasoning.contracts but is a
# separate, response-scoped view: bounded, no raw tool payloads, no internal id
# cross-references the client has no use for). ---


class ReasonRequest(BaseModel):
    dataset_id: str
    message: str


class UncertaintyOut(BaseModel):
    level: Literal["known", "estimated", "uncertain", "unavailable"]
    point_estimate: float | None = None
    interval_low: float | None = None
    interval_high: float | None = None
    confidence_level: float | None = None
    method: str | None = None


class EvidenceOut(BaseModel):
    """Phase (Master Mission) P1 fix: previously, no field anywhere in this API
    carried the actual computed value(s) behind a Finding -- only a generic
    auto-generated sentence like "describe_data produced a result for 'revenue'."
    `result_summary` is the real tool output (already bounded to <=20 top-level
    keys by app.reasoning.executor._bounded_summary before it ever reaches here),
    so a caller can show/trace the real number, not just meta-commentary."""

    id: str
    source_tool: str
    evidence_type: Literal["FACT", "CALCULATED_RESULT", "STATISTICAL_RESULT"]
    metric: str | None = None
    result_summary: dict[str, Any] = Field(default_factory=dict)
    population: str | None = None
    scope: "EvidenceScopeOut | None" = None
    sample_size: int | None = None


class TemporalEvidenceScopeOut(BaseModel):
    current_start: str | None = None
    current_end: str | None = None
    previous_start: str | None = None
    previous_end: str | None = None


class EvidenceScopeOut(BaseModel):
    population: str | None = None
    filters: list[dict[str, Any]] = Field(default_factory=list)
    current_filters: list[dict[str, Any]] = Field(default_factory=list)
    previous_filters: list[dict[str, Any]] = Field(default_factory=list)
    comparison_groups: dict[str, Any] = Field(default_factory=dict)
    temporal: TemporalEvidenceScopeOut | None = None


EvidenceOut.model_rebuild()


class FindingOut(BaseModel):
    id: str
    statement: str
    classification: Literal["FACT", "CALCULATED_RESULT", "STATISTICAL_RESULT", "HYPOTHESIS", "ASSUMPTION", "UNKNOWN"]
    cross_checked: bool = False
    uncertainty: UncertaintyOut | None = None
    supporting_evidence: list[str] = Field(default_factory=list)  # EvidenceOut.id values


class LimitationOut(BaseModel):
    category: str
    text: str
    severity: str
    affected_findings: list[str] = Field(default_factory=list)  # FindingOut.id values


class HypothesisOut(BaseModel):
    id: str
    description: str
    is_causal: bool
    status: str
    evidence_for: list[str] = Field(default_factory=list)  # EvidenceOut.id values
    evidence_against: list[str] = Field(default_factory=list)


class RecommendationOut(BaseModel):
    recommendation: str
    expected_business_effect: str | None = None
    confidence: Literal["high", "medium", "low"] | None = None
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    supporting_findings: list[str] = Field(default_factory=list)  # FindingOut.id values


class AnalyticalAuditOut(BaseModel):
    """v2 reliability mission, Phase 3/5 -- see app.reasoning.contracts.AnalyticalAudit
    for the full field-by-field derivation and conclusion_status priority order."""

    evidence_strength: Literal["strong", "moderate", "weak", "none"] | None = None
    contradictions: list[str] = Field(default_factory=list)
    confounds: list[str] = Field(default_factory=list)
    numerical_issues: list[str] = Field(default_factory=list)
    data_quality_issues: list[str] = Field(default_factory=list)
    causal_language_hedged: bool = False
    hedged_causal_phrases: list[str] = Field(default_factory=list)
    recommendation_grounding_violations: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    conclusion_status: Literal["SUPPORTED", "WEAKLY_SUPPORTED", "UNCERTAIN", "CONTRADICTED", "BLOCKED"]
    final_confidence: Literal["high", "medium", "low"] | None = None


class ReasonResponse(BaseModel):
    answer: str
    intent: str
    evidence: list[EvidenceOut] = Field(default_factory=list)
    findings: list[FindingOut] = Field(default_factory=list)
    limitations: list[LimitationOut] = Field(default_factory=list)
    hypotheses: list[HypothesisOut] = Field(default_factory=list)
    recommendation: RecommendationOut | None = None
    tools_used: list[str] = Field(default_factory=list)
    reasoning_trace: list[str] = Field(default_factory=list)
    # Phase 4 P2's machine-checkable epistemic-principle audit -- computed since
    # Phase 4 but never surfaced through this API until now (Master Mission P1 fix).
    principle_violations: list[str] = Field(default_factory=list)
    analytical_audit: AnalyticalAuditOut | None = None
