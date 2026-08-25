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


class FindingOut(BaseModel):
    statement: str
    classification: Literal["FACT", "CALCULATED_RESULT", "STATISTICAL_RESULT", "HYPOTHESIS", "ASSUMPTION", "UNKNOWN"]
    cross_checked: bool = False
    uncertainty_level: Literal["known", "estimated", "uncertain", "unavailable"] | None = None


class LimitationOut(BaseModel):
    category: str
    text: str
    severity: str


class HypothesisOut(BaseModel):
    description: str
    is_causal: bool
    status: str


class RecommendationOut(BaseModel):
    recommendation: str
    expected_business_effect: str | None = None
    confidence: Literal["high", "medium", "low"] | None = None
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class ReasonResponse(BaseModel):
    answer: str
    intent: str
    findings: list[FindingOut] = Field(default_factory=list)
    limitations: list[LimitationOut] = Field(default_factory=list)
    hypotheses: list[HypothesisOut] = Field(default_factory=list)
    recommendation: RecommendationOut | None = None
    tools_used: list[str] = Field(default_factory=list)
    reasoning_trace: list[str] = Field(default_factory=list)
