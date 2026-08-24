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
