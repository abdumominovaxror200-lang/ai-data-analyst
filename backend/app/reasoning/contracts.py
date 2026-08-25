"""Typed contracts for the Evidence-Based Analytical Reasoning Layer (Phase 3B).

These are the objects the reasoning pipeline (`orchestrator.py`) passes between its
stages. Nothing here performs computation — every numeric value still comes from a
call into the existing, unmodified tool infrastructure (`app.agent.tool_router`,
`app.agent.agent`). This module only gives that computation typed, auditable
structure: what was asked, what was found, how certain it is, and what it does and
does not support concluding.

Reconciliation note (see the Phase 3B completion report / .agent/decisions.md for the
full record): this supersedes the earlier, less granular contract sketch in
`.agent/reasoning-layer-design.md` where the two disagree. Three notable
reconciliations:
  - `Uncertainty` here is a categorical qualifier (known/estimated/uncertain/
    unavailable) with *optional* quantitative CI/point-estimate fields attached,
    merging the original design's quantitative-only shape with this phase's
    explicit categorical requirement.
  - `Hypothesis.status` uses a 5-way scale (untested/supported/weakly_supported/
    unsupported/contradicted) instead of the original 3-way one, for finer-grained
    causal-claim gating (see `causation_guard.py`).
  - `Recommendation.confidence` is nullable — Phase 3B's explicit rule is "do not
    force fake numerical/categorical confidence when the evidence does not justify
    it," so `None` is a valid, meaningful value, not a missing field.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --- 1. AnalyticalQuestion ---------------------------------------------------

QuestionIntent = Literal["descriptive", "diagnostic", "comparative", "predictive", "prescriptive"]


class AnalyticalQuestion(BaseModel):
    original_question: str
    intent: QuestionIntent
    requested_metrics: list[str] = Field(default_factory=list)
    requested_dimensions: list[str] = Field(default_factory=list)
    requested_time_range: str | None = None
    requested_population: str | None = None
    explicit_constraints: list[str] = Field(default_factory=list)
    required_confidence: str | None = None
    language: str = "en"


# --- 2. Claim -----------------------------------------------------------------

ClaimSource = Literal["user_asserted", "system_inferred"]
ClaimStatus = Literal["unverified", "verified_true", "verified_false", "partially_true", "unverifiable"]


class Claim(BaseModel):
    text: str
    source: ClaimSource
    status: ClaimStatus = "unverified"
    note: str | None = None


# --- 3. Evidence ----------------------------------------------------------------

EvidenceType = Literal["FACT", "CALCULATED_RESULT", "STATISTICAL_RESULT"]


class Evidence(BaseModel):
    id: str
    source_tool: str
    evidence_type: EvidenceType
    metric: str | None = None
    result_summary: dict = Field(default_factory=dict)
    population: str | None = None
    time_coverage: str | None = None
    sample_size: int | None = None
    limitations: list[str] = Field(default_factory=list)
    confidence_info: str | None = None
    tool_call_ref: str  # e.g. "tool_call[0]" -- index into the underlying ToolCallRecord list


# --- 4. Hypothesis --------------------------------------------------------------

HypothesisStatus = Literal["untested", "supported", "weakly_supported", "unsupported", "contradicted"]


class Hypothesis(BaseModel):
    id: str
    description: str
    is_causal: bool
    evidence_for: list[str] = Field(default_factory=list)  # Evidence.id
    evidence_against: list[str] = Field(default_factory=list)  # Evidence.id
    status: HypothesisStatus = "untested"


# --- 5. AnalysisPlan --------------------------------------------------------------


class AnalysisPlan(BaseModel):
    objective: str
    capability_categories: list[str] = Field(default_factory=list)  # see categories.py's ToolCategory
    steps: list[str] = Field(default_factory=list)
    tools_required: list[str] = Field(default_factory=list)  # descriptive hint only, not trusted for execution
    expected_outputs: list[str] = Field(default_factory=list)
    validation_steps: list[str] = Field(default_factory=list)
    stopping_conditions: list[str] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)  # populated only for diagnostic questions


# --- 6. Finding -------------------------------------------------------------------

FindingClassification = Literal[
    "FACT", "CALCULATED_RESULT", "STATISTICAL_RESULT", "HYPOTHESIS", "ASSUMPTION", "UNKNOWN"
]


class Finding(BaseModel):
    id: str
    statement: str
    classification: FindingClassification
    supporting_evidence: list[str] = Field(default_factory=list)  # Evidence.id
    uncertainty: "Uncertainty | None" = None
    cross_checked: bool = False


# --- 7. Uncertainty ----------------------------------------------------------------

UncertaintyLevel = Literal["known", "estimated", "uncertain", "unavailable"]


class Uncertainty(BaseModel):
    level: UncertaintyLevel
    metric: str | None = None
    point_estimate: float | None = None
    interval_low: float | None = None
    interval_high: float | None = None
    confidence_level: float | None = None
    method: str | None = None


Finding.model_rebuild()

# --- 8. Limitation ------------------------------------------------------------------

LimitationCategory = Literal[
    "missing_data",
    "insufficient_coverage",
    "sample_size",
    "unavailable_capability",
    "methodological",
    "resource_limit",
    "other",
]


class Limitation(BaseModel):
    category: LimitationCategory
    text: str
    severity: Literal["blocks_conclusion", "reduces_confidence", "minor"] = "reduces_confidence"
    affected_findings: list[str] = Field(default_factory=list)  # Finding.id


# --- 9. Recommendation ---------------------------------------------------------------


class Recommendation(BaseModel):
    recommendation: str
    supporting_findings: list[str] = Field(default_factory=list)  # Finding.id
    expected_business_effect: str | None = None
    confidence: Literal["high", "medium", "low"] | None = None  # None = evidence does not justify a confidence claim
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


# --- Top-level pipeline output -----------------------------------------------------


class AnalysisResult(BaseModel):
    question: AnalyticalQuestion
    claims: list[Claim] = Field(default_factory=list)
    plan: AnalysisPlan | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    limitations: list[Limitation] = Field(default_factory=list)
    recommendation: Recommendation | None = None
    final_answer_text: str
    reasoning_trace: list[str] = Field(default_factory=list)
