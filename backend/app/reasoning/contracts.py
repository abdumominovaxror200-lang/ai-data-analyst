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

HypothesisStatus = Literal["untested", "supported", "weakly_supported", "unsupported", "contradicted", "inconclusive"]
# "unsupported" = evidence exists and does not support the hypothesis. "contradicted" =
# evidence actively points the opposite direction (stronger than "unsupported").
# "inconclusive" (Phase 4 P1) = evidence was gathered but is too weak/ambiguous to
# either support or contradict -- distinct from "untested" (no evidence gathered at all).


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


# --- Analytical audit (v2 reliability mission, Phase 3/5) --------------------------
#
# A single, structured, deterministically-assembled summary of every deterministic
# check this pipeline already runs -- NOT a new source of truth (every field here is
# read from objects `orchestrator.py` already computes: the confound/contradiction/
# numerical-sanity limitation lists it tracks separately before merging them into
# `AnalysisResult.limitations`, `recommendation_grounding.py`'s own report, and
# `causation_guard.py`'s hedging result). Exists so a caller (the LLM at synthesis
# time, or an API consumer) can read ONE object instead of re-deriving "is this
# conclusion actually safe to state confidently" from `limitations`' free text.
#
# `conclusion_status` is a pure function of the fields above it, evaluated in this
# exact priority order (first match wins -- see `_classify_conclusion` in
# `conclusion_guard.py` for the implementation):
#   BLOCKED          -- any blocks_conclusion-severity limitation exists (the
#                       existing, already-enforced mechanism -- this classification
#                       reports it, does not weaken it: `blocks_conclusion` remains
#                       real enforcement via recommendation_grounding.py/
#                       conclusion_guard.py regardless of what this label says).
#   CONTRADICTED      -- a genuine contradiction was detected (contradiction_detection.py)
#                       that was not already severe enough to be BLOCKED.
#   UNCERTAIN         -- evidence strength is "weak"/"none", or a recommendation-
#                       grounding violation exists, with no blocking/contradiction.
#   WEAKLY_SUPPORTED  -- evidence strength is "moderate".
#   SUPPORTED         -- evidence strength is "strong", or no recommendation was
#                       attempted and no issue was found at all.
ConclusionStatus = Literal["SUPPORTED", "WEAKLY_SUPPORTED", "UNCERTAIN", "CONTRADICTED", "BLOCKED"]


class AnalyticalAudit(BaseModel):
    evidence_strength: Literal["strong", "moderate", "weak", "none"] | None = None
    contradictions: list[str] = Field(default_factory=list)
    confounds: list[str] = Field(default_factory=list)
    numerical_issues: list[str] = Field(default_factory=list)
    data_quality_issues: list[str] = Field(default_factory=list)
    causal_language_hedged: bool = False
    hedged_causal_phrases: list[str] = Field(default_factory=list)
    recommendation_grounding_violations: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    conclusion_status: ConclusionStatus = "SUPPORTED"
    final_confidence: Literal["high", "medium", "low"] | None = None


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
    # Phase 4 P2: machine-checkable epistemic-principle violations found in this
    # result by app.reasoning.epistemic_checks -- empty list means no violation was
    # detected, not that none were checked (see that module for exactly what's covered).
    principle_violations: list[str] = Field(default_factory=list)
    # v2 reliability mission, Phase 3/5: the structured audit described above. None
    # only for the two early-stop orchestrator paths that never reach the point
    # where enough of the pipeline has run to assemble one meaningfully.
    analytical_audit: AnalyticalAudit | None = None
    final_answer_text: str
    reasoning_trace: list[str] = Field(default_factory=list)
