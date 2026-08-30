"""Deterministic coverage accounting for diagnostic/root-cause analysis."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.reasoning.categories import TOOL_CATEGORY_MAP, ToolCategory, tool_names_for_categories
from app.reasoning.contracts import AnalysisPlan, AnalyticalQuestion, Evidence

CoverageStage = Literal["planned", "selected", "executed", "evidenced", "unavailable"]
RequirementKind = Literal["temporal", "segment", "statistical", "outlier"]
ObligationKind = Literal["required_analytical", "optional_supporting", "conditional_data_quality"]

_TEMPORAL_TOOLS = {
    "compare_periods", "executive_summary", "train_test_split_timeseries",
    "decompose_timeseries", "forecast", "backtest_forecast",
}
_STATISTICAL_TOOLS = {
    name for name, category in TOOL_CATEGORY_MAP.items() if category == ToolCategory.STATISTICS
}
_PRESENTATION_TOOLS = {
    "generate_report", "generate_chart", "generate_business_insights", "executive_summary",
    "correlation_heatmap_data", "boxplot_data", "pareto_chart_data",
}
_CONDITIONAL_DATA_QUALITY_TOOLS = {
    "duplicate_analysis", "data_quality_report", "analyze_cardinality", "detect_anomalies",
}
_OUTLIER_TOOLS = {"outlier_analysis_multivariate", "detect_anomalies"}
_SPECIALIZED_SEGMENT_TOOLS = {"rfm_analysis", "cohort_analysis", "churn_risk_analysis"}


class ToolCoverage(BaseModel):
    tool_name: str
    obligation: ObligationKind = "required_analytical"
    stage: CoverageStage
    planned: bool = True
    selected: bool = False
    executed: bool = False
    evidenced: bool = False
    unavailable: bool = False
    reason: str
    transitions: list[CoverageStage] = Field(default_factory=list)


class AnalyticalRequirement(BaseModel):
    kind: RequirementKind
    dimension: str | None = None
    required_tools: list[str] = Field(default_factory=list)
    supported: bool = False
    supporting_evidence: list[str] = Field(default_factory=list)
    reason: str


class CoverageAssessment(BaseModel):
    tools: list[ToolCoverage] = Field(default_factory=list)
    requirements: list[AnalyticalRequirement] = Field(default_factory=list)
    recovery_targets: list[str] = Field(default_factory=list)
    unresolved_tools: list[str] = Field(default_factory=list)
    unresolved_requirements: list[str] = Field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.recovery_targets and not self.unresolved_tools and not self.unresolved_requirements

    def gap_explanation(self) -> str:
        parts = []
        if self.unresolved_tools:
            parts.append("required tools without evidence: " + ", ".join(self.unresolved_tools))
        if self.unresolved_requirements:
            parts.append("unsupported analytical requirements: " + ", ".join(self.unresolved_requirements))
        return "; ".join(parts)


def assess_coverage(
    question: AnalyticalQuestion,
    plan: AnalysisPlan,
    evidence: list[Evidence],
    *,
    date_columns: list[str],
    executed_tools: list[str],
    recovery_finished: bool,
) -> CoverageAssessment:
    """Build a lifecycle ledger from validated planned tools and real evidence."""
    planned_tools = list(dict.fromkeys(plan.tools_required))
    if (
        "duplicate_analysis" not in planned_tools
        and _data_quality_relevant("duplicate_analysis", question, plan, evidence)
    ):
        planned_tools.append("duplicate_analysis")
    registered = set(TOOL_CATEGORY_MAP)
    selectable = tool_names_for_categories(plan.capability_categories)
    evidenced_tools = {item.source_tool for item in evidence}
    executed = set(executed_tools)

    ledger: list[ToolCoverage] = []
    recovery_targets: list[str] = []
    unresolved_tools: list[str] = []
    for tool in planned_tools:
        obligation, is_required = _tool_obligation(tool, question, plan, evidence)
        is_registered = tool in registered
        is_selected = is_registered and tool in selectable
        is_evidenced = tool in evidenced_tools
        was_executed = tool in executed
        if is_evidenced:
            ledger.append(ToolCoverage(
                tool_name=tool, obligation=obligation, stage="evidenced", selected=is_selected, executed=True,
                evidenced=True, reason=("The required tool executed and produced usable evidence." if is_required
                                        else "The optional supporting tool produced evidence."),
                transitions=["planned", "selected", "executed", "evidenced"],
            ))
        elif not is_registered:
            ledger.append(ToolCoverage(
                tool_name=tool, obligation=obligation, stage="unavailable", unavailable=True,
                reason="The planned tool is not registered in the analytical tool catalog.",
                transitions=["planned", "unavailable"],
            ))
            if is_required:
                unresolved_tools.append(tool)
        elif not is_selected:
            ledger.append(ToolCoverage(
                tool_name=tool, obligation=obligation, stage="unavailable", unavailable=True,
                reason="The planned tool is outside the validated capability categories.",
                transitions=["planned", "unavailable"],
            ))
            if is_required:
                unresolved_tools.append(tool)
        elif not is_required:
            ledger.append(ToolCoverage(
                tool_name=tool, obligation=obligation, stage="selected", selected=True,
                reason=_optional_reason(obligation), transitions=["planned", "selected"],
            ))
        elif recovery_finished:
            ledger.append(ToolCoverage(
                tool_name=tool, obligation=obligation, stage="unavailable", selected=True,
                executed=was_executed, unavailable=True,
                reason="The exact required tool produced no usable evidence after one bounded recovery pass.",
                transitions=["planned", "selected"] + (["executed"] if was_executed else []) + ["unavailable"],
            ))
            unresolved_tools.append(tool)
        else:
            ledger.append(ToolCoverage(
                tool_name=tool, obligation=obligation, stage="selected", selected=True,
                reason="The required tool is selected but has not yet produced evidence.",
                transitions=["planned", "selected"],
            ))
            recovery_targets.append(tool)

    required_tools = [
        item.tool_name for item in ledger
        if item.obligation == "required_analytical"
        or (item.obligation == "conditional_data_quality" and item.tool_name in unresolved_tools)
    ]
    requirements = _requirements(question, required_tools, evidence, date_columns, plan)
    unresolved_requirements = [
        f"{item.kind}{f'({item.dimension})' if item.dimension else ''}"
        for item in requirements if not item.supported
    ]
    return CoverageAssessment(
        tools=ledger,
        requirements=requirements,
        recovery_targets=recovery_targets,
        unresolved_tools=unresolved_tools,
        unresolved_requirements=unresolved_requirements,
    )


def _requirements(
    question: AnalyticalQuestion,
    planned_tools: list[str],
    evidence: list[Evidence],
    date_columns: list[str],
    plan: AnalysisPlan,
) -> list[AnalyticalRequirement]:
    requirements: list[AnalyticalRequirement] = []
    normalized_dates = {_normalize(item) for item in date_columns}
    temporal_planned = [tool for tool in planned_tools if tool in _TEMPORAL_TOOLS]
    if question.requested_time_range or temporal_planned:
        supporting = [item.id for item in evidence if _covers_temporal(item, normalized_dates)]
        requirements.append(AnalyticalRequirement(
            kind="temporal", required_tools=temporal_planned, supported=bool(supporting),
            supporting_evidence=supporting,
            reason=("Temporal evidence covers the requested time comparison." if supporting
                    else "No evidence uses a temporal tool or the dataset's date dimension."),
        ))

    for dimension in question.requested_dimensions:
        if _normalize(dimension) in normalized_dates:
            continue
        supporting = [item.id for item in evidence if _covers_segment(item, dimension)]
        requirements.append(AnalyticalRequirement(
            kind="segment", dimension=dimension, supported=bool(supporting),
            supporting_evidence=supporting,
            reason=(f"Evidence is grouped or filtered by segment dimension '{dimension}'." if supporting
                    else f"No evidence is grouped or filtered by segment dimension '{dimension}'."),
        ))

    statistical_planned = [tool for tool in planned_tools if tool in _STATISTICAL_TOOLS]
    if statistical_planned or _statistical_relevant(question, plan):
        supporting = [
            item.id for item in evidence
            if item.source_tool in statistical_planned and item.evidence_type == "STATISTICAL_RESULT"
        ]
        requirements.append(AnalyticalRequirement(
            kind="statistical", required_tools=statistical_planned, supported=bool(supporting),
            supporting_evidence=supporting,
            reason=("A required statistical tool produced statistical evidence." if supporting
                    else "No required statistical tool produced statistical evidence."),
        ))
    outlier_planned = [tool for tool in planned_tools if tool in _OUTLIER_TOOLS]
    if outlier_planned or _outlier_relevant(question, plan):
        supporting = [item.id for item in evidence if item.source_tool in _OUTLIER_TOOLS]
        requirements.append(AnalyticalRequirement(
            kind="outlier", required_tools=outlier_planned, supported=bool(supporting),
            supporting_evidence=supporting,
            reason=("Outlier robustness was evaluated with analytical evidence." if supporting
                    else "No analytical evidence evaluates the explicitly requested outlier robustness."),
        ))
    return requirements


def _tool_obligation(
    tool: str, question: AnalyticalQuestion, plan: AnalysisPlan, evidence: list[Evidence]
) -> tuple[ObligationKind, bool]:
    if tool in _PRESENTATION_TOOLS:
        return "optional_supporting", False
    if tool in _CONDITIONAL_DATA_QUALITY_TOOLS:
        required = _data_quality_relevant(tool, question, plan, evidence)
        return "conditional_data_quality", required
    if tool in _STATISTICAL_TOOLS:
        required = _statistical_relevant(question, plan)
        return ("required_analytical" if required else "optional_supporting"), required
    if tool in _TEMPORAL_TOOLS:
        required = bool(question.requested_time_range) or _contains_any(
            _objective_text(question, plan), ("period", "time", "trend", "year over year", "year-over-year")
        )
        return ("required_analytical" if required else "optional_supporting"), required
    if tool in _SPECIALIZED_SEGMENT_TOOLS:
        required = _contains_any(_objective_text(question, plan), ("cohort", "retention", "churn", "rfm"))
        return ("required_analytical" if required else "optional_supporting"), required
    if tool in _OUTLIER_TOOLS:
        required = _outlier_relevant(question, plan)
        return ("required_analytical" if required else "optional_supporting"), required
    return "required_analytical", True


def _data_quality_relevant(
    tool: str, question: AnalyticalQuestion, plan: AnalysisPlan, evidence: list[Evidence]
) -> bool:
    text = _objective_text(question, plan)
    if tool == "duplicate_analysis":
        return _contains_any(
            text, ("duplicate", "deduplic", "double count", "double-count", "repeated row", "join multiplication")
        ) or _evidence_signals_duplicates(evidence)
    return _contains_any(text, ("data quality", "missing", "cardinality", "anomaly", "outlier"))


def _evidence_signals_duplicates(evidence: list[Evidence]) -> bool:
    for item in evidence:
        summary = item.result_summary
        for key in ("duplicate_rows", "duplicate_row_count"):
            value = summary.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                return True
        if "duplicate" in str(summary.get("quality_issues", "")).lower():
            return True
    return False


def _statistical_relevant(question: AnalyticalQuestion, plan: AnalysisPlan) -> bool:
    return bool(question.required_confidence) or _contains_any(
        _objective_text(question, plan),
        ("statistical", "significant", "significance", "confidence interval", "uncertainty", "p-value", "p value"),
    )


def _outlier_relevant(question: AnalyticalQuestion, plan: AnalysisPlan) -> bool:
    return _contains_any(_objective_text(question, plan), ("outlier", "extreme", "anomal", "robustness", "robust"))


def _objective_text(question: AnalyticalQuestion, plan: AnalysisPlan) -> str:
    values = [question.original_question, question.requested_population or "", *question.explicit_constraints,
              plan.objective, *plan.steps, *plan.validation_steps, *plan.expected_outputs]
    return " ".join(values).lower()


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _optional_reason(obligation: ObligationKind) -> str:
    if obligation == "conditional_data_quality":
        return "The data-quality tool is optional because no activating risk signal is present."
    return "The tool supports presentation or non-essential analysis and cannot block a conclusion."


def _covers_temporal(evidence: Evidence, date_columns: set[str]) -> bool:
    if evidence.source_tool in _TEMPORAL_TOOLS:
        return True
    group_by = evidence.result_summary.get("group_by")
    if isinstance(group_by, str) and _normalize(group_by) in date_columns:
        return True
    return any(_population_mentions(evidence.population, column) for column in date_columns)


def _covers_segment(evidence: Evidence, dimension: str) -> bool:
    normalized_dimension = _normalize(dimension)
    group_by = evidence.result_summary.get("group_by")
    if isinstance(group_by, str) and _normalize(group_by) == normalized_dimension:
        return True
    return _population_mentions(evidence.population, normalized_dimension)


def _population_mentions(population: str | None, dimension: str) -> bool:
    if not population:
        return False
    first_token = population.strip().split(maxsplit=1)[0]
    return _normalize(first_token) == _normalize(dimension)


def _normalize(value: str) -> str:
    return value.strip().lower().replace("_", " ")
