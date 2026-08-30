"""Deterministic coverage accounting for diagnostic/root-cause analysis."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.reasoning.categories import TOOL_CATEGORY_MAP, ToolCategory, tool_names_for_categories
from app.reasoning.contracts import AnalysisPlan, AnalyticalQuestion, Evidence

CoverageStage = Literal["planned", "selected", "executed", "evidenced", "unavailable"]
RequirementKind = Literal["temporal", "segment", "statistical"]

_TEMPORAL_TOOLS = {
    "compare_periods", "executive_summary", "train_test_split_timeseries",
    "decompose_timeseries", "forecast", "backtest_forecast",
}
_STATISTICAL_TOOLS = {
    name for name, category in TOOL_CATEGORY_MAP.items() if category == ToolCategory.STATISTICS
}


class ToolCoverage(BaseModel):
    tool_name: str
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
        return not self.unresolved_tools and not self.unresolved_requirements

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
    registered = set(TOOL_CATEGORY_MAP)
    selectable = tool_names_for_categories(plan.capability_categories)
    evidenced_tools = {item.source_tool for item in evidence}
    executed = set(executed_tools)

    ledger: list[ToolCoverage] = []
    recovery_targets: list[str] = []
    unresolved_tools: list[str] = []
    for tool in planned_tools:
        is_registered = tool in registered
        is_selected = is_registered and tool in selectable
        is_evidenced = tool in evidenced_tools
        was_executed = tool in executed
        if is_evidenced:
            ledger.append(ToolCoverage(
                tool_name=tool, stage="evidenced", selected=is_selected, executed=True,
                evidenced=True, reason="The required tool executed and produced usable evidence.",
                transitions=["planned", "selected", "executed", "evidenced"],
            ))
        elif not is_registered:
            ledger.append(ToolCoverage(
                tool_name=tool, stage="unavailable", unavailable=True,
                reason="The planned tool is not registered in the analytical tool catalog.",
                transitions=["planned", "unavailable"],
            ))
            unresolved_tools.append(tool)
        elif not is_selected:
            ledger.append(ToolCoverage(
                tool_name=tool, stage="unavailable", unavailable=True,
                reason="The planned tool is outside the validated capability categories.",
                transitions=["planned", "unavailable"],
            ))
            unresolved_tools.append(tool)
        elif recovery_finished:
            ledger.append(ToolCoverage(
                tool_name=tool, stage="unavailable", selected=True, executed=was_executed, unavailable=True,
                reason="The exact required tool produced no usable evidence after one bounded recovery pass.",
                transitions=["planned", "selected"] + (["executed"] if was_executed else []) + ["unavailable"],
            ))
            unresolved_tools.append(tool)
        else:
            ledger.append(ToolCoverage(
                tool_name=tool, stage="selected", selected=True,
                reason="The required tool is selected but has not yet produced evidence.",
                transitions=["planned", "selected"],
            ))
            recovery_targets.append(tool)

    requirements = _requirements(question, planned_tools, evidence, date_columns)
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
    if statistical_planned:
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
    return requirements


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
