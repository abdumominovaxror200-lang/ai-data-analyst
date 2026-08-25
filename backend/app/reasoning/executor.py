"""Evidence gathering (Phase 3B.2/3B.3/3B.5).

Deliberately thin: this module creates NO new tool-execution machinery. It drives the
existing `DataAnalystAgent` tool-calling loop (dedup, stagnation-stop, untrusted-data
wrapping, `MAX_TOOL_ITERATIONS` -- all unmodified, per Phase 3B critical rule #5: "Do
not create a second tool execution engine") against a category-filtered view of the
real `ToolRouter`, then converts the resulting `ToolCallRecord`s into typed `Evidence`.
"""

from __future__ import annotations

from app.agent.agent import DataAnalystAgent
from app.agent.providers import LLMProvider
from app.agent.tool_router import ToolRouter
from app.datasets.storage import DatasetRecord
from app.reasoning.categories import filtered_tool_schemas
from app.reasoning.contracts import AnalysisPlan, Evidence
from app.schemas import ToolCallRecord
from app.tools.errors import ToolExecutionError

# Tools whose result carries a formal statistical quantity (p-value, confidence
# interval, prediction interval, silhouette score, VIF, ...) -- Evidence.evidence_type
# is set from this, not asked of the LLM, so the classification is deterministic and
# cannot be talked out of by a persuasive-sounding model response.
_STATISTICAL_TOOLS = {
    "t_test",
    "chi_square_test",
    "anova_test",
    "confidence_interval",
    "effect_size",
    "linear_regression",
    "regression_diagnostics",
    "outlier_analysis_multivariate",
    "forecast",
    "backtest_forecast",
    "decompose_timeseries",
    "kmeans_cluster",
    "pca_reduce",
}
# Raw, unaggregated dataset facts (shape/coverage) rather than a computed result.
_FACT_TOOLS = {"profile_dataset"}
# Everything else (aggregation, filtering, SQL, EDA, segmentation, charts, reports) is
# a CALCULATED_RESULT: a real, deterministic computation over the data, but not a
# formal statistical inference.

_MAX_SUMMARY_KEYS = 20


class FilteredToolRouter:
    """Wraps the real `ToolRouter`, restricting `available_tools()` to a category-
    filtered subset while delegating `execute()` to the same, unmodified handler map.
    This -- not trusting the planner's free-text `tools_required` -- is the actual
    Phase 3B.3 enforcement point: even a hallucinated or overreaching tool name from
    the planning call cannot become callable unless it is also in the filtered set."""

    def __init__(self, base: ToolRouter, categories: list[str]) -> None:
        self._base = base
        self._schemas = filtered_tool_schemas(categories)
        self._allowed = {s["function"]["name"] for s in self._schemas}

    def available_tools(self) -> list[dict]:
        return self._schemas

    def execute(self, name: str, record: DatasetRecord, params: dict) -> dict:
        if name not in self._allowed:
            raise ToolExecutionError(
                f"Tool '{name}' is not available for the analytical categories selected for this question."
            )
        return self._base.execute(name, record, params)


def execute_plan(
    provider: LLMProvider,
    record: DatasetRecord,
    question_text: str,
    plan: AnalysisPlan,
    base_router: ToolRouter | None = None,
) -> tuple[list[Evidence], str]:
    """Returns (evidence, raw_agent_narrative). The narrative is an internal,
    intermediate summary from the tool-gathering pass -- the reasoning layer's own
    `synthesizer.py` produces the text actually shown to the user, not this."""
    router = FilteredToolRouter(base_router or ToolRouter(), plan.capability_categories)
    agent = DataAnalystAgent(provider, tool_router=router)

    prompt = _build_execution_prompt(question_text, plan)
    result = agent.ask(record, prompt)

    evidence = [_to_evidence(i, call) for i, call in enumerate(result["tool_calls"])]
    return evidence, result["answer"] or ""


def _build_execution_prompt(question_text: str, plan: AnalysisPlan) -> str:
    steps_text = "; ".join(plan.steps) or "Gather the evidence needed to answer the question."
    return (
        f"Analysis objective: {plan.objective}\n"
        f"Original question: {question_text}\n"
        f"Planned steps: {steps_text}\n"
        "Call whatever tools you need (from those available to you) to gather the concrete "
        "evidence for this objective. When you have enough evidence, briefly summarize what "
        "you found -- you do not need to produce the final business-facing answer; another "
        "stage does that."
    )


def _evidence_type_for_tool(tool_name: str) -> str:
    if tool_name in _FACT_TOOLS:
        return "FACT"
    if tool_name in _STATISTICAL_TOOLS:
        return "STATISTICAL_RESULT"
    return "CALCULATED_RESULT"


def _guess_metric(call: ToolCallRecord) -> str | None:
    params = call.params or {}
    for key in ("column", "value_column", "target_column", "agg_column"):
        value = params.get(key)
        if isinstance(value, str):
            return value
    return None


def _guess_sample_size(result: dict) -> int | None:
    for key in ("n", "n_observations", "n_rows_used", "row_count", "total_rows", "n_customers"):
        value = result.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _bounded_summary(result: dict) -> dict:
    if len(result) <= _MAX_SUMMARY_KEYS:
        return result
    return dict(list(result.items())[:_MAX_SUMMARY_KEYS])


def _to_evidence(index: int, call: ToolCallRecord) -> Evidence:
    result = call.result or {}
    return Evidence(
        id=f"ev_{index}",
        source_tool=call.tool,
        evidence_type=_evidence_type_for_tool(call.tool),
        metric=_guess_metric(call),
        result_summary=_bounded_summary(result),
        sample_size=_guess_sample_size(result),
        tool_call_ref=f"tool_call[{index}]",
    )
