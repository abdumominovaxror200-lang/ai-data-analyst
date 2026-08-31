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
from app.reasoning.contracts import AnalysisPlan, Evidence, EvidenceScope, TemporalEvidenceScope
from app.schemas import ToolCallRecord
from app.tools.errors import ToolExecutionError

# Tools whose result carries a formal statistical quantity (p-value, confidence
# interval, prediction interval, silhouette score, VIF, ...) -- Evidence.evidence_type
# is set from this, not asked of the LLM, so the classification is deterministic and
# cannot be talked out of by a persuasive-sounding model response.
_STATISTICAL_TOOLS = {
    "compare_periods_inference",
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


class ExactToolRouter:
    """Recovery router exposing only validated, explicitly missing tools."""

    def __init__(self, base: ToolRouter, tool_names: list[str]) -> None:
        requested = set(tool_names)
        self._base = base
        self._schemas = [
            schema for schema in base.available_tools()
            if schema["function"]["name"] in requested
        ]
        self._allowed = {schema["function"]["name"] for schema in self._schemas}

    def available_tools(self) -> list[dict]:
        return self._schemas

    def execute(self, name: str, record: DatasetRecord, params: dict) -> dict:
        if name not in self._allowed:
            raise ToolExecutionError(f"Tool '{name}' is not an exact recovery target.")
        return self._base.execute(name, record, params)


def execute_plan(
    provider: LLMProvider,
    record: DatasetRecord,
    question_text: str,
    plan: AnalysisPlan,
    base_router: ToolRouter | None = None,
) -> tuple[list[Evidence], str, list[str]]:
    """Returns (evidence, raw_agent_narrative, attempted_tool_names). The narrative is an internal,
    intermediate summary from the tool-gathering pass -- the reasoning layer's own
    `synthesizer.py` produces the text actually shown to the user, not this."""
    router = FilteredToolRouter(base_router or ToolRouter(), plan.capability_categories)
    agent = DataAnalystAgent(provider, tool_router=router)

    prompt = _build_execution_prompt(question_text, plan)
    result = agent.ask(record, prompt)

    evidence = [_to_evidence(i, call) for i, call in enumerate(result["tool_calls"])]
    return evidence, result["answer"] or "", list(result.get("tool_attempts") or [])


def execute_recovery(
    provider: LLMProvider,
    record: DatasetRecord,
    question_text: str,
    missing_tools: list[str],
    base_router: ToolRouter | None = None,
    *,
    evidence_offset: int = 0,
) -> tuple[list[Evidence], list[str]]:
    """Run one bounded recovery pass exposing only the exact missing tools."""
    if not missing_tools:
        return [], []
    targets = set(missing_tools)
    agent = DataAnalystAgent(provider, tool_router=ExactToolRouter(base_router or ToolRouter(), missing_tools))
    result = agent.ask(
        record,
        f"Coverage recovery for: {question_text}\n"
        f"Call only the exact missing required tool(s): {', '.join(missing_tools)}. "
        "Do not substitute another tool. If one cannot produce valid evidence, state that clearly.",
    )
    evidence = [
        _to_evidence(evidence_offset + index, call)
        for index, call in enumerate(result["tool_calls"])
        if call.tool in targets
    ]
    return evidence, list(result.get("tool_attempts") or [])


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


_POST_OUTCOME_MARKERS = (
    "post_outcome", "postoutcome", "after_outcome", "after_resolution",
    "post_resolution", "resolution_followup", "resolved_at", "closed_at",
)


def _causal_eligibility(call: ToolCallRecord) -> tuple[bool, str | None]:
    """Conservative name/semantic guard; descriptive use remains allowed."""
    params = call.params or {}
    names: list[str] = []
    for key in ("column", "value_column", "target_column", "agg_column", "segment_column"):
        if isinstance(params.get(key), str):
            names.append(params[key])
    for key in ("columns", "predictor_columns"):
        if isinstance(params.get(key), list):
            names.extend(value for value in params[key] if isinstance(value, str))
    flagged = sorted({name for name in names if any(marker in name.lower() for marker in _POST_OUTCOME_MARKERS)})
    if flagged:
        return False, "Post-outcome or outcome-timestamp variable(s) are descriptive only: " + ", ".join(flagged)
    return True, None


def _guess_population(call: ToolCallRecord) -> str | None:
    """Reads which subset of the dataset this call was actually scoped to, directly
    off the tool call's own real `filters` param -- never guessed from the result.
    `None` means "no filters" (the call ran over the whole dataset), a real,
    meaningful distinct value from "unknown" -- callers should not conflate the two.

    `Evidence.population` existed in the contract but was never populated anywhere
    (confirmed by grep before wiring this in) -- found while building the v2
    contradiction engine's "overall vs subgroup" check, which needs exactly this
    signal to tell an unfiltered ("overall") tool call apart from a segment-scoped
    one for the same metric."""
    params = call.params or {}
    filters = params.get("filters")
    if not isinstance(filters, list) or not filters:
        return None
    parts = []
    for f in filters:
        if isinstance(f, dict) and "column" in f and "op" in f and "value" in f:
            parts.append(f"{f['column']} {f['op']} {f['value']}")
    return " AND ".join(parts) if parts else None


def _evidence_scope(call: ToolCallRecord) -> EvidenceScope:
    params = call.params or {}
    def normalized_filters(key: str) -> list[dict]:
        raw = params.get(key)
        return [dict(item) for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []

    filters = normalized_filters("filters")
    current_filters = normalized_filters("current_filters")
    previous_filters = normalized_filters("previous_filters") or normalized_filters("baseline_filters")
    comparison_groups = {
        key: params[key] for key in ("group_column", "group_a", "group_b") if params.get(key) is not None
    }
    temporal = None
    period_keys = ("current_start", "current_end", "previous_start", "previous_end")
    if any(params.get(key) is not None for key in period_keys):
        temporal = TemporalEvidenceScope(**{key: params.get(key) for key in period_keys})
    return EvidenceScope(
        population=_guess_population(call), filters=filters,
        current_filters=current_filters, previous_filters=previous_filters,
        comparison_groups=comparison_groups, temporal=temporal,
    )


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
    causal_eligible, causal_restriction = _causal_eligibility(call)
    return Evidence(
        id=f"ev_{index}",
        source_tool=call.tool,
        evidence_type=_evidence_type_for_tool(call.tool),
        metric=_guess_metric(call),
        result_summary=_bounded_summary(result),
        population=_guess_population(call),
        scope=_evidence_scope(call),
        sample_size=_guess_sample_size(result),
        causal_eligible=causal_eligible,
        causal_restriction=causal_restriction,
        params=dict(call.params or {}),
        tool_call_ref=f"tool_call[{index}]",
    )
