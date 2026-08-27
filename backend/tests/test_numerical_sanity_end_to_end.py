"""End-to-end verification (real ReasoningOrchestrator, real tool execution) for the
three named requirement-#2 scenarios not yet covered by
`test_blocks_conclusion_enforcement.py`: denominator/population mismatch, unit
mismatch, and aggregation (cross-tool) mismatch.

Unlike the confound/singular-covariance/tiny-sample cases, all three of these are
`severity="reduces_confidence"` by design (see `numerical_sanity.py` and
`verifier._cross_check`'s docstrings) -- they are real, mechanically-detected signals,
but none of them alone invalidates a conclusion the way a severe confound or an
impossible value does. These tests prove two things per scenario:

1. The deterministic check fires on real tool output (not just in the module's own
   unit tests) and the resulting Limitation reaches the final `AnalysisResult` --
   the same "computed but silently dropped before synthesis" failure mode already
   fixed once for `blocks_conclusion` could just as easily hide a `reduces_confidence`
   limitation if some later refactor broke the wiring.
2. `reduces_confidence` severity does NOT, by itself, force `confidence` to None the
   way `blocks_conclusion` does -- confirming the two-tier severity design actually
   behaves as documented, not just as claimed in a docstring.

Impossible-percentage (the fourth named scenario) is deliberately NOT tested here:
investigating it surfaced a real, separate architectural gap -- no tool in the current
toolset ever produces a top-level `_pct`-suffixed field from a value it did not itself
compute as an already-bounded ratio (count/total), so `_find_impossible_percentages`
has no real trigger path through any actual tool call today. That gap is documented in
`.agent/FINAL_GO_NO_GO_AUDIT.md` rather than papered over with a synthetic/injected
Evidence object that would prove nothing about real tool behavior.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.agent.providers import MockProvider, ProviderResponse, ToolCall
from app.datasets.storage import DatasetRecord
from app.reasoning.orchestrator import ReasoningOrchestrator


def _record(df: pd.DataFrame, name: str = "synthetic.csv") -> DatasetRecord:
    return DatasetRecord(id="x", original_filename=name, extension=".csv", uploaded_at=pd.Timestamp.utcnow(), df=df, stored_path="u")


def _scripted_provider(parsed_question: dict, plan: dict, tool_calls: list[tuple[str, dict]], final_answer_text: str, recommendation: dict | None):
    responses = [ProviderResponse(content=json.dumps(parsed_question)), ProviderResponse(content=json.dumps(plan))]
    for i, (tool, args) in enumerate(tool_calls):
        responses.append(ProviderResponse(content=None, tool_calls=[ToolCall(id=f"call_{i}", name=tool, arguments=args)]))
    if tool_calls:
        responses.append(ProviderResponse(content="evidence gathered"))
    responses.append(ProviderResponse(content=json.dumps({"final_answer_text": final_answer_text, "recommendation": recommendation})))
    return MockProvider(responses)


_DESCRIPTIVE_PARSED_QUESTION = {
    "intent": "descriptive", "requested_metrics": ["revenue"], "requested_dimensions": [],
    "requested_time_range": None, "requested_population": None, "explicit_constraints": [],
    "required_confidence": None, "language": "en", "claims": [],
}


# --- 1. Denominator / population mismatch (numerical_sanity._find_population_mismatches) --


def test_population_mismatch_across_two_tools_reaches_the_final_result():
    """t_test (one-sample, filtered to a 15-row subset) and linear_regression (full
    100-row dataset) both examine 'revenue' -- a >=5x sample-size mismatch, exactly
    the 'comparing a filtered subset against the whole' pattern this check exists
    for. Both tools are real: t_test's one-sample shape has a top-level 'n', and
    linear_regression's target_column both surface 'revenue' as the guessed metric
    (app/reasoning/executor.py's _guess_metric/_guess_sample_size, not re-derived
    here)."""
    rng = np.random.default_rng(7)
    n = 100
    df = pd.DataFrame({
        "revenue": rng.normal(1000, 100, n),
        "cost": rng.normal(500, 50, n),
        "segment": ["A"] * 15 + ["B"] * 85,
    })
    record = _record(df)
    plan = {
        "objective": "Check revenue", "capability_categories": ["STATISTICS", "REGRESSION"], "steps": [],
        "tools_required": ["t_test", "linear_regression"], "expected_outputs": [], "validation_steps": [],
        "stopping_conditions": ["done"], "hypotheses": [],
    }
    provider = _scripted_provider(
        _DESCRIPTIVE_PARSED_QUESTION, plan,
        [
            ("t_test", {"column": "revenue", "popmean": 1000.0, "filters": [{"column": "segment", "op": "==", "value": "A"}]}),
            ("linear_regression", {"target_column": "revenue", "feature_columns": ["cost"]}),
        ],
        "Revenue looks broadly consistent with the population mean.",
        None,
    )
    orchestrator = ReasoningOrchestrator(provider)
    result = orchestrator.analyze(record, "Is average revenue consistent with our target of $1000?")

    population_limitations = [l for l in result.limitations if "population sizes" in l.text]
    assert population_limitations, f"no population-mismatch limitation was detected: {[l.text for l in result.limitations]}"
    assert population_limitations[0].severity == "reduces_confidence"
    assert population_limitations[0].category == "methodological"


# --- 2. Unit mismatch (numerical_sanity._find_group_magnitude_outliers) -------------------


def test_unit_mismatch_group_outlier_reaches_the_final_result():
    """5 groups' summed revenue via group_and_aggregate; 4 are ordinary dollar
    totals (~1000 each), the 5th is ~100x larger (a cents-vs-dollars mixup for that
    one group specifically) -- the exact 'cents vs dollars' pattern
    _find_group_magnitude_outliers's docstring names."""
    rows = []
    for i, mult in enumerate([1.0, 1.0, 1.0, 1.0, 100.0]):
        group = f"g{i}"
        for _ in range(10):
            rows.append({"group": group, "revenue": 100.0 * mult})
    df = pd.DataFrame(rows)
    record = _record(df)
    plan = {
        "objective": "Break down revenue by group", "capability_categories": ["GENERAL_ANALYSIS"], "steps": [],
        "tools_required": ["group_and_aggregate"], "expected_outputs": [], "validation_steps": [],
        "stopping_conditions": ["done"], "hypotheses": [],
    }
    provider = _scripted_provider(
        _DESCRIPTIVE_PARSED_QUESTION, plan,
        [("group_and_aggregate", {"group_by": "group", "agg_column": "revenue", "agg_func": "sum"})],
        "g4 has dramatically higher revenue than the other groups.",
        None,
    )
    orchestrator = ReasoningOrchestrator(provider)
    result = orchestrator.analyze(record, "Break down total revenue by group.")

    unit_limitations = [l for l in result.limitations if "units or data-entry" in l.text]
    assert unit_limitations, f"no group-magnitude-outlier limitation was detected: {[l.text for l in result.limitations]}"
    assert unit_limitations[0].severity == "reduces_confidence"


# --- 3. Aggregation / cross-tool mismatch (verifier._cross_check) -------------------------


def test_cross_tool_aggregation_mismatch_reaches_the_final_result():
    """t_test (one-sample, filtered to segment A, mean ~200) and confidence_interval
    (filtered to segment B, mean ~2000) both examine 'revenue' via two DIFFERENT
    tools' top-level 'mean' field -- a real, order-of-magnitude disagreement
    verifier._cross_check is designed to catch, not a contrived scalar mismatch."""
    rng = np.random.default_rng(9)
    df = pd.DataFrame({
        "revenue": np.concatenate([rng.normal(200, 10, 20), rng.normal(2000, 50, 20)]),
        "segment": ["A"] * 20 + ["B"] * 20,
    })
    record = _record(df)
    plan = {
        "objective": "Check revenue by segment", "capability_categories": ["STATISTICS"], "steps": [],
        "tools_required": ["t_test", "confidence_interval"], "expected_outputs": [], "validation_steps": [],
        "stopping_conditions": ["done"], "hypotheses": [],
    }
    provider = _scripted_provider(
        _DESCRIPTIVE_PARSED_QUESTION, plan,
        [
            ("t_test", {"column": "revenue", "popmean": 200.0, "filters": [{"column": "segment", "op": "==", "value": "A"}]}),
            ("confidence_interval", {"column": "revenue", "filters": [{"column": "segment", "op": "==", "value": "B"}]}),
        ],
        "Revenue figures from the two checks are reported below.",
        None,
    )
    orchestrator = ReasoningOrchestrator(provider)
    result = orchestrator.analyze(record, "What is average revenue?")

    disagreement_limitations = [l for l in result.limitations if "results disagree between" in l.text]
    assert disagreement_limitations, f"no cross-check disagreement limitation was detected: {[l.text for l in result.limitations]}"
    assert disagreement_limitations[0].severity == "reduces_confidence"
    assert disagreement_limitations[0].category == "methodological"


# --- 4. reduces_confidence must NOT force confidence to None the way blocks_conclusion does --


def test_reduces_confidence_severity_alone_does_not_force_confidence_to_none():
    """Confirms the two-tier severity design actually behaves as documented: a
    reduces_confidence-only limitation (population mismatch here) coexists with a
    recommendation that still gets a real (non-None) adjusted_confidence, capped only
    by the ordinary evidence-strength ceiling -- proving reduces_confidence is
    advisory (feeds the evidence-tier/LLM-hedging path) while blocks_conclusion is
    the hard structural gate, not that reduces_confidence silently does nothing."""
    rng = np.random.default_rng(11)
    n = 100
    df = pd.DataFrame({
        "revenue": rng.normal(1000, 100, n),
        "cost": rng.normal(500, 50, n),
        "segment": ["A"] * 15 + ["B"] * 85,
    })
    record = _record(df)
    plan = {
        "objective": "Check revenue", "capability_categories": ["STATISTICS", "REGRESSION"], "steps": [],
        "tools_required": ["t_test", "linear_regression"], "expected_outputs": [], "validation_steps": [],
        "stopping_conditions": ["done"], "hypotheses": [],
    }
    recommendation = {
        "recommendation": "Revenue is on target; no action needed.",
        "expected_business_effect": "None -- confirmatory only.",
        "confidence": "low",
        "assumptions": [], "risks": [], "supporting_findings": ["finding_0"],
    }
    provider = _scripted_provider(
        _DESCRIPTIVE_PARSED_QUESTION, plan,
        [
            ("t_test", {"column": "revenue", "popmean": 1000.0, "filters": [{"column": "segment", "op": "==", "value": "A"}]}),
            ("linear_regression", {"target_column": "revenue", "feature_columns": ["cost"]}),
        ],
        "Revenue looks broadly consistent with the population mean.",
        recommendation,
    )
    orchestrator = ReasoningOrchestrator(provider)
    result = orchestrator.analyze(record, "Is average revenue consistent with our target of $1000?")

    assert any("population sizes" in l.text for l in result.limitations)
    assert result.recommendation is not None
    # 'low' was already at/below whatever ceiling applies -- reduces_confidence must
    # not additionally null it out the way blocks_conclusion would.
    assert result.recommendation.confidence == "low"
