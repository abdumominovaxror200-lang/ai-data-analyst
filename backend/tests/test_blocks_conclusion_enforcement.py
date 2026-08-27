"""End-to-end verification that a `blocks_conclusion` Limitation has a REAL
deterministic effect on the final result -- not just on the isolated
`evaluate_recommendation_grounding` function tested in
`tests/reasoning/test_recommendation_grounding.py`, but through the complete chain:

    tool output -> Evidence -> Finding -> Limitation (severity) ->
    verifier/confound/numerical-sanity checks -> recommendation grounding ->
    confidence adjustment -> final AnalysisResult -> API response

Every test here drives the REAL `ReasoningOrchestrator` (real deterministic checks,
real tool execution against a real DataFrame) with a `MockProvider` scripted to
represent an ADVERSARIAL model: strong-looking evidence, a confident recommendation,
and no acknowledgment of the underlying problem -- exactly the "give the system
strong-looking evidence plus a blocks_conclusion limitation and attempt to make the
LLM produce a confident recommendation" scenario. If any of these tests ever failed,
it would mean the deterministic safety net can be talked past by a sufficiently
confident model response -- these are regression traps for exactly that failure mode.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.agent.providers import MockProvider, ProviderResponse, ToolCall
from app.datasets.storage import DatasetRecord
from app.reasoning.orchestrator import ReasoningOrchestrator
from tests.benchmark.hard_fixtures import HARD_FIXTURES


def _record(fixture_name: str) -> DatasetRecord:
    return HARD_FIXTURES[fixture_name]()


def _scripted_provider(parsed_question: dict, plan: dict, tool_calls: list[tuple[str, dict]], final_answer_text: str, recommendation: dict | None):
    """Builds the exact MockProvider response sequence
    build_mock_provider_from_script produces, without importing scoring.py (this file
    intentionally does not depend on the benchmark scoring framework -- it tests the
    orchestrator directly, not a benchmark case)."""
    import json

    responses = [ProviderResponse(content=json.dumps(parsed_question)), ProviderResponse(content=json.dumps(plan))]
    for i, (tool, args) in enumerate(tool_calls):
        responses.append(ProviderResponse(content=None, tool_calls=[ToolCall(id=f"call_{i}", name=tool, arguments=args)]))
    if tool_calls:
        responses.append(ProviderResponse(content="evidence gathered"))
    responses.append(ProviderResponse(content=json.dumps({"final_answer_text": final_answer_text, "recommendation": recommendation})))
    return MockProvider(responses)


_CONFIDENT_RECOMMENDATION = {
    "recommendation": "Immediately reallocate budget to North region based on this strong result.",
    "expected_business_effect": "Meaningful basket-size uplift.",
    "confidence": "high",
    "assumptions": [],
    "risks": [],
}


# --- 1. Severe confound: adversarial model claims high confidence anyway -------------


def test_severe_confound_blocks_a_confident_recommendation_end_to_end():
    """region_size_confound: North is 90% large-format stores, South is 90%
    small-format -- an 80-point gap, confound_detection.py's SEVERE threshold
    (>=70 points), which now escalates to blocks_conclusion severity specifically
    (see confound_detection.py's _SEVERE_CONFOUND_PROPORTION_GAP_THRESHOLD).

    Deliberately scripted with t_test + effect_size evidence -- on its own (without
    any confound check) this combination reaches "moderate" tier (ceiling="medium"),
    per recommendation_grounding.py's own documented rule that a t-test's p-value and
    a separate effect_size call are two different Evidence objects that never reach
    "strong" alone. This isolates the test: asserting confidence is None (not merely
    "not high") proves the NEW blocks_conclusion mechanism did the work -- the
    pre-existing evidence-tier ceiling alone would only cap this to "medium", never to
    None. A weaker test that only checked `!= "high"` would silently pass even if the
    blocks_conclusion wiring were reverted, since the old mechanism already prevents
    "high" here on its own -- exactly the kind of test that looks like it proves
    something but doesn't, which is why this specific evidence shape was chosen."""
    record = _record("region_size_confound")
    parsed_question = {
        "intent": "comparative", "requested_metrics": ["avg_basket"], "requested_dimensions": ["region"],
        "requested_time_range": None, "requested_population": None, "explicit_constraints": [],
        "required_confidence": None, "language": "en", "claims": [],
    }
    plan = {
        "objective": "Compare regions", "capability_categories": ["STATISTICS"], "steps": [],
        "tools_required": ["t_test", "effect_size"], "expected_outputs": [], "validation_steps": [],
        "stopping_conditions": ["done"], "hypotheses": [],
    }
    provider = _scripted_provider(
        parsed_question, plan,
        [
            ("t_test", {"column": "avg_basket", "group_column": "region", "group_a": "North", "group_b": "South"}),
            ("effect_size", {"column": "avg_basket", "group_column": "region", "group_a": "North", "group_b": "South"}),
        ],
        "North clearly and significantly outperforms South with a large effect size -- recommend reallocating budget immediately.",
        {**_CONFIDENT_RECOMMENDATION, "recommendation": "Reallocate budget to North immediately."},
    )
    orchestrator = ReasoningOrchestrator(provider)
    result = orchestrator.analyze(record, "Is North a significantly better-performing region than South?")

    confound_limitations = [l for l in result.limitations if "confound" in l.text.lower()]
    assert confound_limitations, f"no confound limitation was detected at all: {[l.text for l in result.limitations]}"
    assert any(l.severity == "blocks_conclusion" for l in confound_limitations), (
        f"the severe (80-point) confound was not escalated to blocks_conclusion severity: "
        f"{[(l.text, l.severity) for l in confound_limitations]}"
    )
    # The load-bearing assertion -- see the docstring above for why `is None`
    # specifically (not merely `!= "high"`) is required to isolate this mechanism.
    if result.recommendation is not None:
        assert result.recommendation.confidence is None, (
            f"a confident recommendation survived a severe, blocks_conclusion-severity confound: {result.recommendation}"
        )


# --- 2. Singular/ill-conditioned statistical calculation ------------------------------


def test_singular_covariance_tool_failure_blocks_any_confident_recommendation():
    """revenue/cost/profit are linearly dependent (profit = revenue - cost) --
    outlier_analysis_multivariate now refuses (see the Mahalanobis condition-number
    fix). Zero evidence is gathered. The scripted model still tries to hand back a
    confident recommendation anyway -- the orchestrator must not let it survive with
    any confidence, since there is no real evidence behind it at all."""
    record = _record("finance_units_mismatch")  # any real dataset; the tool call fails regardless of data
    df = pd.DataFrame({
        "revenue": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
        "cost": [60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0, 480.0],
        "profit": [40.0, 80.0, 120.0, 160.0, 200.0, 240.0, 280.0, 320.0],
    })
    record = DatasetRecord(id="collinear", original_filename="x.csv", extension=".csv", uploaded_at=pd.Timestamp.utcnow(), df=df, stored_path="u")

    parsed_question = {
        "intent": "descriptive", "requested_metrics": ["revenue", "cost", "profit"], "requested_dimensions": [],
        "requested_time_range": None, "requested_population": None, "explicit_constraints": [],
        "required_confidence": None, "language": "en", "claims": [],
    }
    plan = {
        "objective": "Find joint outliers", "capability_categories": ["REGRESSION"], "steps": [],
        "tools_required": ["outlier_analysis_multivariate"], "expected_outputs": [], "validation_steps": [],
        "stopping_conditions": ["done"], "hypotheses": [],
    }
    provider = _scripted_provider(
        parsed_question, plan,
        [("outlier_analysis_multivariate", {"columns": ["revenue", "cost", "profit"]})],
        "The analysis found a clear set of anomalous transactions -- recommend investigating them as fraud immediately.",
        {**_CONFIDENT_RECOMMENDATION, "recommendation": "Investigate the flagged transactions as suspected fraud."},
    )
    orchestrator = ReasoningOrchestrator(provider)
    result = orchestrator.analyze(record, "Detect joint outliers across revenue, cost, and profit.")

    assert result.evidence == []  # the tool call genuinely failed -- no fabricated evidence
    if result.recommendation is not None:
        assert result.recommendation.confidence is None, (
            f"a confident recommendation was produced with zero real evidence: {result.recommendation}"
        )


# --- 3. Insufficient sample size -------------------------------------------------------


def test_tiny_sample_blocks_high_confidence_end_to_end():
    """4 observations -- far below any threshold for a trustworthy statistical
    conclusion. The scripted model still claims high confidence."""
    df = pd.DataFrame({"revenue": [520.0, 610.0, 495.0, 580.0]})
    record = DatasetRecord(id="tiny", original_filename="x.csv", extension=".csv", uploaded_at=pd.Timestamp.utcnow(), df=df, stored_path="u")

    parsed_question = {
        "intent": "descriptive", "requested_metrics": ["revenue"], "requested_dimensions": [],
        "requested_time_range": None, "requested_population": None, "explicit_constraints": [],
        "required_confidence": None, "language": "en", "claims": [],
    }
    plan = {
        "objective": "Test revenue mean", "capability_categories": ["STATISTICS"], "steps": [],
        "tools_required": ["t_test"], "expected_outputs": [], "validation_steps": [],
        "stopping_conditions": ["done"], "hypotheses": [],
    }
    provider = _scripted_provider(
        parsed_question, plan,
        [("t_test", {"column": "revenue", "popmean": 400.0})],
        "The average revenue is significantly higher than $400 -- strongly recommend raising the pricing floor.",
        {**_CONFIDENT_RECOMMENDATION, "recommendation": "Raise the pricing floor based on this result."},
    )
    orchestrator = ReasoningOrchestrator(provider)
    result = orchestrator.analyze(record, "Is average revenue significantly different from $400?")

    assert any(l.category == "sample_size" for l in result.limitations)
    if result.recommendation is not None:
        assert result.recommendation.confidence in (None, "low"), (
            f"a confident recommendation survived a 4-observation sample: {result.recommendation}"
        )


# --- 4. Real chain: verifier -> API response shape (no orchestrator bypass) ----------


def test_blocks_conclusion_limitation_is_visible_in_the_final_analysis_result():
    """Confirms the limitation itself (not just its downstream effect on confidence)
    survives all the way to the final AnalysisResult object the API serializes --
    the synthesizer sees it (via _limitations_text) and the caller can too."""
    record = _record("region_size_confound")
    parsed_question = {
        "intent": "comparative", "requested_metrics": ["avg_basket"], "requested_dimensions": ["region"],
        "requested_time_range": None, "requested_population": None, "explicit_constraints": [],
        "required_confidence": None, "language": "en", "claims": [],
    }
    plan = {
        "objective": "Compare regions", "capability_categories": ["GENERAL_ANALYSIS"], "steps": [],
        "tools_required": ["group_and_aggregate"], "expected_outputs": [], "validation_steps": [],
        "stopping_conditions": ["done"], "hypotheses": [],
    }
    provider = _scripted_provider(
        parsed_question, plan,
        [("group_and_aggregate", {"group_by": "region", "agg_column": "avg_basket", "agg_func": "mean"})],
        "North outperforms South.",
        None,
    )
    orchestrator = ReasoningOrchestrator(provider)
    result = orchestrator.analyze(record, "Is North a better-performing region than South?")

    limitation_texts = [l.text for l in result.limitations]
    assert any("format" in t.lower() for t in limitation_texts), (
        f"the real confound limitation did not survive into the final result: {limitation_texts}"
    )
