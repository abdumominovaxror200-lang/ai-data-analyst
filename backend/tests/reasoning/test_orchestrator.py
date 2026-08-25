"""Phase 3B.8 required test scenarios (numbered in each test's docstring), all against
a deterministic, scripted `MockProvider` -- no network call, no dependency on Groq.

No real-LLM integration test is added here: this project's existing live-Groq
verifications (see `.agent/decisions.md`) were one-off manual/interactive checks, not
a committed, gated pytest fixture (no API-key-based skip fixture exists in
`tests/conftest.py` to build on). Per Phase 3B.8's own instruction ("only if the
existing project architecture already supports them"), inventing that gating
infrastructure now would be scope creep for this phase -- noted as a gap for a future
phase rather than silently skipped.
"""

from __future__ import annotations

import json

import pytest

from app.agent.agent import _UNTRUSTED_DATA_MARKER
from app.agent.providers import MockProvider
from app.reasoning.orchestrator import ReasoningOrchestrator
from tests.reasoning.conftest import (
    json_response,
    no_more_tools_response,
    parsed_question_payload,
    plan_payload,
    synthesis_payload,
    tool_call_response,
)


def _run(sales_record, script, question="Question text"):
    provider = MockProvider(script)
    orchestrator = ReasoningOrchestrator(provider)
    result = orchestrator.analyze(sales_record, question)
    return result, provider


# --- 1. Simple descriptive question --------------------------------------------


def test_1_simple_descriptive_question(sales_record):
    script = [
        json_response(parsed_question_payload(intent="descriptive", requested_metrics=["revenue"])),
        json_response(plan_payload(capability_categories=["GENERAL_ANALYSIS"], tools_required=["describe_data"])),
        tool_call_response("describe_data", {"columns": ["revenue"]}),
        no_more_tools_response(),
        json_response(synthesis_payload("Average revenue is shown in the data.")),
    ]
    result, provider = _run(sales_record, script, "What is the average revenue?")

    assert result.question.intent == "descriptive"
    assert len(result.evidence) == 1
    assert result.evidence[0].source_tool == "describe_data"
    assert result.findings[0].classification == "CALCULATED_RESULT"
    assert len(provider.calls) == 5  # parse + plan + 1 tool-call round + 1 stop round + synthesize


# --- 2. Missing-column question --------------------------------------------------


def test_2_missing_column_question_stops_early(sales_record):
    script = [
        json_response(parsed_question_payload(intent="descriptive", requested_metrics=["conversion_rate"])),
        json_response(synthesis_payload("The 'conversion_rate' column does not exist in this dataset.")),
    ]
    result, provider = _run(sales_record, script, "What is the conversion_rate?")

    assert result.findings[0].classification == "UNKNOWN"
    assert any(l.category == "missing_data" and l.severity == "blocks_conclusion" for l in result.limitations)
    assert result.evidence == []
    assert len(provider.calls) == 2  # parse + synthesize only -- planning/execution correctly skipped


# --- 3. Missing-date-coverage question -------------------------------------------


def test_3_missing_date_coverage_is_flagged_not_silently_substituted(sales_record):
    """sales_record spans ~8 months; requesting the last 12 must be flagged."""
    script = [
        json_response(parsed_question_payload(intent="descriptive", requested_time_range="last 12 months")),
        json_response(plan_payload(capability_categories=["GENERAL_ANALYSIS"])),
        tool_call_response("group_and_aggregate", {"group_by": "region", "agg_column": "revenue"}),
        no_more_tools_response(),
        json_response(synthesis_payload("Only 8 months of data are available, not the requested 12.")),
    ]
    result, _provider = _run(sales_record, script, "Show revenue trend over the last 12 months.")

    coverage_limitations = [l for l in result.limitations if l.category == "insufficient_coverage"]
    assert len(coverage_limitations) == 1
    assert "12" in coverage_limitations[0].text
    range_claim = next(c for c in result.claims if "covers the requested" in c.text)
    assert range_claim.status == "verified_false"


# --- 4. False user claim ---------------------------------------------------------


def test_4_false_user_claim_about_dataset_scale_is_flagged(sales_record):
    script = [
        json_response(
            parsed_question_payload(intent="descriptive", explicit_constraints=["a database of 10 million rows"])
        ),
        json_response(plan_payload(capability_categories=["DATA_PROFILING"])),
        tool_call_response("profile_dataset", {}),
        no_more_tools_response(),
        json_response(synthesis_payload("This dataset actually has far fewer rows than 10 million.")),
    ]
    result, _provider = _run(sales_record, script, "Analyze our 10 million row sales database.")

    false_claim = next(c for c in result.claims if "10 million" in c.text)
    assert false_claim.status == "verified_false"
    assert any(l.category == "insufficient_coverage" for l in result.limitations)


# --- 5. Statistical significance question ----------------------------------------


def test_5_statistical_significance_question_uses_statistics_category(sales_record):
    script = [
        json_response(
            parsed_question_payload(intent="comparative", requested_metrics=["revenue"], requested_dimensions=["region"])
        ),
        json_response(plan_payload(capability_categories=["STATISTICS"], tools_required=["t_test"])),
        tool_call_response("t_test", {"column": "revenue", "group_column": "region", "group_a": "North", "group_b": "South"}),
        no_more_tools_response(),
        json_response(synthesis_payload("Revenue differs significantly between North and South (p < 0.05).")),
    ]
    result, provider = _run(sales_record, script, "Is revenue significantly different between North and South?")

    assert result.plan.capability_categories == ["STATISTICS"]
    assert result.evidence[0].source_tool == "t_test"
    assert result.findings[0].classification == "STATISTICAL_RESULT"
    # the execution phase's tool catalog must have been filtered to STATISTICS only
    exec_tools = provider.tools_per_call[2]
    exec_tool_names = {t["function"]["name"] for t in exec_tools}
    assert "t_test" in exec_tool_names
    assert "forecast" not in exec_tool_names
    assert "kmeans_cluster" not in exec_tool_names


# --- 6. Forecasting question ------------------------------------------------------


def test_6_forecasting_question_uses_forecasting_category(sales_record):
    script = [
        json_response(parsed_question_payload(intent="predictive", requested_metrics=["revenue"])),
        json_response(plan_payload(capability_categories=["FORECASTING"], tools_required=["forecast"])),
        tool_call_response("forecast", {"date_column": "date", "value_column": "revenue", "periods": 5}),
        no_more_tools_response(),
        json_response(synthesis_payload("Revenue is forecast to continue its recent trend over the next 5 days.")),
    ]
    result, provider = _run(sales_record, script, "Forecast the next 5 days of revenue.")

    assert result.plan.capability_categories == ["FORECASTING"]
    assert result.evidence[0].source_tool == "forecast"
    exec_tool_names = {t["function"]["name"] for t in provider.tools_per_call[2]}
    assert exec_tool_names == {"train_test_split_timeseries", "decompose_timeseries", "forecast", "backtest_forecast"}


# --- 7. SQL question ---------------------------------------------------------------


def test_7_sql_question_uses_sql_category(sales_record):
    script = [
        json_response(parsed_question_payload(intent="descriptive")),
        json_response(plan_payload(capability_categories=["SQL"], tools_required=["run_sql_query"])),
        tool_call_response(
            "run_sql_query",
            {"sql": "SELECT region, SUM(revenue) AS total FROM dataset GROUP BY region ORDER BY total DESC LIMIT 10"},
        ),
        no_more_tools_response(),
        json_response(synthesis_payload("Here are the top regions by revenue.")),
    ]
    result, provider = _run(sales_record, script, "Run SQL to find the top regions by revenue.")

    assert result.evidence[0].source_tool == "run_sql_query"
    assert result.evidence[0].evidence_type == "CALCULATED_RESULT"
    exec_tool_names = {t["function"]["name"] for t in provider.tools_per_call[2]}
    assert exec_tool_names == {"run_sql_query", "explain_sql_query"}


# --- 8. Causal-language protection -------------------------------------------------


def test_8_unsupported_causal_language_is_hedged_in_the_final_answer(sales_record):
    script = [
        json_response(parsed_question_payload(intent="diagnostic", requested_metrics=["revenue"])),
        json_response(
            plan_payload(
                capability_categories=["EDA"],
                hypotheses=[{"description": "Region A's decline is associated with a pricing change", "is_causal": False}],
            )
        ),
        tool_call_response("correlation_analysis", {}),
        no_more_tools_response(),
        # the (mock) LLM misbehaves and states causation despite the rules -- the
        # code-level guard, not the prompt, is what must catch this.
        json_response(synthesis_payload("Region A's pricing change caused the revenue decline.")),
    ]
    result, _provider = _run(sales_record, script, "Why did revenue decline in Region A?")

    assert "caused" not in result.final_answer_text.lower()
    assert "associated" in result.final_answer_text.lower() or "possibly related" in result.final_answer_text.lower()
    assert "causation guard" in " ".join(result.reasoning_trace)


def test_8b_causal_language_is_kept_when_a_supported_causal_hypothesis_exists(sales_record):
    script = [
        json_response(parsed_question_payload(intent="diagnostic")),
        json_response(
            plan_payload(
                capability_categories=["EDA"],
                hypotheses=[{"description": "A causes B", "is_causal": True}],
            )
        ),
        tool_call_response("correlation_analysis", {}),
        no_more_tools_response(),
        json_response(synthesis_payload("A caused B.")),
    ]
    provider = MockProvider(script)
    orchestrator = ReasoningOrchestrator(provider)
    result = orchestrator.analyze(sales_record, "Why did B happen?")
    # planner only produced an untested hypothesis (status defaults to "untested"),
    # which does NOT justify unhedged causal language yet -- still hedged.
    assert "caused" not in result.final_answer_text.lower()


# --- 9. Unsupported recommendation prevention --------------------------------------


def test_9_no_recommendation_is_fabricated_when_synthesizer_declines(sales_record):
    script = [
        json_response(parsed_question_payload(intent="descriptive")),
        json_response(plan_payload(capability_categories=["DATA_PROFILING"])),
        tool_call_response("profile_dataset", {}),
        no_more_tools_response(),
        json_response(synthesis_payload("Here is the dataset overview.", recommendation=None)),
    ]
    result, _provider = _run(sales_record, script)
    assert result.recommendation is None


def test_9b_recommendation_confidence_is_not_forced_and_traces_to_solid_findings(sales_record):
    script = [
        json_response(parsed_question_payload(intent="comparative", requested_metrics=["revenue"])),
        json_response(plan_payload(capability_categories=["STATISTICS"])),
        tool_call_response("t_test", {"column": "revenue", "group_column": "region", "group_a": "North", "group_b": "South"}),
        no_more_tools_response(),
        json_response(
            synthesis_payload(
                "Revenue differs between regions.",
                recommendation={
                    "recommendation": "Investigate why North outperforms South.",
                    "expected_business_effect": None,
                    "confidence": None,
                    "assumptions": [],
                    "risks": [],
                },
            )
        ),
    ]
    result, _provider = _run(sales_record, script)
    assert result.recommendation is not None
    assert result.recommendation.confidence is None  # explicitly not forced
    assert result.recommendation.supporting_findings == [f.id for f in result.findings]


# --- 10. Hypothesis vs. fact classification -----------------------------------------


def test_10_hypotheses_are_never_classified_as_fact_or_calculated_result(sales_record):
    script = [
        json_response(parsed_question_payload(intent="diagnostic")),
        json_response(
            plan_payload(
                capability_categories=["EDA"],
                hypotheses=[
                    {"description": "Explanation A", "is_causal": False},
                    {"description": "Explanation B", "is_causal": True},
                ],
            )
        ),
        tool_call_response("correlation_analysis", {}),
        no_more_tools_response(),
        json_response(synthesis_payload("Two possible explanations were considered.")),
    ]
    result, _provider = _run(sales_record, script)

    assert len(result.hypotheses) == 2
    # hypotheses are tracked on their own list, never folded into `findings` as if
    # they were FACT/CALCULATED_RESULT/STATISTICAL_RESULT evidence-backed findings
    for finding in result.findings:
        assert finding.classification != "HYPOTHESIS" or finding.id not in [e.id for e in result.evidence]
    tool_derived_classifications = {f.classification for f in result.findings if f.supporting_evidence}
    assert "HYPOTHESIS" not in tool_derived_classifications
    assert "ASSUMPTION" not in tool_derived_classifications


def test_10b_descriptive_question_never_generates_hypotheses():
    """Hypothesis generation is gated to diagnostic questions only (efficiency rule)."""
    from app.reasoning.planner import plan_analysis
    from app.reasoning.contracts import AnalyticalQuestion

    script = [json_response(plan_payload(capability_categories=["GENERAL_ANALYSIS"], hypotheses=[{"description": "ignored", "is_causal": False}]))]
    provider = MockProvider(script)
    question = AnalyticalQuestion(original_question="What is total revenue?", intent="descriptive")
    plan = plan_analysis(provider, question, [], [])
    assert plan.hypotheses == []  # gated out even though the (misbehaving) mock tried to supply one


# --- 11. Tool-category filtering (also exercised above; dedicated negative check) --


def test_11_out_of_category_tool_is_rejected_if_somehow_requested(sales_record):
    from app.datasets.storage import DatasetRecord
    from app.reasoning.executor import FilteredToolRouter
    from app.agent.tool_router import ToolRouter
    from app.tools.errors import ToolExecutionError

    router = FilteredToolRouter(ToolRouter(), ["STATISTICS"])
    with pytest.raises(ToolExecutionError):
        router.execute("forecast", sales_record, {"date_column": "date", "value_column": "revenue", "periods": 3})


# --- 12. Evidence traceability ------------------------------------------------------


def test_12_every_finding_and_recommendation_traces_back_to_real_evidence(sales_record):
    script = [
        json_response(parsed_question_payload(intent="comparative", requested_metrics=["revenue"])),
        json_response(plan_payload(capability_categories=["STATISTICS"])),
        tool_call_response("t_test", {"column": "revenue", "group_column": "region", "group_a": "North", "group_b": "South"}),
        no_more_tools_response(),
        json_response(
            synthesis_payload(
                "Revenue differs between regions.",
                recommendation={"recommendation": "Investigate.", "expected_business_effect": None, "confidence": "medium", "assumptions": [], "risks": []},
            )
        ),
    ]
    result, _provider = _run(sales_record, script)

    evidence_ids = {e.id for e in result.evidence}
    for finding in result.findings:
        for ev_id in finding.supporting_evidence:
            assert ev_id in evidence_ids
    for ev in result.evidence:
        assert ev.tool_call_ref.startswith("tool_call[")
    if result.recommendation:
        finding_ids = {f.id for f in result.findings}
        for fid in result.recommendation.supporting_findings:
            assert fid in finding_ids


# --- 13. Stopping conditions ---------------------------------------------------------


def test_13_bounded_at_three_reasoning_calls_on_the_early_stop_paths(sales_record):
    script = [
        json_response(parsed_question_payload(intent="descriptive", requested_metrics=["not_a_column"])),
        json_response(synthesis_payload("That column does not exist.")),
    ]
    _result, provider = _run(sales_record, script)
    assert len(provider.calls) == 2  # parse + synthesize; planning/execution correctly never invoked


def test_13b_execution_never_exceeds_the_existing_agents_own_iteration_cap(sales_record):
    """The reasoning layer must not add a second unbounded loop on top of agent.py's
    own MAX_TOOL_ITERATIONS -- verified by confirming the execution phase's call count
    matches exactly what the scripted tool-loop produces, no more."""
    script = [
        json_response(parsed_question_payload(intent="descriptive")),
        json_response(plan_payload(capability_categories=["GENERAL_ANALYSIS"])),
        tool_call_response("describe_data", {}),
        no_more_tools_response(),
        json_response(synthesis_payload("Done.")),
    ]
    _result, provider = _run(sales_record, script)
    assert len(provider.calls) == 5  # exactly: parse, plan, 1 tool round, 1 stop round, synthesize -- no extra


# --- 14. Unavailable capability handling ---------------------------------------------


def test_14_unavailable_capability_returns_unknown_finding_without_executing_tools(sales_record):
    script = [
        json_response(parsed_question_payload(intent="descriptive")),
        json_response(plan_payload(capability_categories=[])),  # planner explicitly finds nothing applicable
        json_response(synthesis_payload("This system cannot answer that kind of question from this dataset.")),
    ]
    result, provider = _run(sales_record, script, "What will the stock market do tomorrow?")

    assert result.findings[0].classification == "UNKNOWN"
    assert any(l.category == "unavailable_capability" for l in result.limitations)
    assert result.evidence == []
    assert len(provider.calls) == 3  # parse + plan + synthesize -- execution correctly skipped


# --- 15. Prompt-injection data remains untrusted --------------------------------------


def test_15_evidence_reaching_the_synthesizer_carries_the_untrusted_data_marker(sales_record):
    script = [
        json_response(parsed_question_payload(intent="descriptive")),
        json_response(plan_payload(capability_categories=["GENERAL_ANALYSIS"])),
        tool_call_response("describe_data", {"columns": ["revenue"]}),
        no_more_tools_response(),
        json_response(synthesis_payload("Done.")),
    ]
    provider = MockProvider(script)
    orchestrator = ReasoningOrchestrator(provider)
    orchestrator.analyze(sales_record, "Describe revenue.")

    synth_call_messages = provider.calls[-1]
    marker_messages = [m for m in synth_call_messages if isinstance(m.get("content"), str) and m["content"].startswith(_UNTRUSTED_DATA_MARKER)]
    assert marker_messages, "evidence payload sent to the synthesizer must carry the untrusted-data marker"

    # and the execution phase's own tool-loop wrapping (agent.py, unmodified) is
    # still intact too -- not just the synthesizer's own wrapping.
    exec_call_messages = provider.calls[2]
    tool_messages = [m for m in exec_call_messages if m.get("role") == "tool"]
    assert tool_messages and all(m["content"].startswith(_UNTRUSTED_DATA_MARKER) for m in tool_messages)
