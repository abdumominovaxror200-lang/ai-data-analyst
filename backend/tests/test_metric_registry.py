from __future__ import annotations

import pandas as pd

from app.datasets.metric_registry import MetricDefinition, MetricRegistry
from app.datasets.storage import DatasetRecord
from app.reasoning.contracts import Evidence
from app.reasoning.numerical_sanity import check_numerical_sanity
from app.tools.advanced_charts import pareto_chart_data
from app.tools.segmentation import churn_risk_analysis, cohort_analysis, rfm_analysis
from app.agent.agent import DataAnalystAgent
from app.agent.providers import MockProvider, ProviderResponse


def test_dataset_record_builds_typed_metric_registry() -> None:
    frame = pd.DataFrame({"revenue": [10.0, 12.0], "converted": [0, 1], "conversion_rate": [0.1, 0.2]})
    record = DatasetRecord("id", "x.csv", ".csv", pd.Timestamp.utcnow(), frame, "unused")
    assert record.metrics is not None
    assert record.metrics.require_resolved("revenue").kind == "measure"
    binary_rate = record.metrics.require_resolved("converted_rate")
    assert binary_rate.denominator_aggregation == "row_count"
    assert record.metrics.definition_for("conversion_rate").status == "needs_definition"


def test_unresolved_stored_rate_blocks_numerical_conclusion() -> None:
    registry = MetricRegistry.from_dataframe(pd.DataFrame({"conversion_rate": [0.1, 0.2]}))
    evidence = [Evidence(id="e1", source_tool="describe_data", evidence_type="CALCULATED_RESULT", metric="conversion_rate", result_summary={"mean": 0.15}, sample_size=2, tool_call_ref="tool_call[0]")]
    limitations = check_numerical_sanity(evidence, registry)
    assert any(item.severity == "blocks_conclusion" and "denominator" in item.text for item in limitations)


def test_resolved_binary_rate_does_not_create_denominator_limitation() -> None:
    registry = MetricRegistry.from_dataframe(pd.DataFrame({"converted": [0, 1, 1, 0]}))
    evidence = [Evidence(id="e1", source_tool="describe_data", evidence_type="CALCULATED_RESULT", metric="converted_rate", result_summary={"mean": 0.5}, sample_size=4, tool_call_ref="tool_call[0]")]
    assert not any("denominator" in item.text for item in check_numerical_sanity(evidence, registry))


def test_explicit_rate_definition_requires_and_records_denominator() -> None:
    registry = MetricRegistry()
    registry.register(MetricDefinition(
        name="completion_rate", kind="rate", numerator_column="completed",
        denominator_column="eligible", numerator_aggregation="sum", denominator_aggregation="sum",
        unit="proportion", reason="Reviewed business definition.",
    ))
    assert registry.require_resolved("completion_rate").denominator_column == "eligible"


def test_chat_asks_for_unresolved_denominator_without_calling_provider() -> None:
    frame = pd.DataFrame({"conversion_rate": [0.1, 0.2], "region": ["a", "b"]})
    record = DatasetRecord("id", "x.csv", ".csv", pd.Timestamp.utcnow(), frame, "unused")
    provider = MockProvider([ProviderResponse(content="unsafe")])
    result = DataAnalystAgent(provider).ask(record, "Compare conversion_rate by region")
    assert provider.calls == []
    assert "numerator" in result["answer"] and "denominator" in result["answer"]
    assert result["limitations"][-1].severity == "blocks_conclusion"


def test_rate_and_share_tools_return_explicit_denominators() -> None:
    frame = pd.DataFrame({
        "customer": ["a", "a", "b", "b"],
        "date": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-01-01", "2025-03-01"]),
        "amount": [10.0, 20.0, 15.0, 25.0], "category": ["x", "x", "y", "y"],
    })
    assert cohort_analysis(frame, "customer", "date")["metric_definition"]["denominator"] == "cohort distinct customers"
    assert churn_risk_analysis(frame, "customer", "date", churn_threshold_days=20)["percentage_denominator"]["value"] == 2
    rfm_frame = pd.DataFrame({
        "customer": list("abcde"), "date": pd.to_datetime(["2025-01-01"] * 5),
        "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    assert rfm_analysis(rfm_frame, "customer", "date", "amount")["percentage_denominator"]["value"] == 5
    assert pareto_chart_data(frame, "category", "amount")["percentage_denominator"]["value"] == 70.0
