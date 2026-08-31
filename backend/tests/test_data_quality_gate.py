from __future__ import annotations

import json

import pandas as pd

from app.agent.agent import DataAnalystAgent
from app.agent.providers import MockProvider, ProviderResponse
from app.data_quality_gate import evaluate_data_quality
from app.datasets.storage import DatasetRecord
from app.reasoning.orchestrator import ReasoningOrchestrator


def _record(df: pd.DataFrame) -> DatasetRecord:
    return DatasetRecord(id="quality", original_filename="quality.csv", extension=".csv", uploaded_at=pd.Timestamp.utcnow(), df=df, stored_path="unused")


def test_clean_data_has_caveats_without_quality_limitation() -> None:
    caveats, limitations = evaluate_data_quality(pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-02"]), "metric": [1, 2]}))
    assert limitations == []
    assert caveats.duplicate_row_count == 0
    assert caveats.rows_dropped == 0
    assert caveats.column_coverage[0].coverage_pct == 100
    assert caveats.actual_date_ranges["date"]["min"].startswith("2025-01-01")


def test_missing_coverage_and_duplicates_raise_typed_limitations() -> None:
    df = pd.DataFrame({"metric": [None, None, None, 1, None, None], "group": ["a"] * 6})
    caveats, limitations = evaluate_data_quality(df)
    assert caveats.duplicate_pct > 5
    assert any(item.severity == "blocks_conclusion" and "metric" in item.text for item in limitations)
    assert any(item.severity == "reduces_confidence" and "duplicate" in item.text.lower() for item in limitations)


def test_chat_response_always_carries_data_caveats() -> None:
    agent = DataAnalystAgent(MockProvider([ProviderResponse(content="Descriptive answer.")]))
    result = agent.ask(_record(pd.DataFrame({"metric": [1, 2, 3]})), "Summarize")
    assert result["data_caveats"].column_coverage[0].coverage_pct == 100
    assert result["limitations"] == []


def test_chat_blocks_conclusion_before_provider_call_below_fifty_percent_coverage() -> None:
    provider = MockProvider([ProviderResponse(content="Unsafe definitive conclusion.")])
    result = DataAnalystAgent(provider).ask(_record(pd.DataFrame({"metric": [None, None, None, 1]})), "Find the driver")
    assert provider.calls == []
    assert "cannot be completed" in result["answer"]
    assert result["limitations"][0].severity == "blocks_conclusion"


def test_reasoning_early_stop_always_carries_data_caveats() -> None:
    parsed = {
        "intent": "descriptive", "requested_metrics": ["missing_metric"], "requested_dimensions": [],
        "requested_time_range": None, "requested_population": None, "explicit_constraints": [],
        "required_confidence": None, "language": "en", "claims": [],
    }
    synthesis = {"final_answer_text": "The requested field is unavailable.", "recommendation": None}
    provider = MockProvider([ProviderResponse(content=json.dumps(parsed)), ProviderResponse(content=json.dumps(synthesis))])
    result = ReasoningOrchestrator(provider).analyze(_record(pd.DataFrame({"metric": [1, 2]})), "Analyze missing_metric")
    assert result.data_caveats.column_coverage[0].column == "metric"
    assert "mandatory data-quality gate completed" in result.reasoning_trace
