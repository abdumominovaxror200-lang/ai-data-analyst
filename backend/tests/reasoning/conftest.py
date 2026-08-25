from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.agent.providers import ProviderResponse, ToolCall
from app.datasets.storage import DatasetRecord


@pytest.fixture
def sales_record() -> DatasetRecord:
    """8 months of daily data, two regions -- deliberately less than a full year, so
    date-coverage-mismatch tests (e.g. 'last 12 months') have a real gap to catch."""
    rng = np.random.default_rng(42)
    n = 240  # ~8 months of daily data
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "customer_id": rng.integers(1, 50, n),
            "revenue": np.concatenate([rng.normal(600, 40, n // 2), rng.normal(500, 40, n - n // 2)]).round(2),
            "cost": rng.normal(300, 30, n).round(2),
            "quantity": rng.integers(1, 8, n),
            "region": rng.choice(["North", "South"], n, p=[0.6, 0.4]),
        }
    )
    return DatasetRecord(
        id="sales-test",
        original_filename="sales.csv",
        extension=".csv",
        uploaded_at=pd.Timestamp.utcnow(),
        df=df,
        stored_path="unused",
    )


def json_response(payload: dict) -> ProviderResponse:
    return ProviderResponse(content=json.dumps(payload))


def tool_call_response(tool: str, arguments: dict, call_id: str = "call_1") -> ProviderResponse:
    return ProviderResponse(content=None, tool_calls=[ToolCall(id=call_id, name=tool, arguments=arguments)])


def no_more_tools_response(text: str = "Evidence gathered.") -> ProviderResponse:
    return ProviderResponse(content=text)


def parsed_question_payload(
    intent: str = "descriptive",
    requested_metrics: list[str] | None = None,
    requested_dimensions: list[str] | None = None,
    requested_time_range: str | None = None,
    requested_population: str | None = None,
    explicit_constraints: list[str] | None = None,
    claims: list[dict] | None = None,
) -> dict:
    return {
        "intent": intent,
        "requested_metrics": requested_metrics or [],
        "requested_dimensions": requested_dimensions or [],
        "requested_time_range": requested_time_range,
        "requested_population": requested_population,
        "explicit_constraints": explicit_constraints or [],
        "required_confidence": None,
        "language": "en",
        "claims": claims or [],
    }


def plan_payload(
    objective: str = "Answer the question",
    capability_categories: list[str] | None = None,
    steps: list[str] | None = None,
    tools_required: list[str] | None = None,
    hypotheses: list[dict] | None = None,
) -> dict:
    return {
        "objective": objective,
        "capability_categories": capability_categories or [],
        "steps": steps or [],
        "tools_required": tools_required or [],
        "expected_outputs": [],
        "validation_steps": [],
        "stopping_conditions": ["sufficient evidence gathered"],
        "hypotheses": hypotheses or [],
    }


def synthesis_payload(final_answer_text: str, recommendation: dict | None = None) -> dict:
    return {"final_answer_text": final_answer_text, "recommendation": recommendation}
