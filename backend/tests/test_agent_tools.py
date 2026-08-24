from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.agent.agent import DataAnalystAgent
from app.agent.providers import MockProvider, ProviderResponse, ToolCall
from app.datasets.storage import DatasetRecord
from app.tools.statistics import describe_data


@pytest.fixture
def record() -> DatasetRecord:
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "revenue": rng.normal(1000, 50, 40),
            "region": rng.choice(["North_ZZTOP_UNIQUE_TOKEN", "South"], 40),
        }
    )
    return DatasetRecord(
        id="test-id",
        original_filename="sales.csv",
        extension=".csv",
        uploaded_at=pd.Timestamp.utcnow(),
        df=df,
        stored_path="unused",
    )


def test_agent_never_sends_raw_row_values_to_the_llm(record):
    """The LLM must only see dataset metadata (shape/column names), never actual cell
    values, before it has made any tool call — that's what makes hallucinated numbers
    architecturally impossible rather than just 'discouraged by the prompt'."""
    script = [ProviderResponse(content="No tools needed, here's a generic answer.")]
    provider = MockProvider(script)
    agent = DataAnalystAgent(provider)

    agent.ask(record, "What can you tell me about this dataset?")

    assert len(provider.calls) == 1
    sent_text = json.dumps(provider.calls[0])
    assert "North_ZZTOP_UNIQUE_TOKEN" not in sent_text


def test_agent_answer_numbers_come_from_tool_results_not_the_llm(record):
    """Simulates a real tool-calling round trip: the mock LLM asks for describe_data,
    receives the real computed result, and its final answer is checked against the
    same number the deterministic Python tool produces independently — proving the
    number flows tool -> LLM, not LLM -> user."""
    expected = describe_data(record.df, columns=["revenue"])
    expected_mean = expected["columns"]["revenue"]["mean"]

    script = [
        ProviderResponse(
            content=None,
            tool_calls=[ToolCall(id="call_1", name="describe_data", arguments={"columns": ["revenue"]})],
        ),
        ProviderResponse(content=f"The average revenue is {expected_mean}."),
    ]
    provider = MockProvider(script)
    agent = DataAnalystAgent(provider)

    result = agent.ask(record, "What is the average revenue?")

    assert len(result["tool_calls"]) == 1
    tool_result = result["tool_calls"][0].result
    assert tool_result["columns"]["revenue"]["mean"] == expected_mean
    assert str(expected_mean) in result["answer"]

    # The number appearing in the final answer must trace back to a tool_call result,
    # not have been invented independently by the (mock) LLM.
    tool_message = next(m for m in provider.calls[1] if m.get("role") == "tool")
    assert str(expected_mean) in tool_message["content"]


def test_agent_surfaces_tool_errors_without_crashing(record):
    script = [
        ProviderResponse(
            content=None,
            tool_calls=[ToolCall(id="call_1", name="describe_data", arguments={"columns": ["not_a_column"]})],
        ),
        ProviderResponse(content="That column doesn't exist in this dataset."),
    ]
    provider = MockProvider(script)
    agent = DataAnalystAgent(provider)

    result = agent.ask(record, "Describe not_a_column")
    assert "doesn't exist" in result["answer"]
    assert len(result["tool_calls"]) == 0  # failed tool call is not recorded as a success


def test_agent_stops_after_max_iterations_without_infinite_loop(record):
    infinite_call = ProviderResponse(
        content=None,
        tool_calls=[ToolCall(id="call_x", name="profile_dataset", arguments={})],
    )
    provider = MockProvider([infinite_call] * 20)
    agent = DataAnalystAgent(provider)

    result = agent.ask(record, "Keep going forever")
    assert "tool-call limit" in result["answer"]
    assert len(provider.calls) == 6  # MAX_TOOL_ITERATIONS
