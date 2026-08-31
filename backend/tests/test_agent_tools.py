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
    """Every call here is genuinely distinct (different top_n each time) so duplicate
    detection never kicks in — this isolates the hard MAX_TOOL_ITERATIONS fallback."""
    from app.agent.agent import MAX_TOOL_ITERATIONS

    varied_calls = [
        ProviderResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id=f"call_{i}",
                    name="group_and_aggregate",
                    arguments={"group_by": "region", "agg_column": "revenue", "top_n": i + 1},
                )
            ],
        )
        for i in range(20)
    ]
    provider = MockProvider(varied_calls)
    agent = DataAnalystAgent(provider)

    result = agent.ask(record, "Keep going forever")
    assert "tool-call limit" in result["answer"]
    assert len(provider.calls) == MAX_TOOL_ITERATIONS


def test_agent_dataset_context_includes_date_coverage():
    """Date coverage must be front-and-center in what the model sees from the start —
    this is what lets it catch a '12 months' request the data can't actually support,
    without needing to call a tool first to discover the mismatch."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", "2024-08-31", freq="D"),
            "revenue": range(244),
        }
    )
    record = DatasetRecord(
        id="test-id-2",
        original_filename="short_history.csv",
        extension=".csv",
        uploaded_at=pd.Timestamp.utcnow(),
        df=df,
        stored_path="unused",
    )
    provider = MockProvider([ProviderResponse(content="ok")])
    agent = DataAnalystAgent(provider)

    agent.ask(record, "Analyze the last 12 months.")

    sent_text = json.dumps(provider.calls[0])
    assert "2024-01-01" in sent_text
    assert "2024-08-31" in sent_text


def test_agent_stops_early_on_duplicate_tool_calls(record):
    """A real benchmark run showed the agent repeating identical tool calls and
    eventually just hitting the hard iteration cap with no answer at all. Duplicate
    calls should instead trigger an early, graceful stop — using far fewer provider
    round-trips than the hard cap, and (when the model cooperates) a real answer."""
    same_call = ProviderResponse(
        content=None,
        tool_calls=[ToolCall(id="call_x", name="profile_dataset", arguments={})],
    )
    script = [
        same_call,  # executed for real
        same_call,  # exact duplicate -> triggers the stagnant-round stop
        ProviderResponse(content="Based on what I already found, here is the summary."),
    ]
    provider = MockProvider(script)
    agent = DataAnalystAgent(provider)

    result = agent.ask(record, "Keep asking the same thing")

    # Only ONE real tool execution should be recorded — the duplicate must not re-run.
    assert len(result["tool_calls"]) == 1
    # Stops after 3 provider round-trips (2 duplicate attempts + 1 forced final answer),
    # nowhere near the 10-iteration hard cap.
    assert len(provider.calls) == 3
    assert result["answer"].startswith("Understood as:")
    assert result["answer"].endswith("Based on what I already found, here is the summary.")

    # The forced final call must have gone out with no tools (so the model can't just
    # request yet another duplicate instead of answering).
    assert provider.tools_per_call[-1] == []
