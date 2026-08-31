from __future__ import annotations

import pandas as pd
import pytest

from app.agent.agent import DataAnalystAgent
from app.agent.param_sanity import validate_tool_params
from app.agent.providers import MockProvider, ProviderResponse, ToolCall
from app.datasets.storage import DatasetRecord
from app.interpretation import interpretation_echo
from app.tools.errors import ToolExecutionError


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=12, freq="D"),
            "region": ["north"] * 6 + ["south"] * 6,
            "revenue": list(range(12)),
            "label": ["x"] * 12,
        }
    )


def test_filters_must_leave_at_least_one_row(frame):
    with pytest.raises(ToolExecutionError, match="match zero rows"):
        validate_tool_params(
            frame, "group_and_aggregate",
            {"filters": [{"column": "region", "op": "==", "value": "missing"}]},
        )


def test_requested_period_must_overlap_available_dates(frame):
    with pytest.raises(ToolExecutionError, match="outside the available data range"):
        validate_tool_params(
            frame, "compare_periods",
            {"date_column": "date", "current_start": "2026-01-01", "current_end": "2026-06-30"},
        )


def test_partial_period_overlap_remains_valid_for_coverage_reporting(frame):
    validate_tool_params(
        frame, "compare_periods",
        {"date_column": "date", "current_start": "2024-12-01", "current_end": "2025-01-05"},
    )


@pytest.mark.parametrize("top_n", [0, 101, True, 2.5])
def test_top_n_is_bounded_integer(frame, top_n):
    with pytest.raises(ToolExecutionError, match="top_n must be an integer between 1 and 100"):
        validate_tool_params(frame, "top_n", {"top_n": top_n})


def test_aggregation_column_must_be_numeric(frame):
    with pytest.raises(ToolExecutionError, match="must be numeric"):
        validate_tool_params(frame, "group_and_aggregate", {"agg_column": "label", "agg_func": "mean"})


def test_statistical_groups_require_five_rows_each(frame):
    sparse = frame.copy()
    sparse.loc[4:, "region"] = "south"
    with pytest.raises(ToolExecutionError, match="north.*n=4.*at least 5"):
        validate_tool_params(
            sparse, "t_test",
            {"group_column": "region", "group_a": "north", "group_b": "south"},
        )


def test_echo_reports_interpretation_and_exact_row_counts(frame):
    text = interpretation_echo(
        frame,
        {
            "value_column": "revenue",
            "group_by": "region",
            "filters": [{"column": "region", "op": "==", "value": "north"}],
        },
    )
    assert text.startswith("Understood as: metric=revenue")
    assert "segment=region" in text
    assert "Computed from 6 rows (6 excluded)." in text


def test_chat_answer_always_begins_with_interpretation_echo(frame):
    record = DatasetRecord("id", "data.csv", ".csv", pd.Timestamp.utcnow(), frame, "unused")
    agent = DataAnalystAgent(MockProvider([ProviderResponse(content="A descriptive answer.")]))
    result = agent.ask(record, "Summarize the data")
    assert result["answer"].startswith("Understood as:")
    assert result["answer"].endswith("A descriptive answer.")


def test_chat_echo_uses_executed_tool_parameters(frame):
    record = DatasetRecord("id", "data.csv", ".csv", pd.Timestamp.utcnow(), frame, "unused")
    provider = MockProvider([
        ProviderResponse(content=None, tool_calls=[ToolCall(
            id="call", name="filter_data",
            arguments={"filters": [{"column": "region", "op": "==", "value": "north"}]},
        )]),
        ProviderResponse(content="Filtered result."),
    ])
    result = DataAnalystAgent(provider).ask(record, "Show north")
    assert result["answer"].startswith("Understood as:")
    assert "Computed from 6 rows (6 excluded)." in result["answer"]
