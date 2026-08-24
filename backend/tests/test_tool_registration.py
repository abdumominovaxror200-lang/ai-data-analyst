from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.agent import tool_router
from app.agent.agent import DataAnalystAgent, _UNTRUSTED_DATA_MARKER
from app.agent.providers import MockProvider, ProviderResponse, ToolCall
from app.agent.tool_router import TOOL_SCHEMAS, ToolRouter
from app.datasets.storage import DatasetRecord
from app.tools.errors import ToolExecutionError

# Wave 1 + Wave 2 tools this task registers, on top of the 10 pre-existing tools.
_NEWLY_REGISTERED_TOOLS = {
    "t_test",
    "chi_square_test",
    "anova_test",
    "confidence_interval",
    "effect_size",
    "linear_regression",
    "regression_diagnostics",
    "outlier_analysis_multivariate",
    "train_test_split_timeseries",
    "decompose_timeseries",
    "forecast",
    "backtest_forecast",
    "kmeans_cluster",
    "pca_reduce",
    "rfm_analysis",
    "cohort_analysis",
    "churn_risk_analysis",
    "automated_eda",
    "analyze_cardinality",
    "analyze_distributions",
    "run_sql_query",
    "explain_sql_query",
}


@pytest.fixture
def router() -> ToolRouter:
    return ToolRouter()


@pytest.fixture
def record() -> DatasetRecord:
    rng = np.random.default_rng(11)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "customer_id": rng.integers(1, 40, n),
            "revenue": rng.normal(500, 80, n).round(2),
            "cost": rng.normal(300, 50, n).round(2),
            "quantity": rng.integers(1, 10, n),
            "region": rng.choice(["North", "South", "East", "West"], n),
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


# --- 1. Discoverability -------------------------------------------------------


def test_every_schema_has_a_handler_and_vice_versa():
    schema_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    handler_names = set(tool_router._HANDLERS.keys())
    assert schema_names == handler_names


def test_all_newly_registered_wave1_wave2_tools_are_discoverable(router):
    available = {s["function"]["name"] for s in router.available_tools()}
    missing = _NEWLY_REGISTERED_TOOLS - available
    assert not missing, f"Tools not registered/discoverable: {missing}"


def test_no_duplicate_tool_names():
    names = [s["function"]["name"] for s in TOOL_SCHEMAS]
    assert len(names) == len(set(names))


# --- 2. Schema validity ---------------------------------------------------------


@pytest.mark.parametrize("schema", TOOL_SCHEMAS, ids=lambda s: s["function"]["name"])
def test_tool_schema_is_well_formed(schema):
    assert schema["type"] == "function"
    fn = schema["function"]
    assert isinstance(fn["name"], str) and fn["name"]
    assert isinstance(fn["description"], str) and len(fn["description"]) > 10
    params = fn["parameters"]
    assert params["type"] == "object"
    assert isinstance(params.get("properties", {}), dict)
    for required_name in params.get("required", []):
        assert required_name in params["properties"], (
            f"{fn['name']}: required field '{required_name}' has no property definition"
        )


# --- 3. Invalid arguments are rejected ------------------------------------------


def test_missing_required_argument_is_rejected(router, record):
    with pytest.raises((TypeError, ToolExecutionError)):
        router.execute("t_test", record, {})  # missing 'column'
    with pytest.raises((TypeError, ToolExecutionError)):
        router.execute("linear_regression", record, {"target_column": "revenue"})  # missing feature_columns
    with pytest.raises((TypeError, ToolExecutionError)):
        router.execute("run_sql_query", record, {})  # missing 'sql'


def test_semantically_invalid_argument_is_rejected(router, record):
    with pytest.raises(ToolExecutionError):
        router.execute("t_test", record, {"column": "revenue"})  # neither group_column nor popmean
    with pytest.raises(ToolExecutionError):
        router.execute("kmeans_cluster", record, {"columns": ["revenue"]})  # needs >= 2 columns
    with pytest.raises(ToolExecutionError):
        router.execute("forecast", record, {"date_column": "date", "value_column": "revenue", "periods": -1})
    with pytest.raises(ToolExecutionError):
        router.execute("run_sql_query", record, {"sql": "SELECT * FROM nonexistent_table"})


def test_unknown_tool_name_is_rejected(router, record):
    with pytest.raises(ToolExecutionError):
        router.execute("not_a_real_tool", record, {})


# --- 4. Dangerous SQL remains blocked (both engines) ----------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE dataset",
        "INSERT INTO dataset (revenue) VALUES (1)",
        "UPDATE dataset SET revenue = 0",
        "DELETE FROM dataset",
        "CREATE TABLE evil (x INT)",
        "SELECT 1; DROP TABLE dataset;",
        "ATTACH 'evil.db' AS evil",
        "PRAGMA table_info(dataset)",
    ],
)
@pytest.mark.parametrize("engine", ["duckdb", "sqlite"])
def test_dangerous_sql_blocked_via_tool_router(router, record, sql, engine):
    with pytest.raises(ToolExecutionError):
        router.execute("run_sql_query", record, {"sql": sql, "engine": engine})


def test_safe_select_still_works_via_tool_router(router, record):
    result = router.execute(
        "run_sql_query", record, {"sql": "SELECT region, COUNT(*) AS n FROM dataset GROUP BY region"}
    )
    assert result["row_count"] > 0
    assert set(result["columns"]) == {"region", "n"}


# --- 5. Resource limits remain active -------------------------------------------


def test_sql_tool_row_cap_is_tighter_than_engine_default(router, record):
    # record fixture has 200 rows, well under both the engine default (10,000) and the
    # tool-level cap (500) -- confirms the cap is real infrastructure, not just documented.
    assert tool_router._SQL_TOOL_MAX_ROWS < 10_000
    result = router.execute("run_sql_query", record, {"sql": "SELECT * FROM dataset"})
    assert result["row_count"] == len(record.df)
    assert result["truncated"] is False


def test_sql_tool_truncates_when_result_exceeds_cap(router):
    big_df = pd.DataFrame({"n": range(tool_router._SQL_TOOL_MAX_ROWS + 100)})
    big_record = DatasetRecord(
        id="big",
        original_filename="big.csv",
        extension=".csv",
        uploaded_at=pd.Timestamp.utcnow(),
        df=big_df,
        stored_path="unused",
    )
    result = router.execute("run_sql_query", big_record, {"sql": "SELECT * FROM dataset"})
    assert result["row_count"] == tool_router._SQL_TOOL_MAX_ROWS
    assert result["truncated"] is True


# --- 6. Tool results remain marked as untrusted data ----------------------------


@pytest.mark.parametrize(
    "tool_name,args",
    [
        ("automated_eda", {}),
        ("forecast", {"date_column": "date", "value_column": "revenue", "periods": 3}),
        ("run_sql_query", {"sql": "SELECT region, SUM(revenue) AS total FROM dataset GROUP BY region"}),
        ("rfm_analysis", {"customer_column": "customer_id", "date_column": "date", "value_column": "revenue"}),
    ],
)
def test_new_tool_results_carry_untrusted_data_marker(record, tool_name, args):
    script = [
        ProviderResponse(content=None, tool_calls=[ToolCall(id="call_1", name=tool_name, arguments=args)]),
        ProviderResponse(content="Done."),
    ]
    provider = MockProvider(script)
    agent = DataAnalystAgent(provider)

    agent.ask(record, f"Please run {tool_name}.")

    tool_message = next(m for m in provider.calls[1] if m.get("role") == "tool")
    assert tool_message["content"].startswith(_UNTRUSTED_DATA_MARKER)


# --- 7. Integration: agent can actually select and use a newly-registered tool --


def test_agent_can_discover_and_execute_a_newly_registered_forecasting_tool(record):
    """End-to-end proof that a Wave 2 tool -- previously built but unreachable from the
    LLM -- is now both listed in the tool catalog the model sees AND actually callable
    through the full DataAnalystAgent.ask() loop, not just present in TOOL_SCHEMAS."""
    script = [
        ProviderResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="forecast",
                    arguments={"date_column": "date", "value_column": "revenue", "periods": 5},
                )
            ],
        ),
        ProviderResponse(content="Forecast complete."),
    ]
    provider = MockProvider(script)
    agent = DataAnalystAgent(provider)

    result = agent.ask(record, "Forecast the next 5 days of revenue.")

    # The tool catalog offered to the model on the first turn must include the tool
    # the (scripted) model chose -- proves it was actually selectable, not hardcoded.
    first_turn_tool_names = {t["function"]["name"] for t in provider.tools_per_call[0]}
    assert "forecast" in first_turn_tool_names

    assert len(result["tool_calls"]) == 1
    call_record = result["tool_calls"][0]
    assert call_record.tool == "forecast"
    assert call_record.result["method_used"] in {"arima", "ets"}
    assert len(call_record.result["forecast"]) == 5


def test_agent_can_choose_among_several_new_analytical_capabilities(record):
    """Verifies several distinct new capability classes (stats, clustering, SQL) are
    each independently reachable through the same agent loop -- not just one tool
    wired up as a special case."""
    for tool_name, args in [
        ("t_test", {"column": "revenue", "popmean": 400}),
        ("kmeans_cluster", {"columns": ["revenue", "cost", "quantity"]}),
        ("run_sql_query", {"sql": "SELECT COUNT(*) AS n FROM dataset"}),
    ]:
        script = [
            ProviderResponse(content=None, tool_calls=[ToolCall(id="call_1", name=tool_name, arguments=args)]),
            ProviderResponse(content="Done."),
        ]
        provider = MockProvider(script)
        agent = DataAnalystAgent(provider)
        result = agent.ask(record, f"Use {tool_name}.")
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0].tool == tool_name
        assert "error" not in json_result_keys(result["tool_calls"][0].result)


def json_result_keys(result: dict) -> set:
    return set(result.keys())
