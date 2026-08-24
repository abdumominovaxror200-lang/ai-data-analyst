from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.sql import QueryResult, create_data_source
from app.sql.duckdb_source import DuckDBDataSource
from app.sql.sqlite_source import SQLiteDataSource
from app.tools.errors import ToolExecutionError

ENGINES = ["duckdb", "sqlite"]


@pytest.fixture
def orders_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": range(1, 11),
            "region": ["North", "South"] * 5,
            "amount": [10, 20, 30, 40, 50, 15, 25, 35, 45, 55],
        }
    )


@pytest.fixture(params=ENGINES)
def engine(request) -> str:
    return request.param


def make_ds(df: pd.DataFrame, engine: str, **kwargs):
    ds = create_data_source(df, engine=engine, **kwargs)
    yield ds
    ds.close()


@pytest.fixture
def orders_ds(orders_df, engine):
    yield from make_ds(orders_df, engine)


# --- Factory / construction -------------------------------------------------


def test_create_data_source_defaults_to_duckdb(orders_df):
    ds = create_data_source(orders_df)
    try:
        assert isinstance(ds, DuckDBDataSource)
    finally:
        ds.close()


def test_create_data_source_unknown_engine_raises(orders_df):
    with pytest.raises(ValueError):
        create_data_source(orders_df, engine="postgres")


def test_data_source_satisfies_provisional_protocol(orders_df):
    from app.sql.models import DataSource

    ds = create_data_source(orders_df, engine="duckdb")
    try:
        assert isinstance(ds, DataSource)
        result = ds.execute_query("SELECT * FROM dataset")
        assert isinstance(result, QueryResult)
    finally:
        ds.close()


# --- Correctness: basic query shapes, parametrized across both engines -----


def test_basic_select_returns_all_rows(orders_ds, orders_df):
    result = orders_ds.execute_query("SELECT id, region, amount FROM dataset")
    assert result.columns == ["id", "region", "amount"]
    assert result.row_count == len(orders_df)
    assert result.truncated is False
    assert {row["id"] for row in result.rows} == set(orders_df["id"])


def test_where_filter(orders_ds):
    result = orders_ds.execute_query("SELECT id FROM dataset WHERE amount > 40")
    assert {row["id"] for row in result.rows} == {5, 9, 10}


def test_group_by_aggregation(orders_ds):
    result = orders_ds.execute_query(
        "SELECT region, SUM(amount) AS total, COUNT(*) AS n FROM dataset GROUP BY region ORDER BY region"
    )
    by_region = {row["region"]: row for row in result.rows}
    assert by_region["North"]["total"] == 160
    assert by_region["South"]["total"] == 165
    assert by_region["North"]["n"] == 5


def test_cte_and_window_function(orders_ds):
    result = orders_ds.execute_query(
        """
        WITH ranked AS (
            SELECT id, region, amount,
                   RANK() OVER (PARTITION BY region ORDER BY amount DESC) AS rnk
            FROM dataset
        )
        SELECT id, region, amount FROM ranked WHERE rnk = 1 ORDER BY region
        """
    )
    assert result.rows == [
        {"id": 5, "region": "North", "amount": 50},
        {"id": 10, "region": "South", "amount": 55},
    ]


def test_self_join(orders_ds):
    result = orders_ds.execute_query(
        "SELECT COUNT(*) AS n FROM dataset a JOIN dataset b ON a.region = b.region AND a.id < b.id"
    )
    assert result.rows[0]["n"] == 20  # 2 groups of 5 -> C(5,2)=10 pairs each


def test_subquery(orders_ds):
    # North avg = (10+30+50+25+45)/5 = 32, South avg = (20+40+15+35+55)/5 = 33
    result = orders_ds.execute_query(
        "SELECT * FROM (SELECT region, AVG(amount) AS avg_amt FROM dataset GROUP BY region) sub "
        "WHERE avg_amt > 32.5 ORDER BY region"
    )
    assert result.rows == [{"region": "South", "avg_amt": 33.0}]


def test_trailing_semicolon_is_tolerated(orders_ds):
    result = orders_ds.execute_query("SELECT COUNT(*) AS n FROM dataset;")
    assert result.rows[0]["n"] == 10


# --- Truncation --------------------------------------------------------


def test_truncation_flag_set_when_over_max_rows(orders_df, engine):
    for ds in make_ds(orders_df, engine, max_rows=3):
        result = ds.execute_query("SELECT * FROM dataset ORDER BY id")
        assert result.truncated is True
        assert result.row_count == 3
        assert len(result.rows) == 3


def test_truncation_flag_false_when_under_limit(orders_ds):
    result = orders_ds.execute_query("SELECT * FROM dataset")
    assert result.truncated is False


# --- Multi-table registration (forward-compatible with joins across datasets) --


def test_dict_of_dataframes_registers_multiple_tables(engine):
    customers = pd.DataFrame({"cust_id": [1, 2], "name": ["Alice", "Bob"]})
    orders = pd.DataFrame({"order_id": [100, 101], "cust_id": [1, 2], "amount": [50, 75]})
    for ds in make_ds({"customers": customers, "orders": orders}, engine):
        result = ds.execute_query(
            "SELECT c.name, o.amount FROM customers c JOIN orders o ON c.cust_id = o.cust_id ORDER BY c.name"
        )
        assert result.rows == [
            {"name": "Alice", "amount": 50},
            {"name": "Bob", "amount": 75},
        ]


def test_custom_default_table_name(engine):
    df = pd.DataFrame({"x": [1, 2, 3]})
    for ds in make_ds(df, engine, default_table="my_table"):
        result = ds.execute_query("SELECT SUM(x) AS total FROM my_table")
        assert result.rows[0]["total"] == 6


# --- EXPLAIN -----------------------------------------------------------


def test_explain_returns_a_plan_without_executing_mutation(orders_ds):
    result = orders_ds.explain("SELECT * FROM dataset WHERE amount > 20")
    assert result.row_count >= 1
    assert len(result.columns) >= 1
    # plan text/rows should mention the table somewhere
    plan_text = " ".join(str(v) for row in result.rows for v in row.values()).lower()
    assert "dataset" in plan_text


def test_explain_rejects_non_select(orders_ds):
    with pytest.raises(ToolExecutionError):
        orders_ds.explain("DROP TABLE dataset")


# --- Clean error handling (ToolExecutionError, not raw driver exceptions) --


def test_unknown_table_raises_tool_execution_error(orders_ds):
    with pytest.raises(ToolExecutionError):
        orders_ds.execute_query("SELECT * FROM nosuchtable")


def test_unknown_column_raises_tool_execution_error(orders_ds):
    with pytest.raises(ToolExecutionError):
        orders_ds.execute_query("SELECT bogus_col FROM dataset")


def test_syntax_error_raises_tool_execution_error(orders_ds):
    with pytest.raises(ToolExecutionError):
        orders_ds.execute_query("SELECT * FROM dataset WHERE (")


def test_empty_query_raises_tool_execution_error(orders_ds):
    with pytest.raises(ToolExecutionError):
        orders_ds.execute_query("")
    with pytest.raises(ToolExecutionError):
        orders_ds.execute_query("   ")


def test_invalid_table_name_rejected(engine):
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ToolExecutionError):
        next(make_ds(df, engine, default_table="bad; name"))


def test_closed_data_source_raises_on_query(orders_df, engine):
    ds = create_data_source(orders_df, engine=engine)
    ds.close()
    with pytest.raises(ToolExecutionError):
        ds.execute_query("SELECT 1")


def test_context_manager_closes(orders_df, engine):
    with create_data_source(orders_df, engine=engine) as ds:
        result = ds.execute_query("SELECT COUNT(*) AS n FROM dataset")
        assert result.rows[0]["n"] == len(orders_df)
    with pytest.raises(ToolExecutionError):
        ds.execute_query("SELECT 1")


# --- Result shape / JSON safety (datetime columns, per dataframe_to_records) --


def test_datetime_column_is_json_safe(engine):
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "created_at": pd.to_datetime(["2024-01-01", "2024-06-15"]),
        }
    )
    for ds in make_ds(df, engine):
        result = ds.execute_query("SELECT * FROM dataset ORDER BY id")
        import json

        json.dumps(result.rows)  # must not raise
        assert result.rows[0]["created_at"].startswith("2024-01-01")


def test_null_values_become_none(engine):
    df = pd.DataFrame({"id": [1, 2], "v": [1.0, np.nan]})
    for ds in make_ds(df, engine):
        result = ds.execute_query("SELECT * FROM dataset ORDER BY id")
        assert result.rows[1]["v"] is None
