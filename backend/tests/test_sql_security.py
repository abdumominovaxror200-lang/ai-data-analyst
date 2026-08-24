"""Adversarial security tests for the read-only SQL layer.

Every test in this file represents an attempt to make a supposedly read-only
`DataSource` mutate data, exfiltrate a file, or otherwise escape the
read-only sandbox. Every one of them MUST raise `ToolExecutionError` (never a
raw duckdb/sqlite3 exception, never silently succeed) and MUST leave the
underlying data untouched.

See the docstrings on `DuckDBDataSource` / `SQLiteDataSource` for exactly
which access-control mechanism blocks which class of attack.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from app.sql import create_data_source
from app.tools.errors import ToolExecutionError

ENGINES = ["duckdb", "sqlite"]


@pytest.fixture(params=ENGINES)
def engine(request) -> str:
    return request.param


@pytest.fixture
def seed_df() -> pd.DataFrame:
    return pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})


@pytest.fixture
def ds(seed_df, engine):
    data_source = create_data_source(seed_df, engine=engine)
    yield data_source
    data_source.close()


def _assert_blocked_and_intact(ds, sql: str) -> None:
    with pytest.raises(ToolExecutionError):
        ds.execute_query(sql)
    # Data must be completely unchanged after every blocked attempt.
    result = ds.execute_query("SELECT COUNT(*) AS n FROM dataset")
    assert result.rows[0]["n"] == 3


# --- 1. Direct DML/DDL, one statement at a time -----------------------


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO dataset VALUES (4, 'd')",
        "UPDATE dataset SET name = 'x'",
        "DELETE FROM dataset",
        "DROP TABLE dataset",
        "CREATE TABLE evil (x INT)",
        "ALTER TABLE dataset ADD COLUMN y INT",
        "TRUNCATE TABLE dataset",
    ],
)
def test_direct_mutation_blocked(ds, sql):
    _assert_blocked_and_intact(ds, sql)


# --- 2. Statement stacking (multiple statements in one call) ----------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1; DROP TABLE dataset;",
        "SELECT * FROM dataset; DELETE FROM dataset;",
        "SELECT 1 /* harmless comment */; DROP TABLE dataset; --",
        "SELECT * FROM dataset -- trailing comment hides nothing\n; DROP TABLE dataset;",
        "SELECT * FROM dataset; SELECT * FROM dataset",  # two harmless SELECTs: still blocked
    ],
)
def test_statement_stacking_blocked(ds, sql):
    _assert_blocked_and_intact(ds, sql)


# --- 3. Mutation smuggled inside a subquery / CTE ----------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM (INSERT INTO dataset VALUES (4, 'd')) AS x",
        "WITH x AS (INSERT INTO dataset VALUES (4, 'd') RETURNING *) SELECT * FROM x",
        "WITH x AS (DELETE FROM dataset RETURNING *) SELECT * FROM x",
        "WITH x AS (UPDATE dataset SET name = 'z' RETURNING *) SELECT * FROM x",
        "SELECT * FROM (DROP TABLE dataset)",
    ],
)
def test_mutation_hidden_in_subquery_or_cte_blocked(ds, sql):
    _assert_blocked_and_intact(ds, sql)


# --- 4. Session/catalog-level escapes -----------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "ATTACH 'C:/temp/evil.db' AS evil",
        "ATTACH DATABASE 'C:/temp/evil.db' AS evil",
        "DETACH dataset",
        "PRAGMA memory_limit='100MB'",
        "PRAGMA query_only = OFF",
        "SET memory_limit='100MB'",
        "VACUUM",
        "CALL pragma_table_info('dataset')",
        "BEGIN TRANSACTION",
        "COMMIT",
    ],
)
def test_session_and_catalog_escapes_blocked(ds, sql):
    _assert_blocked_and_intact(ds, sql)


# --- 5. EXPLAIN cannot be used to smuggle a mutation --------------------


@pytest.mark.parametrize(
    "sql",
    [
        "EXPLAIN DROP TABLE dataset",
        "EXPLAIN ANALYZE DROP TABLE dataset",
        "EXPLAIN INSERT INTO dataset VALUES (4, 'd')",
    ],
)
def test_explain_of_mutation_blocked_via_execute_query(ds, sql):
    # A caller should use .explain() for cost visibility on a SELECT; passing
    # an EXPLAIN-wrapped mutation to execute_query() must still be rejected.
    _assert_blocked_and_intact(ds, sql)


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE dataset",
        "INSERT INTO dataset VALUES (4, 'd')",
    ],
)
def test_explain_method_rejects_mutations(ds, sql):
    with pytest.raises(ToolExecutionError):
        ds.explain(sql)
    result = ds.execute_query("SELECT COUNT(*) AS n FROM dataset")
    assert result.rows[0]["n"] == 3


# --- 6. Filesystem exfiltration via COPY --------------------------------
#
# Empirically, DuckDB's read_only=True connection mode does NOT prevent
# `COPY <table> TO '<path>'` — it writes a *new* file, not a write to the
# attached database, so the engine-level read-only guard alone misses it.
# This makes the statement-type allowlist load bearing for this specific
# case, not merely a redundant layer.


def test_copy_to_file_is_blocked_and_no_file_written(ds, tmp_path):
    target = tmp_path / "exfil.csv"
    sql = f"COPY dataset TO '{target.as_posix()}'"
    _assert_blocked_and_intact(ds, sql)
    assert not target.exists()


def test_export_database_blocked(ds, tmp_path):
    target = tmp_path / "exported_db"
    sql = f"EXPORT DATABASE '{target.as_posix()}'"
    _assert_blocked_and_intact(ds, sql)
    assert not target.exists()


# --- 7. Arbitrary local file read via table-valued functions (DuckDB) --
#
# `SELECT * FROM read_csv_auto(...)` is a syntactically ordinary SELECT, so
# neither the read-only connection nor the statement-type allowlist blocks
# it on its own — verified empirically that this reads real files off disk
# (e.g. a Windows system file) if not separately denylisted. This is the
# one gap that is a denylist by construction (see validation.py) and is
# therefore NOT exhaustively future-proof — documented as a known residual
# risk in the task report.


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM read_csv_auto('C:/Windows/win.ini')",
        "SELECT * FROM read_csv('C:/Windows/win.ini')",
        "SELECT * FROM read_parquet('C:/some/file.parquet')",
        "SELECT * FROM glob('C:/*')",
    ],
)
def test_file_reading_table_functions_blocked(ds, sql):
    _assert_blocked_and_intact(ds, sql)


# --- 8. Empty / degenerate input -----------------------------------------


@pytest.mark.parametrize("sql", ["", "   ", ";", "-- just a comment"])
def test_empty_or_degenerate_query_blocked(ds, sql):
    with pytest.raises(ToolExecutionError):
        ds.execute_query(sql)


# --- 9. SQLite-specific: authorizer cannot be disabled from within SQL --


def test_sqlite_query_only_pragma_cannot_be_turned_off():
    df = pd.DataFrame({"id": [1, 2, 3]})
    with create_data_source(df, engine="sqlite") as sqlite_ds:
        # PRAGMA query_only = OFF is itself blocked by the authorizer (a
        # PRAGMA action), so the write guard cannot be disabled from SQL.
        with pytest.raises(ToolExecutionError):
            sqlite_ds.execute_query("PRAGMA query_only = OFF")
        with pytest.raises(ToolExecutionError):
            sqlite_ds.execute_query("INSERT INTO dataset VALUES (4)")


# --- 10. Read-only enforcement survives across many queries on one source --


def test_repeated_attacks_never_succeed_across_many_calls(ds):
    attacks = [
        "DROP TABLE dataset",
        "INSERT INTO dataset VALUES (4, 'd')",
        "SELECT 1; DROP TABLE dataset;",
        "DELETE FROM dataset",
    ]
    for _ in range(3):
        for sql in attacks:
            with pytest.raises(ToolExecutionError):
                ds.execute_query(sql)
    result = ds.execute_query("SELECT COUNT(*) AS n FROM dataset")
    assert result.rows[0]["n"] == 3


# --- 11. No raw driver exception ever escapes execute_query/explain -----


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE dataset",
        "SELECT 1; DROP TABLE dataset;",
        "SELECT * FROM nosuchtable",
        "SELECT bogus_col FROM dataset",
        "SELECT * FROM dataset WHERE (",
        "ATTACH 'x.db' AS x",
    ],
)
def test_only_tool_execution_error_type_ever_raised(ds, sql):
    try:
        ds.execute_query(sql)
        pytest.fail(f"expected ToolExecutionError for: {sql!r}")
    except ToolExecutionError:
        pass  # exactly what must happen
    except Exception as exc:  # pragma: no cover - failure path
        pytest.fail(
            f"raw driver exception leaked instead of ToolExecutionError for {sql!r}: "
            f"{type(exc).__name__}: {exc}"
        )
