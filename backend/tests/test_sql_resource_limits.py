"""P0 remediation: query timeout and memory-limit enforcement for both SQL
backends (see .agent/decisions.md — the orchestrator's cross-review of
SQL-ENGINEER's Wave 1 output against the threat model found this was the one
missing checklist item: "query timeout and resource limits").

Every test here uses a deliberately expensive-but-syntactically-valid SELECT
(a large cross join / forced full materialization) — the exact attack shape
the threat model describes: not blocked by the statement-type/read-only
checks because it IS a legitimate SELECT, just an unboundedly costly one.
"""

from __future__ import annotations

import time

import pandas as pd
import pytest

from app.sql.duckdb_source import DuckDBDataSource
from app.sql.sqlite_source import SQLiteDataSource
from app.tools.errors import ToolExecutionError

# Calibrated (see .agent/decisions.md) so the expensive query can't be
# short-circuited by query-planner LIMIT pushdown — both engines' optimizers
# can stop early on a plain `SELECT *` cross join once enough rows exist to
# satisfy our own `LIMIT max_rows+1` wrapper, so these use a COUNT(*) (which
# can't return until every row has actually been produced) instead.
_ROWS = 12_000  # 12,000 x 12,000 = 144M rows — ~1.4s for SQLite's count(*),
# comfortably over a 1s timeout with margin.


@pytest.fixture
def big_df() -> pd.DataFrame:
    return pd.DataFrame({"id": range(_ROWS), "value": [i % 97 for i in range(_ROWS)]})


# DuckDB is columnar/vectorized and processes even a 144M-row cross join of
# `big_df` in milliseconds (verified empirically) — range() x range() at this
# size reliably takes ~2.4s regardless of loaded data, so the DuckDB timeout
# tests use this instead and don't depend on big_df's size at all.
_DUCKDB_EXPENSIVE_QUERY = "SELECT count(*) FROM range(100000) t1, range(100000) t2"
_SQLITE_EXPENSIVE_QUERY = "SELECT count(*) FROM dataset a, dataset b"

# A plain `SELECT * ... LIMIT n` cross join gets optimized to stop early and
# never hits a low memory_limit (verified empirically) — an ORDER BY forces
# full materialization/sort before any row can be returned, which does.
_DUCKDB_MEMORY_HEAVY_QUERY = "SELECT * FROM range(50000000) t1, range(10) t2 ORDER BY 1"


class TestDuckDBTimeout:
    def test_expensive_query_is_cancelled_within_timeout(self, big_df):
        source = DuckDBDataSource(big_df, timeout_seconds=1.0)
        try:
            start = time.monotonic()
            with pytest.raises(ToolExecutionError, match="time limit"):
                source.execute_query(_DUCKDB_EXPENSIVE_QUERY)
            elapsed = time.monotonic() - start
            # Generous upper bound: the watchdog fires at ~1s, DuckDB needs a
            # moment to actually unwind — assert it's cancelled promptly, not
            # that it ran to completion (which would take ~2.4s) and only
            # then coincidentally raised for some other reason.
            assert elapsed < 5.0
        finally:
            source.close()

    def test_fast_query_unaffected_by_timeout(self, big_df):
        """The timeout must not fire early / interfere with normal queries."""
        source = DuckDBDataSource(big_df, timeout_seconds=30.0)
        try:
            result = source.execute_query("SELECT count(*) AS n FROM dataset")
            assert result.rows[0]["n"] == _ROWS
        finally:
            source.close()

    def test_explain_does_not_hang_on_an_expensive_query(self, big_df):
        """EXPLAIN (without ANALYZE) only plans the query — cheap regardless
        of the plan's eventual cost — so it must complete quickly rather than
        wait for/trigger the timeout, proving the timeout wraps actual
        execution, not merely "any call into execute()"."""
        source = DuckDBDataSource(big_df, timeout_seconds=1.0)
        try:
            result = source.explain(_DUCKDB_EXPENSIVE_QUERY)
            assert result.row_count > 0
        finally:
            source.close()

    def test_memory_limit_produces_clean_tool_error(self):
        """A query whose intermediate result exceeds memory_limit must fail
        with a clean ToolExecutionError, not an unhandled crash. Verified
        empirically that DuckDB can surface this as either a
        duckdb.OutOfMemoryException OR a plain RuntimeError depending on
        exactly where allocation fails — both must produce the same clean
        message (see duckdb_source.py's _is_memory_limit_error).

        Uses a tiny dataset (not `big_df`) so the 1MB limit is exceeded by
        the QUERY, not by loading the dataset itself — that's a separate
        case, see test_memory_limit_also_applies_to_initial_data_load."""
        tiny_df = pd.DataFrame({"id": range(10), "value": range(10)})
        source = DuckDBDataSource(tiny_df, memory_limit="1MB", timeout_seconds=30.0)
        try:
            with pytest.raises(ToolExecutionError, match="[Mm]emory"):
                source.execute_query(_DUCKDB_MEMORY_HEAVY_QUERY)
        finally:
            source.close()

    def test_memory_limit_also_applies_to_initial_data_load(self, big_df):
        """A too-small memory_limit relative to the dataset being loaded must
        fail cleanly at construction time too, not just at query time — the
        write connection that loads the DataFrame is subject to the same
        memory_limit as the read-only query connection."""
        with pytest.raises(ToolExecutionError, match="[Mm]emory|large"):
            DuckDBDataSource(big_df, memory_limit="1MB", timeout_seconds=30.0)

    def test_timeout_does_not_weaken_read_only_enforcement(self, big_df):
        """Sanity check: adding resource limits must not have disturbed the
        existing read-only guarantees (full adversarial coverage already
        lives in test_sql_security.py; this is a smoke check that the two
        concerns compose correctly)."""
        source = DuckDBDataSource(big_df, timeout_seconds=1.0)
        try:
            with pytest.raises(ToolExecutionError):
                source.execute_query("DROP TABLE dataset")
            # Data must still be intact and queryable after the blocked attempt.
            result = source.execute_query("SELECT count(*) AS n FROM dataset")
            assert result.rows[0]["n"] == _ROWS
        finally:
            source.close()


class TestSQLiteTimeout:
    def test_expensive_query_is_cancelled_within_timeout(self, big_df):
        source = SQLiteDataSource(big_df, timeout_seconds=1.0)
        try:
            start = time.monotonic()
            with pytest.raises(ToolExecutionError, match="time limit"):
                source.execute_query(_SQLITE_EXPENSIVE_QUERY)
            elapsed = time.monotonic() - start
            assert elapsed < 5.0
        finally:
            source.close()

    def test_fast_query_unaffected_by_timeout(self, big_df):
        source = SQLiteDataSource(big_df, timeout_seconds=30.0)
        try:
            result = source.execute_query("SELECT count(*) AS n FROM dataset")
            assert result.rows[0]["n"] == _ROWS
        finally:
            source.close()

    def test_explain_does_not_hang_on_an_expensive_query(self, big_df):
        source = SQLiteDataSource(big_df, timeout_seconds=1.0)
        try:
            result = source.explain(_SQLITE_EXPENSIVE_QUERY)
            assert result.row_count > 0
        finally:
            source.close()

    def test_repeated_queries_each_get_a_fresh_deadline(self, big_df):
        """The deadline is per-query (reset before each execute), not a single
        deadline from construction — a slow/timed-out query must not "use up"
        timeout budget that a later fast query then inherits."""
        source = SQLiteDataSource(big_df, timeout_seconds=1.0)
        try:
            with pytest.raises(ToolExecutionError, match="time limit"):
                source.execute_query(_SQLITE_EXPENSIVE_QUERY)
            # If the deadline weren't reset, this would immediately fail too.
            result = source.execute_query("SELECT count(*) AS n FROM dataset")
            assert result.rows[0]["n"] == _ROWS
        finally:
            source.close()

    def test_timeout_does_not_weaken_read_only_enforcement(self, big_df):
        source = SQLiteDataSource(big_df, timeout_seconds=1.0)
        try:
            with pytest.raises(ToolExecutionError):
                source.execute_query("DROP TABLE dataset")
            result = source.execute_query("SELECT count(*) AS n FROM dataset")
            assert result.rows[0]["n"] == _ROWS
        finally:
            source.close()

    def test_soft_heap_limit_is_configured_without_error(self, big_df):
        """Best-effort check that the pragma was accepted at construction time.
        SQLite's soft_heap_limit is advisory/process-global (not a hard
        per-connection cap the way DuckDB's memory_limit is), and the value
        can't be read back through this connection (the authorizer blocks
        PRAGMA reads too, by design) — so successful construction with a
        custom value, with no exception raised, IS the assertion."""
        source = SQLiteDataSource(big_df, soft_heap_limit_bytes=64 * 1024 * 1024)
        try:
            result = source.execute_query("SELECT count(*) AS n FROM dataset")
            assert result.rows[0]["n"] == _ROWS
        finally:
            source.close()
