"""Sanity checks for the forward-looking `tmp_sqlite_db` / `tmp_duckdb_conn` fixtures
added to conftest.py for the upcoming SQL-ENGINEER / LARGE-DATA-ENGINEER tracks.

Nothing in this worktree consumes those fixtures yet (the SQL and large-data layers
are being built in separate Wave 1 worktrees this agent cannot see) - these tests only
prove the fixtures themselves work: they produce a queryable table with the expected
row count.
"""
from __future__ import annotations

import sqlite3

import pandas as pd


def test_tmp_sqlite_db_is_queryable_with_expected_row_count(tmp_sqlite_db: str, sample_df: pd.DataFrame) -> None:
    conn = sqlite3.connect(tmp_sqlite_db)
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM sales")
        (row_count,) = cursor.fetchone()
        assert row_count == len(sample_df)

        cols = [row[1] for row in conn.execute("PRAGMA table_info(sales)").fetchall()]
        assert set(cols) == set(sample_df.columns)
    finally:
        conn.close()


def test_tmp_duckdb_conn_is_queryable_with_expected_row_count(tmp_duckdb_conn, sample_df: pd.DataFrame) -> None:
    (row_count,) = tmp_duckdb_conn.execute("SELECT COUNT(*) FROM sales").fetchone()
    assert row_count == len(sample_df)

    cols = [row[0] for row in tmp_duckdb_conn.execute("DESCRIBE sales").fetchall()]
    assert set(cols) == set(sample_df.columns)


def test_tmp_duckdb_conn_supports_aggregation(tmp_duckdb_conn) -> None:
    """A minimal proof the connection supports real SQL, not just a row-count probe -
    something a future SQL-ENGINEER test will lean on heavily."""
    (total_revenue,) = tmp_duckdb_conn.execute("SELECT SUM(revenue) FROM sales").fetchone()
    assert total_revenue is not None
    assert total_revenue > 0
