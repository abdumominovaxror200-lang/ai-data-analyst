from __future__ import annotations

import io
import sqlite3

import duckdb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings


@pytest.fixture(autouse=True)
def _isolated_dataset_store(tmp_path, monkeypatch):
    """Fresh dataset store + storage dir per test, so tests never see each other's data."""
    import app.datasets.storage as storage_module

    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))
    get_settings.cache_clear()
    storage_module._store = None
    yield
    storage_module._store = None
    get_settings.cache_clear()


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    return TestClient(app)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 100
    revenue = rng.normal(1000, 100, n)
    revenue[0] = 50_000  # intentional outlier
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "region": rng.choice(["North", "South", "East", "West"], n),
            "product": rng.choice(["Widget", "Gadget", "Gizmo"], n),
            "quantity": rng.integers(1, 50, n),
            "revenue": revenue,
            "cost": revenue * rng.uniform(0.4, 0.6, n),
        }
    )


def make_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def make_xlsx_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    return buffer.getvalue()


# --- Forward-looking fixtures for upcoming SQL/large-data work -------------------
#
# Not consumed by any test in this worktree yet - added ahead of time so the
# SQL-ENGINEER and LARGE-DATA-ENGINEER tracks (built in separate Wave 1 worktrees)
# have a ready-made, known-good way to get a small DB-backed table to test against
# once their code is integrated. Additive only; existing fixtures above are untouched.


@pytest.fixture
def tmp_sqlite_db(tmp_path, sample_df: pd.DataFrame) -> str:
    """Creates a temp SQLite file pre-loaded with `sample_df` in a table named
    'sales'. Yields the filesystem path (str) to the .db file. The connection is
    closed before yielding so callers get a fresh connection of their own, and the
    underlying tmp_path directory is cleaned up automatically by pytest afterward."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    try:
        sample_df.to_sql("sales", conn, index=False, if_exists="replace")
        conn.commit()
    finally:
        conn.close()
    yield str(db_path)


@pytest.fixture
def tmp_duckdb_conn(sample_df: pd.DataFrame):
    """Creates an in-memory DuckDB connection pre-loaded with `sample_df` in a table
    named 'sales'. Yields the live `duckdb.DuckDBPyConnection`; closes it on
    teardown."""
    conn = duckdb.connect(":memory:")
    try:
        conn.register("sample_df_view", sample_df)
        conn.execute("CREATE TABLE sales AS SELECT * FROM sample_df_view")
        conn.unregister("sample_df_view")
        yield conn
    finally:
        conn.close()
