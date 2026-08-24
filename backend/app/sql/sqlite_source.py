from __future__ import annotations

import sqlite3
import threading

import pandas as pd

from app.tools.errors import ToolExecutionError
from app.tools.serialization import dataframe_to_records

from .models import QueryResult
from .validation import quick_reject_non_select, validate_identifier

DEFAULT_MAX_ROWS = 10_000

_ALLOWED_AUTHORIZER_ACTIONS = frozenset(
    {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
)


def _authorizer(action: int, arg1, arg2, dbname, source) -> int:
    if action in _ALLOWED_AUTHORIZER_ACTIONS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


class SQLiteDataSource:
    """Read-only SQL query engine over a pandas DataFrame, backed by SQLite.

    Lighter second backend than DuckDB (per `.agent/decisions.md`), no extra
    dependency (stdlib `sqlite3`). Same `QueryResult`-returning surface as
    `DuckDBDataSource`.

    Read-only enforcement, layered:

    1. **Engine-level (primary): `sqlite3.Connection.set_authorizer`.** This
       installs SQLite's own C-level access-control callback, invoked for
       every action SQLite is about to perform while *preparing* a
       statement — not a post-hoc string check. Only `SQLITE_SELECT`,
       `SQLITE_READ`, and `SQLITE_FUNCTION` are allowed; everything else
       (INSERT, UPDATE, DELETE, CREATE_TABLE/INDEX, DROP_*, ALTER_TABLE,
       ATTACH, PRAGMA, TRANSACTION, ...) is denied before it can run.
       Verified empirically that even `PRAGMA query_only = OFF` — an attempt
       to turn off layer 2 below — is itself denied by the authorizer,
       because issuing a PRAGMA is a `SQLITE_PRAGMA` action.
    2. **`PRAGMA query_only = ON`.** A second, independent engine-level
       guard. Must be set *before* `set_authorizer` is installed — the
       authorizer denies PRAGMA statements, including this one, once active.
       Confirmed empirically that `query_only` alone is bypassable
       (`PRAGMA query_only = OFF` succeeds if nothing else blocks it, and
       `ATTACH` is not covered by `query_only` at all) — hence layer 1 is
       the real control, and this is defense-in-depth, not sufficient alone.
    3. **`sqlite3.Connection.execute()` only ever runs one statement.**
       Verified empirically: `SELECT 1; DROP TABLE x;` raises
       `sqlite3.ProgrammingError: You can only execute one statement at a
       time.` — stacked-statement injection is rejected by the driver
       itself, no custom parsing needed.
    4. A light keyword pre-check (`quick_reject_non_select`) for a fast,
       clean `ToolExecutionError` before ever touching the connection.

    `sqlite3.enable_load_extension` is never called (it defaults to
    disabled), so `load_extension()` — which could otherwise pull in
    file/network-capable virtual tables — is unreachable here.
    """

    engine = "sqlite"

    def __init__(
        self,
        data: pd.DataFrame | dict[str, pd.DataFrame],
        default_table: str = "dataset",
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> None:
        self.max_rows = max_rows
        tables = {default_table: data} if isinstance(data, pd.DataFrame) else dict(data)
        if not tables:
            raise ToolExecutionError("At least one table is required.")
        for name in tables:
            validate_identifier(name)

        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        try:
            for name, df in tables.items():
                df.to_sql(name, self._conn, index=False, if_exists="fail")
            self._conn.commit()
        except Exception as exc:  # noqa: BLE001 - surfaced as a clean tool error
            self._conn.close()
            raise ToolExecutionError(f"Could not load data into SQLite: {exc}") from exc

        # Order matters: query_only must be set before the authorizer locks
        # PRAGMA out entirely.
        self._conn.execute("PRAGMA query_only = ON")
        self._conn.set_authorizer(_authorizer)

        self._lock = threading.Lock()
        self._closed = False

    def execute_query(self, sql: str) -> QueryResult:
        self._check_open()
        cleaned = self._validate(sql)
        wrapped = f"SELECT * FROM ({cleaned}) AS _query_result LIMIT {self.max_rows + 1}"
        with self._lock:
            try:
                cursor = self._conn.execute(wrapped)
                columns = [d[0] for d in cursor.description] if cursor.description else []
                rows = cursor.fetchall()
            except sqlite3.Error as exc:
                raise ToolExecutionError(_clean_error(exc)) from exc

        truncated = len(rows) > self.max_rows
        if truncated:
            rows = rows[: self.max_rows]
        df = pd.DataFrame(rows, columns=columns)
        return QueryResult(
            columns=columns,
            rows=dataframe_to_records(df),
            row_count=len(rows),
            truncated=truncated,
        )

    def explain(self, sql: str) -> QueryResult:
        """Return SQLite's query plan for `sql` without executing it, for
        cost visibility before a potentially expensive query runs."""
        self._check_open()
        cleaned = self._validate(sql)
        with self._lock:
            try:
                cursor = self._conn.execute(f"EXPLAIN QUERY PLAN {cleaned}")
                columns = [d[0] for d in cursor.description]
                rows = cursor.fetchall()
            except sqlite3.Error as exc:
                raise ToolExecutionError(_clean_error(exc)) from exc
        return QueryResult(
            columns=columns,
            rows=[dict(zip(columns, r)) for r in rows],
            row_count=len(rows),
            truncated=False,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._conn.close()
        self._closed = True

    def __enter__(self) -> "SQLiteDataSource":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self) -> None:  # best-effort cleanup
        try:
            self.close()
        except Exception:  # noqa: BLE001 - never raise from __del__
            pass

    def _check_open(self) -> None:
        if self._closed:
            raise ToolExecutionError("This SQL data source has been closed.")

    def _validate(self, sql: str) -> str:
        quick_reject_non_select(sql)
        return sql.strip().rstrip(";")


def _clean_error(exc: Exception) -> str:
    msg = str(exc)
    if msg == "not authorized":
        return (
            "SQL error: statement blocked by read-only policy (write, DDL, "
            "PRAGMA, and ATTACH operations are not permitted)."
        )
    return f"SQL error: {msg}"
