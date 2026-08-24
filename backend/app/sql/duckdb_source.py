from __future__ import annotations

import logging
import tempfile
import threading
import uuid
from pathlib import Path

import duckdb
import pandas as pd

from app.tools.errors import ToolExecutionError
from app.tools.serialization import dataframe_to_records

from .models import QueryResult
from .validation import (
    quick_reject_non_select,
    reject_blocked_functions,
    validate_identifier,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_ROWS = 10_000


class DuckDBDataSource:
    """Read-only DuckDB query engine over one or more pandas DataFrames.

    This is the primary backend (per `.agent/decisions.md`): supports JOIN,
    CTE, window functions, aggregation, and subqueries via real DuckDB SQL.

    Read-only enforcement is layered, per the task spec's preference for a
    real access-control mechanism over string-pattern matching alone:

    1. **Engine-level (primary).** The DataFrame(s) are written to a
       throwaway DuckDB file by a short-lived *writable* connection, which is
       then closed. All querying happens on a *second* connection opened with
       `read_only=True` against that same file. Verified empirically (see
       task report BENCHMARK/SECURITY sections): DuckDB itself refuses to run
       INSERT/UPDATE/DELETE/CREATE/DROP/ALTER/ATTACH(-writable) statements
       against a database attached in read-only mode, raising
       `InvalidInputException` before any mutation happens. Note DuckDB does
       NOT support `read_only=True` for `:memory:` databases — hence the
       throwaway file.
    2. **Statement-shape validation (defense-in-depth).** `conn.extract_statements`
       — DuckDB's own parser, not a regex — must return exactly one
       statement, of type SELECT. This blocks stacked-statement injection
       (`SELECT 1; DROP TABLE x;`) and blocks any DDL/DML/PRAGMA/SET/ATTACH/
       COPY/EXPORT/CALL/VACUUM statement by type, before it reaches the
       connection at all.
    3. **Function denylist (defense-in-depth for a real, confirmed gap).**
       DuckDB's read-only mode protects the *attached database*, not "can
       this query touch the filesystem." `SELECT * FROM read_csv_auto('C:/x')`
       is a syntactically ordinary SELECT and is NOT blocked by (1) or (2) —
       confirmed empirically. `reject_blocked_functions` blocks the known
       table-valued functions that read local files or attach other
       databases. This is a denylist and is therefore incomplete against a
       function DuckDB adds in a future version — see the task report's
       RISKS section.

    KNOWN GAP (documented, not silently swallowed): `COPY <table> TO
    '<path>'` was observed to still execute successfully against a
    `read_only=True` connection — it writes a *new* file, which is not a
    write to the *attached* database, so DuckDB's read-only check does not
    catch it. It is blocked here by (2), since COPY is not a SELECT
    statement. This means layer (2) is not optional/cosmetic — it is load
    bearing for this specific bypass.
    """

    engine = "duckdb"

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

        tmp_dir = Path(tempfile.gettempdir()) / "ai_data_analyst_sql"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = tmp_dir / f"ds_{uuid.uuid4().hex}.duckdb"

        writer = duckdb.connect(str(self._db_path))
        try:
            for name, df in tables.items():
                writer.register("_src", df)
                writer.execute(f'CREATE TABLE "{name}" AS SELECT * FROM _src')
                writer.unregister("_src")
        finally:
            writer.close()

        self._conn = duckdb.connect(str(self._db_path), read_only=True)
        self._lock = threading.Lock()
        self._closed = False

    def execute_query(self, sql: str) -> QueryResult:
        self._check_open()
        cleaned = self._validate(sql)
        wrapped = f"SELECT * FROM ({cleaned}) AS _query_result LIMIT {self.max_rows + 1}"
        with self._lock:
            try:
                df = self._conn.execute(wrapped).fetchdf()
            except duckdb.Error as exc:
                raise ToolExecutionError(_clean_error(exc)) from exc

        truncated = len(df) > self.max_rows
        if truncated:
            df = df.iloc[: self.max_rows]
        return QueryResult(
            columns=[str(c) for c in df.columns],
            rows=dataframe_to_records(df),
            row_count=int(len(df)),
            truncated=truncated,
        )

    def explain(self, sql: str) -> QueryResult:
        """Return DuckDB's query plan for `sql` without executing it, for
        cost visibility before a potentially expensive query runs."""
        self._check_open()
        cleaned = self._validate(sql)
        with self._lock:
            try:
                result = self._conn.execute(f"EXPLAIN {cleaned}")
                columns = [d[0] for d in result.description]
                rows = result.fetchall()
            except duckdb.Error as exc:
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
        try:
            self._db_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("could not remove temp duckdb file %s", self._db_path)

    def __enter__(self) -> "DuckDBDataSource":
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
        reject_blocked_functions(sql)
        try:
            statements = self._conn.extract_statements(sql)
        except duckdb.Error as exc:
            raise ToolExecutionError(_clean_error(exc)) from exc
        if len(statements) == 0:
            raise ToolExecutionError("Empty SQL query.")
        if len(statements) > 1:
            raise ToolExecutionError(
                f"Only a single SQL statement is allowed; found {len(statements)} "
                "statements (statement stacking, e.g. via ';', is blocked)."
            )
        stmt_type = statements[0].type.name
        if stmt_type != "SELECT":
            raise ToolExecutionError(
                "Only read-only SELECT queries are allowed; this statement was "
                f"parsed as '{stmt_type}'."
            )
        return sql.strip().rstrip(";")


def _clean_error(exc: Exception) -> str:
    """Strip DuckDB's multi-line caret/context block, keep the useful part."""
    first_line = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
    return f"SQL error: {first_line}"
