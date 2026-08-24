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
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MEMORY_LIMIT = "512MB"


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

    RESOURCE LIMITS (P0 remediation — see .agent/decisions.md): two
    independent guards against a *valid* SELECT that is simply too expensive
    (large cross join, huge `range()`, deep subqueries) to let the row-count
    cap alone protect against:

    1. **Execution-time timeout.** A `threading.Timer` scheduled for
       `timeout_seconds` calls `conn.interrupt()` — DuckDB's own supported
       mechanism for cancelling a query from another thread — if the query
       is still running when it fires; cancelled on normal completion.
       Verified empirically to raise `duckdb.InterruptException` (a
       `duckdb.Error` subclass) at the expected wall-clock time.
    2. **Memory ceiling.** `memory_limit` is set via the connection `config`
       at connect time (not a runtime `PRAGMA`, which a query cannot
       override since PRAGMA is already blocked by the statement-type
       check). Verified empirically: a query whose intermediate result
       exceeds the limit fails with `RuntimeError: Could not allocate tuple
       object!` — note this is a plain `RuntimeError`, NOT a `duckdb.Error`
       subclass, so the except clause below must catch both.
    """

    engine = "duckdb"

    def __init__(
        self,
        data: pd.DataFrame | dict[str, pd.DataFrame],
        default_table: str = "dataset",
        max_rows: int = DEFAULT_MAX_ROWS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        memory_limit: str = DEFAULT_MEMORY_LIMIT,
    ) -> None:
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds
        tables = {default_table: data} if isinstance(data, pd.DataFrame) else dict(data)
        if not tables:
            raise ToolExecutionError("At least one table is required.")
        for name in tables:
            validate_identifier(name)

        tmp_dir = Path(tempfile.gettempdir()) / "ai_data_analyst_sql"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = tmp_dir / f"ds_{uuid.uuid4().hex}.duckdb"

        conn_config = {"memory_limit": memory_limit}

        writer = duckdb.connect(str(self._db_path), config=conn_config)
        try:
            try:
                for name, df in tables.items():
                    writer.register("_src", df)
                    writer.execute(f'CREATE TABLE "{name}" AS SELECT * FROM _src')
                    writer.unregister("_src")
            except (duckdb.Error, RuntimeError, MemoryError) as exc:
                # The initial load is also subject to memory_limit — a
                # genuinely large dataset with a small configured limit can
                # fail here, not just on a later query. Must produce the same
                # clean error as a query-time failure, not an unhandled crash
                # (see _execute_with_limits / _is_memory_limit_error below).
                if _is_memory_limit_error(exc):
                    raise ToolExecutionError(
                        "Dataset is too large to load under the configured memory limit."
                    ) from exc
                raise ToolExecutionError(f"Could not load data into DuckDB: {exc}") from exc
        except ToolExecutionError:
            writer.close()
            self._db_path.unlink(missing_ok=True)
            raise
        else:
            writer.close()

        self._conn = duckdb.connect(str(self._db_path), read_only=True, config=conn_config)
        self._lock = threading.Lock()
        self._closed = False

    def _execute_with_limits(self, fn):
        """Runs `fn()` (a zero-arg callable performing the actual `.execute()` +
        fetch) under the timeout watchdog, translating any resource-limit failure
        into a clean, specific ToolExecutionError."""
        timer = threading.Timer(self.timeout_seconds, self._conn.interrupt)
        timer.start()
        try:
            return fn()
        except duckdb.InterruptException as exc:
            raise ToolExecutionError(
                f"Query exceeded the {self.timeout_seconds:.0f}s time limit and was cancelled."
            ) from exc
        except duckdb.Error as exc:
            if _is_memory_limit_error(exc):
                raise ToolExecutionError(
                    "Query exceeded the available memory limit for this session."
                ) from exc
            raise ToolExecutionError(_clean_error(exc)) from exc
        except (RuntimeError, MemoryError) as exc:
            # DuckDB's memory_limit rejection has been observed to surface as
            # EITHER a duckdb.OutOfMemoryException (caught above) OR a plain
            # RuntimeError from the Python binding, depending on exactly where
            # in query execution the allocation fails — verified empirically
            # with two different query shapes (see class docstring). Both
            # must be caught or one shape would escape as an unhandled 500.
            raise ToolExecutionError(
                "Query exceeded the available memory limit for this session."
            ) from exc
        finally:
            timer.cancel()

    def execute_query(self, sql: str) -> QueryResult:
        self._check_open()
        cleaned = self._validate(sql)
        wrapped = f"SELECT * FROM ({cleaned}) AS _query_result LIMIT {self.max_rows + 1}"
        with self._lock:
            df = self._execute_with_limits(lambda: self._conn.execute(wrapped).fetchdf())

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

        def _run():
            result = self._conn.execute(f"EXPLAIN {cleaned}")
            return [d[0] for d in result.description], result.fetchall()

        with self._lock:
            columns, rows = self._execute_with_limits(_run)
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


def _is_memory_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "could not allocate" in text
