from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class QueryResult:
    """JSON-serializable result of a read-only SQL query.

    `rows` is a list of plain dicts (not a DataFrame) so it can be returned
    directly from a FastAPI route or a future tool-router response without
    any extra conversion step — the same convention `dataframe_to_records`
    already establishes for the rest of the tool surface.
    """

    columns: list[str]
    rows: list[dict]
    row_count: int
    truncated: bool


@runtime_checkable
class DataSource(Protocol):
    """Provisional query interface (per Wave 1 SQL-ENGINEER task spec).

    DATA-ARCHITECT is formalizing the real `DataSource`/`Dataset`/`Schema`
    contracts in parallel this wave. `DuckDBDataSource` and `SQLiteDataSource`
    satisfy this minimal Protocol today; reconciling with the real contract
    should only require adapting construction (how a `Dataset`/table gets
    handed in), not the query surface itself.
    """

    def execute_query(self, sql: str) -> QueryResult: ...
