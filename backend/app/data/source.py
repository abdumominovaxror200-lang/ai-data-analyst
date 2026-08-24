"""`DataSource` — the extension point every future data backend implements.

This is the contract SQL-ENGINEER (DuckDB/SQLite, per `.agent/decisions.md`)
and LARGE-DATA-ENGINEER (chunking/streaming/pushdown) build against. Both
teams can start from a provisional stub of this ABC and integrate once this
file lands, per `.agent/dependency_graph.md`.

Implemented as an `abc.ABC` (not `typing.Protocol`) deliberately: subclasses
are meant to *inherit* shared behavior (see `profile()` below, which every
concrete source gets for free from `get_schema()` + `read()`), not just
structurally match a shape. A `Protocol` would work for pure duck-typing but
would give up that free default-method sharing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Optional

import pandas as pd

from app.data.models import DataProfile, QueryResult, TableSchema
from app.data.profiling import build_data_profile


class DataSource(ABC):
    """Read-only access to one table/dataset, regardless of backend.

    All four abstract methods are read-only by contract: no `DataSource`
    implementation should expose a write/mutate path. `execute_query` in
    particular MUST reject non-SELECT statements once a real SQL engine sits
    behind it — that enforcement is SQL-ENGINEER's responsibility, but the
    read-only *intent* is fixed here at the contract level so every backend
    is held to it.
    """

    @abstractmethod
    def get_schema(self) -> TableSchema:
        """Return this source's structural schema (columns, dtypes, roles,
        row_count if cheaply known). Should not require reading the full
        dataset for sources that can answer this more cheaply (e.g. from a
        database catalog or file metadata)."""
        raise NotImplementedError

    @abstractmethod
    def read(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Materialize (up to `limit`) rows as a DataFrame. Intended for
        sources small enough to fit in memory whole, or callers that already
        know they only need a bounded slice. Large sources should prefer
        `read_chunks`."""
        raise NotImplementedError

    @abstractmethod
    def read_chunks(self, chunk_size: int) -> Iterator[pd.DataFrame]:
        """Yield the dataset as successive DataFrames of up to `chunk_size`
        rows each, without materializing the whole dataset in memory at once.
        This is the extension point LARGE-DATA-ENGINEER implements against
        for sources too big to load whole (see `.agent/roadmap.md`)."""
        raise NotImplementedError

    @abstractmethod
    def execute_query(self, sql: str) -> QueryResult:
        """Execute a read-only SQL query against this source and return a
        `QueryResult`. This is the extension point SQL-ENGINEER implements
        against (DuckDB/SQLite, per `.agent/decisions.md`) — read-only
        enforcement (no INSERT/UPDATE/DELETE/DDL) is that implementation's
        responsibility, not this ABC's, but every implementation MUST honor
        it. A source with no SQL engine behind it (e.g.
        `InMemoryDataFrameSource`) may raise `NotImplementedError` — see that
        class's docstring for why that's an acceptable answer for Wave 1."""
        raise NotImplementedError

    def profile(self) -> DataProfile:
        """Default profile implementation: reads the full dataset and
        delegates to `app.data.profiling.build_data_profile`. Concrete
        sources with a cheaper path (e.g. pushdown aggregate queries for
        missing/duplicate counts instead of pulling every row into pandas)
        should override this method rather than relying on this default."""
        schema = self.get_schema()
        df = self.read()
        return build_data_profile(df, name=schema.name)
