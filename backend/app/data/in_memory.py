"""`InMemoryDataFrameSource` — wraps the EXISTING
`app.datasets.storage.DatasetRecord` (a pandas DataFrame already parsed from
an uploaded CSV/XLSX by `DatasetStore.save`) so the current single-dataset,
process-lifetime in-memory storage satisfies the `DataSource` contract
without any changes to `app/datasets/storage.py`.

This is the proof-of-fit for Wave 1: `DataSource`/`TableSchema`/`DataProfile`
are not a paper design, they're an abstraction layer that the *existing,
working* storage already implements today by way of this thin wrapper. Every
one of the 86 pre-existing backend tests continues to exercise
`DatasetStore`/`DatasetRecord` directly and unmodified — this file only adds
a new, additive way to look at the same data.

`execute_query` design choice (documented per the task spec, since either
option was acceptable): it raises `NotImplementedError`. There is no SQL
engine for raw in-memory DataFrames in this codebase yet — `pandas.query()`
is a real option but only covers a WHERE-style row filter (no JOIN/GROUP
BY/window functions), which would be a misleadingly small subset of "SQL"
to advertise through a method literally named `execute_query`. SQL-ENGINEER's
DuckDB/SQLite-backed `DataSource` (per `.agent/decisions.md`) is the intended
real implementation of this method; DuckDB can in fact query a pandas
DataFrame directly (`duckdb.sql("select * from df")`), so that team may well
choose to layer their executor on top of this exact class rather than
building a separate source from scratch.
"""

from __future__ import annotations

from typing import Iterator, Optional

import pandas as pd

from app.data.models import DataProfile, QueryResult, TableSchema
from app.data.profiling import build_data_profile, build_table_schema
from app.data.source import DataSource
from app.datasets.storage import DatasetRecord


class InMemoryDataFrameSource(DataSource):
    """`DataSource` view over one already-loaded `DatasetRecord`.

    Construct directly from a record obtained via the existing
    `DatasetStore`, e.g.::

        record = get_dataset_store().get(dataset_id)
        source = InMemoryDataFrameSource(record)
        schema = source.get_schema()
    """

    def __init__(self, record: DatasetRecord) -> None:
        self._record = record

    @property
    def name(self) -> str:
        return self._record.original_filename

    @property
    def dataset_id(self) -> str:
        return self._record.id

    def get_schema(self) -> TableSchema:
        return build_table_schema(self._record.df, name=self.name)

    def read(self, limit: Optional[int] = None) -> pd.DataFrame:
        df = self._record.df
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            return df.head(limit).copy()
        return df.copy()

    def read_chunks(self, chunk_size: int) -> Iterator[pd.DataFrame]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        df = self._record.df
        for start in range(0, len(df), chunk_size):
            yield df.iloc[start : start + chunk_size].copy()

    def execute_query(self, sql: str) -> QueryResult:
        raise NotImplementedError(
            "InMemoryDataFrameSource has no SQL engine for raw DataFrames — "
            "see this module's docstring. Use a SQL-ENGINEER-provided "
            "DataSource (DuckDB/SQLite) for execute_query support."
        )

    def profile(self) -> DataProfile:
        # Overrides DataSource.profile()'s default only to avoid a redundant
        # get_schema() + read() pair when we already know the df is right
        # here; behaviorally identical to the base implementation.
        return build_data_profile(self._record.df, name=self.name)
