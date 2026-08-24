"""Data contracts package (DATA-ARCHITECT, Wave 1).

Defines the shared vocabulary for describing tables/datasets and query
results (`ColumnSchema`, `ColumnProfile`, `TableSchema`, `DataProfile`,
`QueryResult`), the `DataSource` extension point future backends (SQL,
large-data) implement against, and `InMemoryDataFrameSource`, a concrete
implementation proving the contract fits the existing in-memory storage
(`app.datasets.storage`) unmodified.

See `.agent/architecture.md`, `.agent/decisions.md`, and
`.agent/roadmap.md` for the cross-agent context this package was built
against.
"""

from __future__ import annotations

from app.data.in_memory import InMemoryDataFrameSource
from app.data.models import ColumnProfile, ColumnRole, ColumnSchema, DataProfile, QueryResult, TableSchema
from app.data.profiling import build_data_profile, build_table_schema
from app.data.source import DataSource

__all__ = [
    "ColumnRole",
    "ColumnSchema",
    "ColumnProfile",
    "TableSchema",
    "DataProfile",
    "QueryResult",
    "DataSource",
    "InMemoryDataFrameSource",
    "build_table_schema",
    "build_data_profile",
]
