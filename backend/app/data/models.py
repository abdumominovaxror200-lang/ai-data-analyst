"""Data contracts — the shared vocabulary for describing tables/datasets and
query results across the current in-memory storage, the future SQL layer
(DuckDB/SQLite, per .agent/decisions.md), and the future large-data/streaming
layer.

These are read-only *contracts*, not a replacement for the existing storage.
`app/datasets/storage.py` (DatasetRecord/DatasetStore) keeps owning the actual
upload/parse/persist logic; `InMemoryDataFrameSource` in `in_memory.py` wraps
it to prove these contracts fit the system as it exists today.

Role vocabulary (`ColumnRole`) intentionally matches
`app.tools.profiler._role_for`'s five buckets exactly (numeric / categorical /
datetime / boolean / text) so profiles built from these contracts stay
drop-in compatible with the existing `profile_dataset` tool and the
`DatasetProfile` API schema in `app/schemas.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

ColumnRole = Literal["numeric", "categorical", "datetime", "boolean", "text"]


class ColumnSchema(BaseModel):
    """Static structural description of one column.

    `min_value`/`max_value` are populated only for datetime columns (ISO-8601
    strings, so the model stays trivially JSON-serializable) and represent the
    same "date coverage" concept the existing agent context-building surfaces
    to the LLM (see `.agent/architecture.md` section 5) — kept here as a
    formal schema field rather than an ad hoc computation.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    dtype: str
    role: ColumnRole
    nullable: bool = True
    min_value: Optional[str] = None
    max_value: Optional[str] = None


class ColumnProfile(ColumnSchema):
    """A `ColumnSchema` plus the per-column profiling stats
    `app.tools.profiler.profile_dataset` computes for every column
    (missing_count, missing_pct, unique_count). Field names match that dict's
    `column_info` entries exactly."""

    missing_count: int
    missing_pct: float
    unique_count: int


class TableSchema(BaseModel):
    """Structural description of one table/dataset.

    `row_count` is Optional because a source that is too large (or too
    expensive) to count cheaply — a remote SQL table pre-`EXPLAIN`, or a file
    LARGE-DATA-ENGINEER streams without a full pass — may not have an exact
    count available without a potentially expensive scan.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    columns: list[ColumnSchema]
    row_count: Optional[int] = None

    def column(self, name: str) -> ColumnSchema:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(f"Column '{name}' not found in table '{self.name}'.")

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


class DataProfile(BaseModel):
    """A schema-compatible superset of `app.tools.profiler.profile_dataset`'s
    return dict.

    Every key that dict returns (`rows`, `columns`, `column_info`,
    `numeric_columns`, `categorical_columns`, `date_columns`,
    `boolean_columns`, `missing_total`, `duplicate_rows`) has an exact
    equivalent field here — `to_profile_dataset_dict()` converts back to that
    precise legacy shape for callers that expect it (e.g. the `/datasets`
    upload route today). `table_schema` is the new field: a `TableSchema`
    (with per-column datetime min/max coverage) that the plain dict shape
    can't represent.
    """

    model_config = ConfigDict(frozen=True)

    table_schema: TableSchema
    rows: int
    columns: int
    column_info: list[ColumnProfile]
    numeric_columns: list[str]
    categorical_columns: list[str]
    date_columns: list[str]
    boolean_columns: list[str]
    text_columns: list[str] = Field(default_factory=list)
    missing_total: int
    duplicate_rows: int
    date_ranges: dict[str, dict[str, str]] = Field(default_factory=dict)

    def to_profile_dataset_dict(self) -> dict[str, Any]:
        """Adapter back to the exact dict shape
        `app.tools.profiler.profile_dataset()` returns (including the
        `min_date`/`max_date` per-column fields and top-level `text_columns`/
        `date_ranges` added alongside the P0/P1 reliability fixes — kept in
        sync here rather than duplicated ad hoc)."""
        return {
            "rows": self.rows,
            "columns": self.columns,
            "column_info": [
                {
                    "name": c.name,
                    "dtype": c.dtype,
                    "role": c.role,
                    "missing_count": c.missing_count,
                    "missing_pct": c.missing_pct,
                    "unique_count": c.unique_count,
                    "min_date": c.min_value,
                    "max_date": c.max_value,
                }
                for c in self.column_info
            ],
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "date_columns": self.date_columns,
            "boolean_columns": self.boolean_columns,
            "text_columns": self.text_columns,
            "missing_total": self.missing_total,
            "duplicate_rows": self.duplicate_rows,
            "date_ranges": self.date_ranges,
        }


@dataclass
class QueryResult:
    """Generic result envelope for `DataSource.execute_query` (and usable by
    `read`/`read_chunks` callers that want a uniform wrapper).

    Deliberately a plain dataclass, not a pydantic model: it may carry a live
    `pandas.DataFrame` (`dataframe`), which isn't meant to round-trip through
    JSON validation the way the metadata models above are. Exactly one of
    `rows`/`dataframe` is expected to be populated by a given `DataSource`
    implementation; the other stays `None`. Use `to_dataframe()`/
    `to_records()` to read the result without caring which one a given source
    chose to populate.

    `truncated` is `True` when the underlying source capped the result before
    it was fully materialized (e.g. a `LIMIT` applied by the executor, or a
    LARGE-DATA-ENGINEER pushdown/sampling cap) — `row_count` reflects the
    number of rows actually returned, not necessarily the number that exist.
    """

    columns: list[str]
    row_count: int
    truncated: bool = False
    rows: Optional[list[dict[str, Any]]] = None
    dataframe: Optional[pd.DataFrame] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rows is None and self.dataframe is None:
            raise ValueError("QueryResult requires at least one of `rows` or `dataframe`.")

    def to_dataframe(self) -> pd.DataFrame:
        if self.dataframe is not None:
            return self.dataframe
        return pd.DataFrame(self.rows, columns=self.columns)

    def to_records(self) -> list[dict[str, Any]]:
        if self.rows is not None:
            return self.rows
        assert self.dataframe is not None
        return self.dataframe.to_dict(orient="records")
