"""Builds `TableSchema`/`DataProfile` contract objects (see `models.py`) from
a `pandas.DataFrame`.

Role-inference logic intentionally mirrors `app.tools.profiler._role_for`'s
algorithm exactly (same five buckets, same categorical-cardinality heuristic:
`nunique <= max(20, 5% of rows)`) so a `DataProfile` built here classifies
columns identically to the existing `profile_dataset` tool.

It is re-implemented here rather than imported because `_role_for` is a
private (underscore-prefixed) helper owned by EDA-ANALYST per
`.agent/agent_registry.md` — depending on another team's private
implementation detail would couple this package to it in a way that could
silently break. Instead, `DataProfile.to_profile_dataset_dict()` gives an
exact-shape adapter back to the legacy dict, and
`backend/tests/test_data_contracts.py` asserts byte-for-byte equality against
a live `profile_dataset()` call on the same DataFrame — that's the contract
that actually matters and it's regression-tested, not just documented.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.data.models import ColumnProfile, ColumnRole, ColumnSchema, DataProfile, TableSchema


def _infer_role(series: pd.Series) -> ColumnRole:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if series.nunique(dropna=True) <= max(20, int(len(series) * 0.05)):
        return "categorical"
    return "text"


def _date_bounds(series: pd.Series) -> tuple[str | None, str | None]:
    non_null = series.dropna()
    if non_null.empty:
        return None, None
    # Date-only (not full ISO datetime) to match app.tools.profiler.profile_dataset
    # exactly — verified byte-for-byte by test_data_contracts.py.
    return non_null.min().strftime("%Y-%m-%d"), non_null.max().strftime("%Y-%m-%d")


def _column_schema(series: pd.Series, label: Any) -> ColumnSchema:
    role = _infer_role(series)
    min_value = max_value = None
    if role == "datetime":
        min_value, max_value = _date_bounds(series)
    return ColumnSchema(
        name=str(label),
        dtype=str(series.dtype),
        role=role,
        nullable=bool(series.isna().any()),
        min_value=min_value,
        max_value=max_value,
    )


def build_table_schema(df: pd.DataFrame, name: str, row_count: int | None = None) -> TableSchema:
    """Structural schema only — no missing/duplicate stats. Cheap: one pass
    over each column for dtype/role/nullability (+ min/max for datetime
    columns), no full-dataframe duplicate scan.

    `row_count`: pass an explicit value (or `None`) when the caller knows a
    cheap count isn't available (e.g. a source LARGE-DATA-ENGINEER streams
    without a full pass); defaults to `len(df)` since callers building a
    schema from an already-materialized DataFrame always have an exact count
    for free.
    """
    columns = [_column_schema(df[label], label) for label in df.columns]
    return TableSchema(
        name=name,
        columns=columns,
        row_count=row_count if row_count is not None else int(len(df)),
    )


def build_data_profile(df: pd.DataFrame, name: str) -> DataProfile:
    """Full profile: schema + missing/duplicate stats, equivalent to (and a
    superset of) `app.tools.profiler.profile_dataset(df)`."""
    from app.tools.errors import ToolExecutionError

    if df.empty:
        raise ToolExecutionError("Dataset is empty.")

    role_buckets: dict[str, list[str]] = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "boolean": [],
        "text": [],
    }
    schema_columns: list[ColumnSchema] = []
    column_info: list[ColumnProfile] = []
    date_ranges: dict[str, dict[str, str]] = {}

    for label in df.columns:
        series = df[label]
        base = _column_schema(series, label)
        schema_columns.append(base)
        if base.role == "datetime" and base.min_value and base.max_value:
            date_ranges[base.name] = {"min": base.min_value, "max": base.max_value}

        missing = int(series.isna().sum())
        column_info.append(
            ColumnProfile(
                **base.model_dump(),
                missing_count=missing,
                missing_pct=round(missing / len(df) * 100, 2) if len(df) else 0.0,
                unique_count=int(series.nunique(dropna=True)),
            )
        )
        if base.role in role_buckets:
            role_buckets[base.role].append(base.name)

    table_schema = TableSchema(name=name, columns=schema_columns, row_count=int(len(df)))

    return DataProfile(
        table_schema=table_schema,
        rows=int(len(df)),
        columns=int(df.shape[1]),
        column_info=column_info,
        numeric_columns=role_buckets["numeric"],
        categorical_columns=role_buckets["categorical"],
        date_columns=role_buckets["datetime"],
        boolean_columns=role_buckets["boolean"],
        text_columns=role_buckets["text"],
        missing_total=int(df.isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
        date_ranges=date_ranges,
    )
