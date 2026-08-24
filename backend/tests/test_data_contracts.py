from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest
from pydantic import ValidationError

from app.data import (
    ColumnProfile,
    ColumnSchema,
    DataProfile,
    DataSource,
    InMemoryDataFrameSource,
    QueryResult,
    TableSchema,
    build_data_profile,
    build_table_schema,
)
from app.datasets.storage import DatasetRecord, DatasetStore
from app.tools.errors import ToolExecutionError
from app.tools.profiler import profile_dataset


# ---------------------------------------------------------------------------
# ColumnSchema / TableSchema
# ---------------------------------------------------------------------------


def test_column_schema_rejects_unknown_role():
    with pytest.raises(ValidationError):
        ColumnSchema(name="x", dtype="object", role="not-a-real-role")


def test_column_schema_min_max_default_none():
    col = ColumnSchema(name="x", dtype="int64", role="numeric")
    assert col.min_value is None
    assert col.max_value is None


def test_table_schema_column_lookup_and_names():
    schema = TableSchema(
        name="t",
        columns=[
            ColumnSchema(name="a", dtype="int64", role="numeric"),
            ColumnSchema(name="b", dtype="object", role="categorical"),
        ],
        row_count=10,
    )
    assert schema.column_names == ["a", "b"]
    assert schema.column("b").role == "categorical"
    with pytest.raises(KeyError):
        schema.column("missing")


def test_table_schema_row_count_is_optional_for_uncountable_sources():
    # A source too large/expensive to count cheaply (SQL table, streamed file)
    # must be representable without an exact row count.
    schema = TableSchema(name="huge_source", columns=[], row_count=None)
    assert schema.row_count is None


# ---------------------------------------------------------------------------
# build_table_schema / build_data_profile — the core acceptance bar: must fit
# the existing profile_dataset() shape exactly.
# ---------------------------------------------------------------------------


def test_build_data_profile_matches_profile_dataset_exactly(sample_df: pd.DataFrame):
    legacy = profile_dataset(sample_df)
    profile = build_data_profile(sample_df, name="sample.csv")
    assert profile.to_profile_dataset_dict() == legacy


def test_build_data_profile_matches_profile_dataset_with_missing_and_duplicates():
    df = pd.DataFrame(
        {
            "id": [1, 1, 2, None],
            "label": ["x", "x", "y", "z"],
        }
    )
    legacy = profile_dataset(df)
    profile = build_data_profile(df, name="dupes.csv")
    assert profile.to_profile_dataset_dict() == legacy
    assert profile.duplicate_rows == legacy["duplicate_rows"] == 1


def test_build_data_profile_empty_dataframe_raises_tool_execution_error():
    with pytest.raises(ToolExecutionError):
        build_data_profile(pd.DataFrame(), name="empty.csv")


def test_build_table_schema_datetime_min_max_matches_actual_bounds(sample_df: pd.DataFrame):
    schema = build_table_schema(sample_df, name="sample.csv")
    date_col = schema.column("date")
    assert date_col.role == "datetime"
    # Date-only (not full ISO datetime) — matches app.tools.profiler.profile_dataset's
    # format exactly, reconciled during Wave 1 integration (see profiling.py).
    assert date_col.min_value == sample_df["date"].min().strftime("%Y-%m-%d")
    assert date_col.max_value == sample_df["date"].max().strftime("%Y-%m-%d")


def test_build_table_schema_non_datetime_columns_have_no_bounds(sample_df: pd.DataFrame):
    schema = build_table_schema(sample_df, name="sample.csv")
    numeric_col = schema.column("revenue")
    assert numeric_col.role == "numeric"
    assert numeric_col.min_value is None
    assert numeric_col.max_value is None


def test_build_table_schema_row_count_defaults_to_len(sample_df: pd.DataFrame):
    schema = build_table_schema(sample_df, name="sample.csv")
    assert schema.row_count == len(sample_df)


def test_build_table_schema_row_count_explicit_override_wins_over_len():
    # An explicit row_count (e.g. a fast COUNT(*) from a source that knows a
    # more authoritative number than len(df), such as a pre-filtered sample)
    # takes precedence over len(df).
    df = pd.DataFrame({"a": [1, 2, 3]})
    schema = build_table_schema(df, name="t", row_count=999)
    assert schema.row_count == 999


def test_column_profile_carries_schema_and_profiling_fields():
    df = pd.DataFrame({"a": [1, 2, 2, None]})
    profile = build_data_profile(df, name="t")
    col = profile.column_info[0]
    assert isinstance(col, ColumnProfile)
    assert col.name == "a"
    assert col.role == "numeric"
    assert col.missing_count == 1
    assert col.unique_count == 2


# ---------------------------------------------------------------------------
# QueryResult
# ---------------------------------------------------------------------------


def test_query_result_requires_rows_or_dataframe():
    with pytest.raises(ValueError):
        QueryResult(columns=["a"], row_count=0)


def test_query_result_to_dataframe_from_rows():
    result = QueryResult(
        columns=["a", "b"],
        row_count=2,
        rows=[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}],
    )
    df = result.to_dataframe()
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_query_result_to_records_from_dataframe():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    result = QueryResult(columns=["a", "b"], row_count=2, dataframe=df)
    records = result.to_records()
    assert records == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]


def test_query_result_truncated_flag_defaults_false_and_is_settable():
    result = QueryResult(columns=["a"], row_count=1, rows=[{"a": 1}])
    assert result.truncated is False
    capped = QueryResult(columns=["a"], row_count=1000, rows=[{"a": 1}], truncated=True)
    assert capped.truncated is True


# ---------------------------------------------------------------------------
# DataSource ABC
# ---------------------------------------------------------------------------


def test_data_source_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        DataSource()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# InMemoryDataFrameSource — proves the contract fits the EXISTING storage.
# ---------------------------------------------------------------------------


def _make_record(df: pd.DataFrame, filename: str = "sample.csv") -> DatasetRecord:
    return DatasetRecord(
        id="test-id",
        original_filename=filename,
        extension=".csv",
        uploaded_at=datetime.now(timezone.utc),
        df=df,
        stored_path="unused",
    )


def test_in_memory_source_is_a_data_source(sample_df: pd.DataFrame):
    source = InMemoryDataFrameSource(_make_record(sample_df))
    assert isinstance(source, DataSource)


def test_in_memory_source_get_schema_matches_build_table_schema(sample_df: pd.DataFrame):
    source = InMemoryDataFrameSource(_make_record(sample_df))
    assert source.get_schema() == build_table_schema(sample_df, name="sample.csv")


def test_in_memory_source_read_full_and_limited(sample_df: pd.DataFrame):
    source = InMemoryDataFrameSource(_make_record(sample_df))
    full = source.read()
    assert len(full) == len(sample_df)
    limited = source.read(limit=5)
    assert len(limited) == 5
    pd.testing.assert_frame_equal(limited.reset_index(drop=True), sample_df.head(5).reset_index(drop=True))


def test_in_memory_source_read_does_not_mutate_original(sample_df: pd.DataFrame):
    source = InMemoryDataFrameSource(_make_record(sample_df))
    out = source.read()
    out.iloc[0, 0] = None
    # Original record's df must be untouched (read() returns a copy).
    assert source.read().iloc[0, 0] == sample_df.iloc[0, 0]


def test_in_memory_source_read_chunks_covers_all_rows_no_overlap(sample_df: pd.DataFrame):
    source = InMemoryDataFrameSource(_make_record(sample_df))
    chunks = list(source.read_chunks(chunk_size=30))
    assert sum(len(c) for c in chunks) == len(sample_df)
    assert all(len(c) <= 30 for c in chunks)
    reassembled = pd.concat(chunks).reset_index(drop=True)
    pd.testing.assert_frame_equal(reassembled, sample_df.reset_index(drop=True))


def test_in_memory_source_read_chunks_rejects_non_positive_chunk_size(sample_df: pd.DataFrame):
    source = InMemoryDataFrameSource(_make_record(sample_df))
    with pytest.raises(ValueError):
        list(source.read_chunks(chunk_size=0))


def test_in_memory_source_execute_query_not_implemented(sample_df: pd.DataFrame):
    source = InMemoryDataFrameSource(_make_record(sample_df))
    with pytest.raises(NotImplementedError):
        source.execute_query("SELECT 1")


def test_in_memory_source_profile_matches_build_data_profile(sample_df: pd.DataFrame):
    source = InMemoryDataFrameSource(_make_record(sample_df))
    assert source.profile() == build_data_profile(sample_df, name="sample.csv")


def test_in_memory_source_profile_matches_legacy_profile_dataset(sample_df: pd.DataFrame):
    source = InMemoryDataFrameSource(_make_record(sample_df))
    assert source.profile().to_profile_dataset_dict() == profile_dataset(sample_df)


def test_in_memory_source_default_abc_profile_method_matches_override(sample_df: pd.DataFrame):
    # DataSource.profile() (the ABC's default, built from get_schema()+read())
    # and InMemoryDataFrameSource's override must agree, since the override
    # exists only to skip a redundant read(), not to change behavior.
    source = InMemoryDataFrameSource(_make_record(sample_df))
    assert DataSource.profile(source) == source.profile()


# ---------------------------------------------------------------------------
# Integration: wrap a real DatasetStore-produced record (the actual upload
# path), not just a hand-built DatasetRecord.
# ---------------------------------------------------------------------------


def test_in_memory_source_wraps_real_dataset_store_record(sample_df: pd.DataFrame, tmp_path, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))
    get_settings.cache_clear()
    store = DatasetStore()
    record = store.save("sample.csv", sample_df.to_csv(index=False).encode("utf-8"))

    source = InMemoryDataFrameSource(record)
    schema = source.get_schema()
    assert schema.row_count == len(sample_df)
    assert set(schema.column_names) == set(sample_df.columns)
    assert source.profile().rows == len(sample_df)
    get_settings.cache_clear()
