from __future__ import annotations

import pandas as pd
import pytest

from app.tools.errors import ToolExecutionError
from app.tools.profiler import profile_dataset


def test_profile_dataset_basic(sample_df):
    result = profile_dataset(sample_df)
    assert result["rows"] == len(sample_df)
    assert result["columns"] == len(sample_df.columns)
    assert "revenue" in result["numeric_columns"]
    assert "region" in result["categorical_columns"]
    assert "date" in result["date_columns"]
    assert result["missing_total"] == 0
    assert result["duplicate_rows"] == 0


def test_profile_dataset_detects_missing_and_duplicates():
    df = pd.DataFrame({"a": [1, 1, None, 3], "b": ["x", "x", "y", "z"]})
    result = profile_dataset(df)
    assert result["missing_total"] == 1
    assert result["duplicate_rows"] == 1
    a_info = next(c for c in result["column_info"] if c["name"] == "a")
    assert a_info["missing_count"] == 1
    assert a_info["missing_pct"] == 25.0


def test_profile_dataset_rejects_empty_dataframe():
    with pytest.raises(ToolExecutionError):
        profile_dataset(pd.DataFrame())


def test_profile_dataset_buckets_high_cardinality_columns_as_text():
    """A benchmark run found the agent wrongly claiming a customer-id-style column
    didn't exist, because 'text' role columns were left out of every named bucket
    the agent's context message actually reads from."""
    df = pd.DataFrame(
        {
            "revenue": range(50),
            "customer_id": [f"CUST-{i:05d}" for i in range(50)],
        }
    )
    result = profile_dataset(df)
    assert result["text_columns"] == ["customer_id"]
    assert "customer_id" not in result["categorical_columns"]
    assert "customer_id" not in result["numeric_columns"]


def test_profile_dataset_reports_date_range_for_datetime_columns(sample_df):
    """Surfacing actual date coverage lets the agent (and compare_periods) catch
    'last 12 months' type requests that exceed what the data actually contains."""
    result = profile_dataset(sample_df)
    assert result["date_ranges"]["date"] == {"min": "2024-01-01", "max": "2024-04-09"}
    date_info = next(c for c in result["column_info"] if c["name"] == "date")
    assert date_info["min_date"] == "2024-01-01"
    assert date_info["max_date"] == "2024-04-09"

    non_date_info = next(c for c in result["column_info"] if c["name"] == "region")
    assert non_date_info["min_date"] is None
    assert non_date_info["max_date"] is None
