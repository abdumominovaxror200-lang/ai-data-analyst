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
