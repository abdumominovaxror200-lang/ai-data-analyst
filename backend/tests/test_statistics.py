from __future__ import annotations

import pandas as pd
import pytest

from app.tools.errors import ToolExecutionError
from app.tools.statistics import describe_data


def test_describe_data_matches_pandas():
    df = pd.DataFrame({"value": [10, 20, 30, 40, 50]})
    result = describe_data(df, columns=["value"])
    assert result["columns"]["value"]["mean"] == 30.0
    assert result["columns"]["value"]["sum"] == 150.0
    assert result["columns"]["value"]["min"] == 10.0
    assert result["columns"]["value"]["max"] == 50.0
    assert result["row_count"] == 5


def test_describe_data_categorical_top_values():
    df = pd.DataFrame({"region": ["North", "North", "South", "East"]})
    result = describe_data(df, columns=["region"])
    assert result["columns"]["region"]["top_values"]["North"] == 2


def test_describe_data_unknown_column_raises():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ToolExecutionError):
        describe_data(df, columns=["nope"])


def test_describe_data_applies_filters():
    df = pd.DataFrame({"region": ["North", "South"], "value": [100, 200]})
    result = describe_data(df, columns=["value"], filters=[{"column": "region", "op": "==", "value": "North"}])
    assert result["row_count"] == 1
    assert result["columns"]["value"]["mean"] == 100.0


def test_describe_data_sum_does_not_silently_overflow_on_huge_integers():
    """Same real int64-wraparound bug fixed in aggregation.py::group_and_aggregate
    (v2 reliability mission, Phase 8)."""
    df = pd.DataFrame({"value": [2**62, 2**62, 2**62, 2**62]})
    result = describe_data(df, columns=["value"])
    assert result["columns"]["value"]["sum"] > 0
    assert result["columns"]["value"]["sum"] == pytest.approx(4 * 2**62, rel=1e-9)
