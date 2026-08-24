from __future__ import annotations

import pandas as pd
import pytest

from app.tools.errors import ToolExecutionError
from app.tools.filtering import apply_filters, filter_data


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["North", "South", "East", "West"],
            "value": [10, 20, 30, 40],
        }
    )


def test_apply_filters_equals(df):
    result = apply_filters(df, [{"column": "region", "op": "==", "value": "North"}])
    assert len(result) == 1


def test_apply_filters_greater_than(df):
    result = apply_filters(df, [{"column": "value", "op": ">", "value": 20}])
    assert sorted(result["region"]) == ["East", "West"]


def test_apply_filters_in(df):
    result = apply_filters(df, [{"column": "region", "op": "in", "value": ["North", "South"]}])
    assert len(result) == 2


def test_apply_filters_between(df):
    result = apply_filters(df, [{"column": "value", "op": "between", "value": [15, 35]}])
    assert sorted(result["region"]) == ["East", "South"]


def test_apply_filters_chains_multiple_conditions(df):
    result = apply_filters(
        df,
        [
            {"column": "value", "op": ">", "value": 10},
            {"column": "region", "op": "!=", "value": "West"},
        ],
    )
    assert sorted(result["region"]) == ["East", "South"]


def test_apply_filters_unknown_column_raises(df):
    with pytest.raises(ToolExecutionError):
        apply_filters(df, [{"column": "nope", "op": "==", "value": 1}])


def test_apply_filters_unknown_op_raises(df):
    with pytest.raises(ToolExecutionError):
        apply_filters(df, [{"column": "value", "op": "??", "value": 1}])


def test_filter_data_tool_returns_summary(df):
    result = filter_data(df, [{"column": "value", "op": ">=", "value": 20}])
    assert result["matched_rows"] == 3
    assert result["total_rows"] == 4
    assert len(result["preview"]) == 3


def test_filter_data_requires_at_least_one_condition(df):
    with pytest.raises(ToolExecutionError):
        filter_data(df, [])
