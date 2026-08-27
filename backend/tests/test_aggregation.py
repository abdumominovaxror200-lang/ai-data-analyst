from __future__ import annotations

import pandas as pd
import pytest

from app.tools.aggregation import group_and_aggregate
from app.tools.errors import ToolExecutionError


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["North", "North", "South", "South"],
            "revenue": [100, 200, 50, 50],
        }
    )


def test_group_and_aggregate_sum(df):
    result = group_and_aggregate(df, group_by="region", agg_column="revenue", agg_func="sum")
    values = {g["group"]: g["value"] for g in result["groups"]}
    assert values["North"] == 300
    assert values["South"] == 100


def test_group_and_aggregate_mean(df):
    result = group_and_aggregate(df, group_by="region", agg_column="revenue", agg_func="mean")
    values = {g["group"]: g["value"] for g in result["groups"]}
    assert values["North"] == 150
    assert values["South"] == 50


def test_group_and_aggregate_count(df):
    result = group_and_aggregate(df, group_by="region", agg_column="revenue", agg_func="count")
    values = {g["group"]: g["value"] for g in result["groups"]}
    assert values["North"] == 2
    assert values["South"] == 2


def test_group_and_aggregate_unknown_group_column(df):
    with pytest.raises(ToolExecutionError):
        group_and_aggregate(df, group_by="nope", agg_column="revenue")


def test_group_and_aggregate_non_numeric_column_raises():
    df = pd.DataFrame({"region": ["North", "South"], "label": ["a", "b"]})
    with pytest.raises(ToolExecutionError):
        group_and_aggregate(df, group_by="region", agg_column="label", agg_func="sum")


def test_group_and_aggregate_respects_top_n(df):
    result = group_and_aggregate(df, group_by="region", agg_column="revenue", agg_func="sum", top_n=1)
    assert len(result["groups"]) == 1
    assert result["groups"][0]["group"] == "North"


def test_group_and_aggregate_sum_does_not_silently_overflow_on_huge_integers():
    """Real bug (v2 reliability mission, Phase 8): pandas/numpy int64 sum silently
    WRAPS AROUND on overflow with no warning -- verified directly before this fix:
    summing four 2**62 values returned a large NEGATIVE number. Object-dtype
    summation forces Python's arbitrary-precision int arithmetic instead."""
    df = pd.DataFrame({"g": ["A", "A", "B", "B"], "x": [2**62, 2**62, 2**62, 2**62]})
    result = group_and_aggregate(df, group_by="g", agg_column="x", agg_func="sum")
    values = {g["group"]: g["value"] for g in result["groups"]}
    assert values["A"] > 0
    assert values["B"] > 0
    assert values["A"] == pytest.approx(2**63, rel=1e-9)
