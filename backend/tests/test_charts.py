from __future__ import annotations

import pandas as pd
import pytest

from app.tools.charts import generate_chart
from app.tools.errors import ToolExecutionError


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["North", "North", "South", "South"],
            "revenue": [100, 200, 50, 150],
        }
    )


def test_generate_chart_bar(df):
    result = generate_chart(df, chart_type="bar", x="region", y="revenue", agg_func="sum")
    assert result["chart_type"] == "bar"
    assert dict(zip(result["labels"], result["series"][0]["values"]))["North"] == 300


def test_generate_chart_line_sorts_by_x():
    df = pd.DataFrame({"day": [3, 1, 2], "value": [30, 10, 20]})
    result = generate_chart(df, chart_type="line", x="day", y="value", agg_func="sum")
    assert result["labels"] == ["1", "2", "3"]


def test_generate_chart_histogram(df):
    result = generate_chart(df, chart_type="histogram", x="revenue", bins=4)
    assert result["chart_type"] == "histogram"
    assert sum(result["series"][0]["values"]) == 4


def test_generate_chart_scatter_requires_y(df):
    with pytest.raises(ToolExecutionError):
        generate_chart(df, chart_type="scatter", x="revenue")


def test_generate_chart_invalid_type_raises(df):
    with pytest.raises(ToolExecutionError):
        generate_chart(df, chart_type="pyramid", x="region", y="revenue")


def test_generate_chart_unknown_column_raises(df):
    with pytest.raises(ToolExecutionError):
        generate_chart(df, chart_type="bar", x="nope", y="revenue")
