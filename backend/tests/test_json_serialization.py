from __future__ import annotations

import json

import pandas as pd
import pytest

from app.tools.anomaly import detect_anomalies
from app.tools.charts import generate_chart
from app.tools.filtering import filter_data


@pytest.fixture
def df_with_dates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=12, freq="D"),
            "region": ["North", "South"] * 6,
            "value": [10, 12, 11, 9, 500, 10, 11, 13, 9, 12, 8, 11],
        }
    )


def test_filter_data_preview_is_json_serializable_with_datetime_column(df_with_dates):
    result = filter_data(df_with_dates, [{"column": "value", "op": ">", "value": 0}])
    json.dumps(result)  # must not raise TypeError: Object of type Timestamp is not JSON serializable


def test_detect_anomalies_records_are_json_serializable_with_datetime_column(df_with_dates):
    result = detect_anomalies(df_with_dates, column="value", method="iqr")
    assert result["anomaly_count"] >= 1
    json.dumps(result)


def test_scatter_chart_points_are_json_serializable_with_datetime_x(df_with_dates):
    result = generate_chart(df_with_dates, chart_type="scatter", x="date", y="value")
    json.dumps(result)
