from __future__ import annotations

import pandas as pd
import pytest

from app.tools.anomaly import detect_anomalies
from app.tools.errors import ToolExecutionError


def test_detect_anomalies_iqr_finds_injected_outlier():
    values = [100, 102, 98, 101, 99, 100, 103, 97, 100, 5000]
    df = pd.DataFrame({"value": values})
    result = detect_anomalies(df, column="value", method="iqr")
    assert result["anomaly_count"] == 1
    assert result["anomalies"][0]["value"] == 5000


def test_detect_anomalies_zscore_finds_injected_outlier():
    values = [100, 102, 98, 101, 99, 100, 103, 97, 100, 5000]
    df = pd.DataFrame({"value": values})
    result = detect_anomalies(df, column="value", method="zscore", threshold=2.0)
    assert result["anomaly_count"] >= 1


def test_detect_anomalies_clean_data_has_none():
    df = pd.DataFrame({"value": [100, 101, 99, 100, 102, 98, 100]})
    result = detect_anomalies(df, column="value", method="iqr")
    assert result["anomaly_count"] == 0


def test_detect_anomalies_non_numeric_column_raises():
    df = pd.DataFrame({"value": ["a", "b", "c", "d"]})
    with pytest.raises(ToolExecutionError):
        detect_anomalies(df, column="value")


def test_detect_anomalies_too_few_points_raises():
    df = pd.DataFrame({"value": [1, 2]})
    with pytest.raises(ToolExecutionError):
        detect_anomalies(df, column="value")


def test_detect_anomalies_invalid_method_raises():
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
    with pytest.raises(ToolExecutionError):
        detect_anomalies(df, column="value", method="bogus")
