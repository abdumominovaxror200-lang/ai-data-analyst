from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.tools.errors import ToolExecutionError
from app.tools.forecasting import (
    backtest_forecast,
    decompose_timeseries,
    forecast,
    train_test_split_timeseries,
)

DEMO_XLSX = Path(__file__).resolve().parents[2] / "data" / "demo" / "sales_data.xlsx"


def _seasonal_series(n: int = 120, start: str = "2023-01-01", freq: str = "D", noise_std: float = 0.5) -> pd.DataFrame:
    """trend + weekly-seasonal + small noise, with KNOWN properties: linear trend slope
    of 1.0/day starting at 100, seasonal amplitude 15 with period 7."""
    rng = np.random.default_rng(0)
    dates = pd.date_range(start, periods=n, freq=freq)
    idx = np.arange(n)
    trend = 100 + 1.0 * idx
    seasonal = 15 * np.sin(2 * np.pi * idx / 7)
    values = trend + seasonal + rng.normal(0, noise_std, n)
    return pd.DataFrame({"date": dates, "value": values})


def _trend_only_series(n: int = 30, slope: float = 2.0, intercept: float = 50.0, noise_std: float = 0.2) -> pd.DataFrame:
    """Pure linear trend, no seasonality, small noise -> known next-value expectation."""
    rng = np.random.default_rng(1)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    idx = np.arange(n)
    values = intercept + slope * idx + rng.normal(0, noise_std, n)
    return pd.DataFrame({"date": dates, "value": values})


# --- train_test_split_timeseries ---------------------------------------------------


def test_train_test_split_is_chronological_not_random():
    df = _trend_only_series(n=20)
    result = train_test_split_timeseries(df, "date", "value", test_size=0.2)
    assert result["total_rows"] == 20
    assert result["train"]["row_count"] == 16
    assert result["test"]["row_count"] == 4
    assert result["train"]["start_date"] == "2024-01-01"
    assert result["test"]["end_date"] == "2024-01-20"


def test_train_test_split_refuses_too_few_points():
    df = _trend_only_series(n=5)
    with pytest.raises(ToolExecutionError):
        train_test_split_timeseries(df, "date", "value")


def test_train_test_split_invalid_test_size():
    df = _trend_only_series(n=20)
    with pytest.raises(ToolExecutionError):
        train_test_split_timeseries(df, "date", "value", test_size=0)
    with pytest.raises(ToolExecutionError):
        train_test_split_timeseries(df, "date", "value", test_size=1)


def test_train_test_split_refuses_degenerate_split_even_with_enough_total_points():
    # 10 total points (passes the >= _MIN_SPLIT_POINTS floor) but test_size=0.9 would
    # leave only 1 train row -> must still refuse.
    df = _trend_only_series(n=10)
    with pytest.raises(ToolExecutionError):
        train_test_split_timeseries(df, "date", "value", test_size=0.9)


# --- decompose_timeseries -----------------------------------------------------------


def test_decompose_recovers_known_trend_and_seasonal_period():
    df = _seasonal_series(n=140, noise_std=0.3)
    result = decompose_timeseries(df, "date", "value")

    assert result["period"] == 7
    assert result["period_inferred"] is True
    # Known trend: 100 -> 100 + 1.0*139 = 239 over the series.
    assert result["trend"]["min"] == pytest.approx(100, abs=5)
    assert result["trend"]["max"] == pytest.approx(239, abs=5)
    # Known seasonal amplitude 15 -> peak-to-trough of the seasonal pattern ~30.
    pattern = result["seasonal_pattern"]
    assert len(pattern) == 7
    assert (max(pattern) - min(pattern)) == pytest.approx(30, abs=3)
    # Strong, clean seasonality should score high on the strength metric.
    assert result["seasonality_strength"] > 0.8


def test_decompose_explicit_period_overrides_inference():
    df = _seasonal_series(n=140, noise_std=0.3)
    result = decompose_timeseries(df, "date", "value", period=14)
    assert result["period"] == 14
    assert result["period_inferred"] is False


def test_decompose_refuses_insufficient_cycles():
    # period=7 requires >= 14 points; only 12 available.
    df = _seasonal_series(n=12)
    with pytest.raises(ToolExecutionError):
        decompose_timeseries(df, "date", "value", period=7)


def test_decompose_refuses_when_period_cannot_be_inferred():
    # Irregular ~45-day median spacing falls outside every recognized bucket
    # (daily/weekly/monthly/quarterly), so auto-inference must give up cleanly.
    rng = np.random.default_rng(2)
    dates = pd.to_datetime("2020-01-01") + pd.to_timedelta(np.cumsum(rng.integers(40, 50, 20)), unit="D")
    df = pd.DataFrame({"date": dates, "value": np.arange(20, dtype=float)})
    with pytest.raises(ToolExecutionError):
        decompose_timeseries(df, "date", "value")


def test_decompose_multiplicative_requires_strictly_positive_values():
    df = _seasonal_series(n=60, noise_std=0.3)
    df["value"] = df["value"] - df["value"].max()  # force non-positive values
    with pytest.raises(ToolExecutionError):
        decompose_timeseries(df, "date", "value", model="multiplicative")


def test_decompose_multiplicative_succeeds_when_positive():
    df = _seasonal_series(n=60, noise_std=0.3)
    df["value"] = df["value"] + 1000  # guarantee strictly positive
    result = decompose_timeseries(df, "date", "value", model="multiplicative")
    assert result["model"] == "multiplicative"


def test_decompose_invalid_model_raises():
    df = _seasonal_series(n=60)
    with pytest.raises(ToolExecutionError):
        decompose_timeseries(df, "date", "value", model="bogus")


# --- forecast -------------------------------------------------------------------------


def test_forecast_auto_selects_ets_for_clearly_seasonal_data():
    df = _seasonal_series(n=140, noise_std=0.3)
    result = forecast(df, "date", "value", periods=14)
    assert result["method_used"] == "ets"
    assert result["seasonality"]["period_detected"] == 7
    assert result["seasonality"]["seasonality_strength"] > 0.8
    assert len(result["forecast"]) == 14


def test_forecast_auto_selects_arima_for_nonseasonal_trend_data():
    df = _trend_only_series(n=40)
    result = forecast(df, "date", "value", periods=5)
    assert result["method_used"] == "arima"
    assert len(result["forecast"]) == 5


def test_forecast_recovers_approximate_linear_trend_with_arima():
    # y = 50 + 2*i with tiny noise; forecasting 3 steps ahead should continue near the
    # known trend line (values at i=30,31,32 -> 110, 112, 114).
    df = _trend_only_series(n=30, slope=2.0, intercept=50.0, noise_std=0.1)
    result = forecast(df, "date", "value", periods=3, method="arima")
    forecasted = [p["forecast"] for p in result["forecast"]]
    expected = [110.0, 112.0, 114.0]
    for actual, exp in zip(forecasted, expected):
        assert actual == pytest.approx(exp, abs=3.0)


def test_forecast_confidence_intervals_bracket_point_forecast():
    df = _seasonal_series(n=140, noise_std=0.3)
    result = forecast(df, "date", "value", periods=10, confidence_level=0.95)
    for point in result["forecast"]:
        assert point["lower_bound"] <= point["forecast"] <= point["upper_bound"]


def test_forecast_wider_confidence_level_gives_wider_interval():
    df = _seasonal_series(n=140, noise_std=0.3)
    narrow = forecast(df, "date", "value", periods=5, confidence_level=0.80, method="arima")
    wide = forecast(df, "date", "value", periods=5, confidence_level=0.99, method="arima")
    narrow_width = narrow["forecast"][0]["upper_bound"] - narrow["forecast"][0]["lower_bound"]
    wide_width = wide["forecast"][0]["upper_bound"] - wide["forecast"][0]["lower_bound"]
    assert wide_width > narrow_width


def test_forecast_reports_diagnostics_and_model_metadata():
    df = _trend_only_series(n=30)
    result = forecast(df, "date", "value", periods=3, method="arima")
    assert result["diagnostics"]["aic"] is not None
    assert result["diagnostics"]["bic"] is not None
    assert result["model"]["method"] == "arima"
    assert "order" in result["model"]


def test_forecast_refuses_insufficient_data():
    df = _trend_only_series(n=8)
    with pytest.raises(ToolExecutionError):
        forecast(df, "date", "value", periods=2)


def test_forecast_refuses_unparseable_date_column():
    df = pd.DataFrame({"date": ["not", "a", "date", "at", "all"] * 3, "value": range(15)})
    with pytest.raises(ToolExecutionError):
        forecast(df, "date", "value", periods=2)


def test_forecast_refuses_horizon_beyond_available_history():
    # ~2 months (60 daily points) forecasting ~5 years (1825 days) out -> must refuse.
    df = _trend_only_series(n=60)
    with pytest.raises(ToolExecutionError):
        forecast(df, "date", "value", periods=1825)


def test_forecast_refuses_duplicate_timestamps():
    dates = list(pd.date_range("2024-01-01", periods=12)) * 2
    df = pd.DataFrame({"date": dates, "value": range(24)})
    with pytest.raises(ToolExecutionError):
        forecast(df, "date", "value", periods=2)


def test_forecast_refuses_non_numeric_value_column():
    df = _trend_only_series(n=20)
    df["value"] = df["value"].astype(str)
    with pytest.raises(ToolExecutionError):
        forecast(df, "date", "value", periods=2)


def test_forecast_refuses_unknown_columns():
    df = _trend_only_series(n=20)
    with pytest.raises(ToolExecutionError):
        forecast(df, "nope", "value", periods=2)
    with pytest.raises(ToolExecutionError):
        forecast(df, "date", "nope", periods=2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"periods": 0},
        {"periods": -1},
        {"periods": 2.5},
        {"method": "prophet"},
        {"confidence_level": 0.0},
        {"confidence_level": 1.0},
        {"confidence_level": 1.5},
    ],
)
def test_forecast_rejects_invalid_parameters(kwargs):
    df = _trend_only_series(n=20)
    call_kwargs = {"periods": 3, **kwargs}
    with pytest.raises(ToolExecutionError):
        forecast(df, "date", "value", **call_kwargs)


# --- backtest_forecast ----------------------------------------------------------------


def test_backtest_forecast_low_error_on_clean_trend_data():
    df = _trend_only_series(n=40, slope=2.0, intercept=50.0, noise_std=0.1)
    result = backtest_forecast(df, "date", "value")
    assert result["train_size"] + result["test_size"] == 40
    assert result["mae"] < 5.0
    assert result["mape_pct"] < 10.0


def test_backtest_forecast_recovers_seasonal_pattern_reasonably():
    df = _seasonal_series(n=140, noise_std=0.3)
    result = backtest_forecast(df, "date", "value")
    assert result["mape_pct"] is not None
    # Values here run roughly 85-240; a well-fit seasonal model should keep MAPE well
    # under 20% on such a clean synthetic signal.
    assert result["mape_pct"] < 20.0
    assert len(result["comparison"]) == result["test_size"]


def test_backtest_forecast_respects_explicit_horizon():
    df = _trend_only_series(n=40)
    result = backtest_forecast(df, "date", "value", horizon=3)
    assert result["test_size"] == 3
    assert len(result["comparison"]) == 3


def test_backtest_forecast_horizon_exceeding_test_split_raises():
    df = _trend_only_series(n=40)  # test split (20%) = 8 rows
    with pytest.raises(ToolExecutionError):
        backtest_forecast(df, "date", "value", horizon=100)


def test_backtest_forecast_refuses_insufficient_data():
    df = _trend_only_series(n=5)
    with pytest.raises(ToolExecutionError):
        backtest_forecast(df, "date", "value")


def test_backtest_forecast_invalid_method_raises():
    df = _trend_only_series(n=40)
    with pytest.raises(ToolExecutionError):
        backtest_forecast(df, "date", "value", method="prophet")


# --- Integration test against the real demo dataset ------------------------------------


@pytest.mark.skipif(not DEMO_XLSX.exists(), reason="demo dataset not present in this checkout")
def test_forecast_and_backtest_against_real_demo_dataset_monthly_revenue():
    raw = pd.read_excel(DEMO_XLSX)
    assert {"date", "revenue"}.issubset(raw.columns)

    monthly = raw.set_index("date").resample("MS")["revenue"].sum().reset_index()
    assert len(monthly) >= 20  # ~24 months expected (2024-01 through 2025-12)

    decomposition = decompose_timeseries(monthly, "date", "revenue")
    assert decomposition["period"] == 12
    assert decomposition["n_observations"] == len(monthly)

    result = forecast(monthly, "date", "revenue", periods=6)
    assert result["periods"] == 6
    assert len(result["forecast"]) == 6
    for point in result["forecast"]:
        assert point["lower_bound"] <= point["forecast"] <= point["upper_bound"]

    backtest = backtest_forecast(monthly, "date", "revenue")
    assert backtest["mae"] >= 0
    assert backtest["rmse"] >= backtest["mae"]  # RMSE >= MAE always holds
    # Sanity floor: the fitted model should do noticeably better than a MAE equal to
    # the full scale of monthly revenue (i.e. it is actually fitting something, not
    # returning nonsense on the order of the series' own magnitude).
    assert backtest["mae"] < monthly["revenue"].mean()


@pytest.mark.skipif(not DEMO_XLSX.exists(), reason="demo dataset not present in this checkout")
def test_forecast_against_real_demo_dataset_daily_revenue():
    raw = pd.read_excel(DEMO_XLSX)
    daily = raw.set_index("date").resample("D")["revenue"].sum().reset_index()

    split = train_test_split_timeseries(daily, "date", "revenue", test_size=0.2)
    assert split["train"]["row_count"] + split["test"]["row_count"] == len(daily)

    result = forecast(daily, "date", "revenue", periods=30)
    assert len(result["forecast"]) == 30
    assert result["method_used"] in {"arima", "ets"}
