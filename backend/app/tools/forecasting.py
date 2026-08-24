from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.seasonal import seasonal_decompose

from app.tools.errors import ToolExecutionError
from app.tools.filtering import apply_filters

# --- Documented thresholds (see module docstrings below for rationale) ---

# `forecast`/`backtest_forecast` refuse to run below this many (date, value) points.
_MIN_POINTS = 10

# `train_test_split_timeseries`/`backtest_forecast` refuse a chronological split below
# this many points (same floor as _MIN_POINTS, kept as a separate constant since the
# two tools' "not enough data" reasons are conceptually distinct).
_MIN_SPLIT_POINTS = 10

# Hyndman & Athanasopoulos "strength of seasonality" score, F_s = max(0, 1 - Var(R)/Var(S+R)),
# bounded in [0, 1]. In `forecast(method="auto")`, a series whose decomposition scores at or
# above this is treated as having "clear seasonality" and gets ETS (Holt-Winters); below it,
# ARIMA is used. 0.3 is a conservative middle ground: purely noisy/trend-only series typically
# score well under 0.1, while genuinely seasonal business data (weekly retail cycles, monthly
# billing cycles) commonly scores 0.5+.
_SEASONALITY_STRENGTH_THRESHOLD = 0.3

# Small, fast-fitting ARIMA(p, d, q) candidate grid used by `_fit_arima`. Kept intentionally
# small (this is a request-time tool call, not an offline model-selection job) — covers the
# common low-order cases (random walk, random walk with drift, ARMA(1,1), AR(1), AR(2)-ish
# with differencing) and picks whichever candidate converges with the lowest AIC.
_ARIMA_ORDER_GRID = [(1, 1, 1), (0, 1, 1), (1, 1, 0), (2, 1, 1), (1, 0, 0), (0, 1, 0)]


def train_test_split_timeseries(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    test_size: float = 0.2,
    filters: list[dict] | None = None,
) -> dict:
    """Chronological (not random) train/test split of a time series.

    Time-series data cannot be split randomly without leaking future information into
    the training set, so this always takes the first `1 - test_size` fraction of the
    (date-sorted) rows as train and the remaining tail as test.
    """
    if not 0 < test_size < 1:
        raise ToolExecutionError("test_size must be between 0 and 1 (exclusive).")

    sub = _prepare_series(df, date_column, value_column, filters, min_points=_MIN_SPLIT_POINTS)
    train, test = _split_train_test(sub, test_size)

    return {
        "date_column": date_column,
        "value_column": value_column,
        "total_rows": int(len(sub)),
        "test_size": test_size,
        "train": {
            "row_count": int(len(train)),
            "start_date": _fmt_date(train["date"].iloc[0]),
            "end_date": _fmt_date(train["date"].iloc[-1]),
        },
        "test": {
            "row_count": int(len(test)),
            "start_date": _fmt_date(test["date"].iloc[0]),
            "end_date": _fmt_date(test["date"].iloc[-1]),
        },
    }


def decompose_timeseries(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    period: int | None = None,
    model: str = "additive",
    filters: list[dict] | None = None,
) -> dict:
    """Trend/seasonal/residual decomposition via `statsmodels.tsa.seasonal.seasonal_decompose`.

    `period` auto-inference (used when `period` is not given), based on the median gap
    between consecutive observation dates:
      - ~1 day spacing  -> period=7  (weekly seasonality in daily data)
      - ~7 day spacing  -> period=52 (yearly seasonality in weekly data)
      - ~30 day spacing -> period=12 (yearly seasonality in monthly data)
      - ~90 day spacing -> period=4  (yearly seasonality in quarterly data)
      - anything else   -> cannot infer; caller must pass `period` explicitly.

    The payload is deliberately bounded to summary statistics per component (count/mean/
    std/min/max) plus one representative seasonal cycle, not every raw point — this
    project has a documented history of oversized tool payloads causing LLM request
    failures (see `app/tools/insights.py`), and a full-length series here would be worse.
    """
    if model not in {"additive", "multiplicative"}:
        raise ToolExecutionError("model must be 'additive' or 'multiplicative'.")

    sub = _prepare_series(df, date_column, value_column, filters, min_points=_MIN_POINTS)
    n = len(sub)
    values = sub["value"].to_numpy(dtype=float)

    period_inferred = period is None
    if period_inferred:
        period = _infer_period(sub)
        if period is None:
            raise ToolExecutionError(
                "Could not automatically infer a seasonal period from the spacing between "
                "dates in this data. Auto-inference only recognizes roughly daily (-> 7), "
                "weekly (-> 52), monthly (-> 12), or quarterly (-> 4) spacing. Pass an "
                "explicit `period` for this data."
            )
    elif not isinstance(period, int) or isinstance(period, bool) or period < 2:
        raise ToolExecutionError("period must be an integer >= 2.")

    if n < 2 * period:
        raise ToolExecutionError(
            f"Not enough data to decompose with period={period}: need at least {2 * period} "
            f"observations (2 full cycles), found {n}."
        )
    if model == "multiplicative" and (values <= 0).any():
        raise ToolExecutionError(
            "Multiplicative decomposition requires all values to be strictly positive; "
            f"column '{value_column}' contains zero or negative values. Use model='additive' instead."
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = seasonal_decompose(values, model=model, period=period, extrapolate_trend="freq")
    except Exception as exc:
        raise ToolExecutionError(f"Could not decompose this series: {exc}") from exc

    trend = np.asarray(result.trend, dtype=float)
    seasonal = np.asarray(result.seasonal, dtype=float)
    resid = np.asarray(result.resid, dtype=float)

    return {
        "date_column": date_column,
        "value_column": value_column,
        "model": model,
        "period": int(period),
        "period_inferred": period_inferred,
        "n_observations": n,
        "trend": _component_summary(trend),
        "seasonal": _component_summary(seasonal),
        "residual": _component_summary(resid),
        # One representative cycle of the (repeating) seasonal component — bounded to
        # `period` values regardless of how long the series is.
        "seasonal_pattern": [_round(v) for v in seasonal[:period]],
        # Same Hyndman & Athanasopoulos strength-of-seasonality score `forecast()` uses
        # for its auto method choice — surfaced here too so a caller can inspect *why*
        # `forecast(method="auto")` would pick ETS vs ARIMA on this data.
        "seasonality_strength": _round(_seasonality_strength(resid, seasonal)),
    }


def forecast(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    periods: int,
    method: str = "auto",
    confidence_level: float = 0.95,
    filters: list[dict] | None = None,
) -> dict:
    """Forecast `periods` future values of `value_column` using ARIMA or ETS (Holt-Winters).

    `method="auto"` heuristic: decompose the series (using the same date-spacing-based
    period inference as `decompose_timeseries`) and compute its seasonality strength
    (Hyndman & Athanasopoulos F_s). If F_s >= 0.3 (a genuinely seasonal pattern, not just
    noise), use ETS with an additive trend + seasonal component sized to the detected
    period. Otherwise use ARIMA (small-grid order search by AIC, see `_ARIMA_ORDER_GRID`).

    Refuses (raises `ToolExecutionError`, never fabricates a result) when:
      - fewer than 10 (date, value) points are available after cleaning,
      - `date_column` cannot be parsed as dates at all,
      - `periods` exceeds the number of historical observations available. This project
        forecasts no further into the future than the length of the observed history —
        e.g. 2 months of data cannot support a 5-year forecast — which is a simple,
        auditable, conservative rule (not a statistical guarantee, but a hard floor
        against clearly unsupportable extrapolation).

    Prediction intervals at `confidence_level` come directly from the fitted model's own
    forecast object (`ARIMAResults.get_forecast().conf_int()` for ARIMA,
    `ETSResults.get_prediction().summary_frame()`'s `pi_lower`/`pi_upper` for ETS — the
    ETS equivalent of `conf_int()` in this statsmodels version) — never a hand-rolled
    approximation.
    """
    if method not in {"auto", "arima", "ets"}:
        raise ToolExecutionError("method must be one of: auto, arima, ets.")
    if not isinstance(periods, int) or isinstance(periods, bool) or periods < 1:
        raise ToolExecutionError("periods must be a positive integer.")
    if not 0 < confidence_level < 1:
        raise ToolExecutionError("confidence_level must be between 0 and 1.")

    sub = _prepare_series(df, date_column, value_column, filters, min_points=_MIN_POINTS)
    n = len(sub)
    if periods > n:
        raise ToolExecutionError(
            f"Requested forecast horizon of {periods} periods exceeds the {n} historical "
            "observations available. This tool refuses to forecast further into the future "
            "than the length of the observed history (extrapolating beyond your entire "
            f"observed span is not statistically supportable). Provide more historical data "
            f"or request at most {n} periods."
        )

    values = sub["value"].to_numpy(dtype=float)
    period = _infer_period(sub)

    seasonality_strength = 0.0
    if period is not None and n >= 2 * period:
        seasonality_strength = _try_seasonality_strength(values, period)

    if method == "auto":
        chosen_method = "ets" if seasonality_strength >= _SEASONALITY_STRENGTH_THRESHOLD else "arima"
    else:
        chosen_method = method

    alpha = 1 - confidence_level

    if chosen_method == "ets":
        seasonal_period_used = period if (period is not None and n >= 2 * period) else None
        fitted = _fit_ets(values, seasonal_period_used)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = fitted.get_prediction(start=n, end=n + periods - 1)
            frame = pred.summary_frame(alpha=alpha)
        point = frame["mean"].to_numpy(dtype=float)
        lower = frame["pi_lower"].to_numpy(dtype=float)
        upper = frame["pi_upper"].to_numpy(dtype=float)
        aic, bic = _safe_ic(fitted)
        model_desc = {
            "method": "ets",
            "trend": "add",
            "seasonal": "add" if seasonal_period_used else None,
            "seasonal_periods": seasonal_period_used,
        }
    else:
        order, fitted = _fit_arima(values)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gf = fitted.get_forecast(periods)
            point = np.asarray(gf.predicted_mean, dtype=float)
            ci = np.asarray(gf.conf_int(alpha=alpha), dtype=float)
        lower, upper = ci[:, 0], ci[:, 1]
        aic, bic = _safe_ic(fitted)
        model_desc = {"method": "arima", "order": list(order)}

    future_dates = _future_dates(sub["date"], periods)
    forecast_points = [
        {
            "date": _fmt_date(d),
            "forecast": _round(p),
            "lower_bound": _round(lo),
            "upper_bound": _round(hi),
        }
        for d, p, lo, hi in zip(future_dates, point, lower, upper)
    ]

    return {
        "date_column": date_column,
        "value_column": value_column,
        "periods": periods,
        "confidence_level": confidence_level,
        "method_requested": method,
        "method_used": chosen_method,
        "model": model_desc,
        "n_observations": n,
        "seasonality": {
            "period_detected": period,
            "seasonality_strength": _round(seasonality_strength),
            "auto_selection_threshold": _SEASONALITY_STRENGTH_THRESHOLD,
        },
        "diagnostics": {"aic": _round(aic), "bic": _round(bic)},
        "forecast": forecast_points,
    }


def backtest_forecast(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    method: str = "auto",
    horizon: int | None = None,
    filters: list[dict] | None = None,
) -> dict:
    """Measure forecast accuracy on held-out data: chronological split (via
    `train_test_split_timeseries`'s 80/20 logic), fit on train, forecast the test
    horizon, compare against the actual test values with MAE / RMSE / MAPE.

    This is the tool that proves a forecast method is trustworthy on THIS dataset,
    rather than merely "ran without crashing".
    """
    if method not in {"auto", "arima", "ets"}:
        raise ToolExecutionError("method must be one of: auto, arima, ets.")

    sub = _prepare_series(df, date_column, value_column, filters, min_points=_MIN_SPLIT_POINTS)
    train, test = _split_train_test(sub, test_size=0.2)

    if horizon is not None:
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ToolExecutionError("horizon must be a positive integer.")
        if horizon > len(test):
            raise ToolExecutionError(
                f"Requested horizon ({horizon}) exceeds the test split available "
                f"({len(test)} points held out from a {len(sub)}-point series at test_size=0.2)."
            )
        test = test.iloc[:horizon].reset_index(drop=True)

    train_df = train.rename(columns={"date": date_column, "value": value_column})
    result = forecast(
        train_df,
        date_column=date_column,
        value_column=value_column,
        periods=len(test),
        method=method,
        confidence_level=0.95,
    )

    predicted = np.array([p["forecast"] for p in result["forecast"]], dtype=float)
    actual = test["value"].to_numpy(dtype=float)
    errors = predicted - actual

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    nonzero_mask = actual != 0
    mape_note = None
    if nonzero_mask.all():
        mape = float(np.mean(np.abs(errors / actual)) * 100)
    elif nonzero_mask.any():
        mape = float(np.mean(np.abs(errors[nonzero_mask] / actual[nonzero_mask])) * 100)
        mape_note = (
            "One or more test-period actual values are zero; MAPE is undefined at those "
            "points and was computed only over the non-zero actuals."
        )
    else:
        mape = None
        mape_note = "All test-period actual values are zero; MAPE is undefined."

    return {
        "date_column": date_column,
        "value_column": value_column,
        "method_requested": method,
        "method_used": result["method_used"],
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "mae": _round(mae),
        "rmse": _round(rmse),
        "mape_pct": _round(mape, 2) if mape is not None else None,
        "mape_note": mape_note,
        "comparison": [
            {"date": pt["date"], "actual": _round(a), "forecast": _round(p)}
            for pt, a, p in zip(result["forecast"], actual, predicted)
        ],
    }


# --- Internal helpers -------------------------------------------------------------


def _prepare_series(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    filters: list[dict] | None,
    min_points: int,
) -> pd.DataFrame:
    """Validates inputs, applies filters, and returns a clean, chronologically-sorted
    two-column frame (`date`, `value`) with one row per timestamp. Raises
    `ToolExecutionError` with a specific, actionable message for every way this can go
    wrong, so a bad call never reaches statsmodels as an opaque exception."""
    working = apply_filters(df, filters) if filters else df
    if date_column not in working.columns:
        raise ToolExecutionError(f"Unknown column '{date_column}'.")
    if value_column not in working.columns:
        raise ToolExecutionError(f"Unknown column '{value_column}'.")
    if date_column == value_column:
        raise ToolExecutionError("date_column and value_column must be different columns.")
    if not pd.api.types.is_numeric_dtype(working[value_column]):
        raise ToolExecutionError(f"Column '{value_column}' is not numeric.")

    raw_dates = working[date_column]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed_dates = pd.to_datetime(raw_dates, errors="coerce")
    if raw_dates.notna().sum() > 0 and parsed_dates.notna().sum() == 0:
        raise ToolExecutionError(f"Column '{date_column}' could not be parsed as dates.")

    sub = pd.DataFrame({"date": parsed_dates, "value": working[value_column]}).dropna()
    if len(sub) < min_points:
        raise ToolExecutionError(
            f"At least {min_points} data points (valid date + valid numeric value) are "
            f"required for time-series analysis; found {len(sub)}."
        )

    sub = sub.sort_values("date").reset_index(drop=True)
    if sub["date"].duplicated().any():
        raise ToolExecutionError(
            f"Column '{date_column}' has multiple rows sharing the same date/timestamp. "
            "Aggregate to a single value per time period first (e.g. group_and_aggregate, "
            "or a pandas groupby/resample) before running a forecasting tool."
        )
    return sub


def _split_train_test(sub: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(sub)
    n_test = max(1, round(n * test_size))
    n_train = n - n_test
    if n_train < 5 or n_test < 1:
        raise ToolExecutionError(
            f"Not enough data for a meaningful chronological split with test_size={test_size} "
            f"(would produce {n_train} train / {n_test} test rows from {n} total points). "
            "Provide more data or a different test_size."
        )
    train = sub.iloc[:n_train].reset_index(drop=True)
    test = sub.iloc[n_train:].reset_index(drop=True)
    return train, test


def _infer_period(sub: pd.DataFrame) -> int | None:
    """See `decompose_timeseries`'s docstring for the documented spacing -> period map."""
    if len(sub) < 3:
        return None
    diffs_days = sub["date"].diff().dropna().dt.total_seconds() / 86400.0
    if diffs_days.empty:
        return None
    median_days = float(diffs_days.median())
    if median_days <= 0:
        return None
    if median_days <= 1.5:
        return 7
    if 5.5 <= median_days <= 8.5:
        return 52
    if 26 <= median_days <= 32:
        return 12
    if 85 <= median_days <= 95:
        return 4
    return None


def _seasonality_strength(resid: np.ndarray, seasonal: np.ndarray) -> float:
    """Hyndman & Athanasopoulos strength-of-seasonality score:
    F_s = max(0, 1 - Var(residual) / Var(seasonal + residual)), bounded in [0, 1]."""
    combined = seasonal + resid
    mask = ~np.isnan(combined) & ~np.isnan(resid)
    combined = combined[mask]
    resid = resid[mask]
    if combined.size < 2 or np.var(combined) == 0:
        return 0.0
    return float(max(0.0, 1 - np.var(resid) / np.var(combined)))


def _try_seasonality_strength(values: np.ndarray, period: int) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = seasonal_decompose(values, model="additive", period=period, extrapolate_trend="freq")
    except Exception:
        return 0.0
    return _seasonality_strength(np.asarray(result.resid, dtype=float), np.asarray(result.seasonal, dtype=float))


def _fit_arima(values: np.ndarray):
    best: tuple | None = None
    for order in _ARIMA_ORDER_GRID:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = ARIMA(values, order=order).fit()
        except Exception:
            continue
        if not np.isfinite(fitted.aic):
            continue
        if best is None or fitted.aic < best[1].aic:
            best = (order, fitted)
    if best is None:
        raise ToolExecutionError(
            "Could not fit an ARIMA model to this data (all candidate model orders failed "
            "to converge)."
        )
    return best


def _fit_ets(values: np.ndarray, seasonal_period: int | None):
    series = pd.Series(values, index=pd.RangeIndex(len(values)))
    kwargs: dict = {"trend": "add"}
    if seasonal_period:
        kwargs.update(seasonal="add", seasonal_periods=seasonal_period)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ETSModel(series, **kwargs).fit(disp=False)
    except Exception as exc:
        raise ToolExecutionError(f"Could not fit an ETS model to this data: {exc}") from exc
    return fitted


def _future_dates(dates: pd.Series, periods: int) -> list[pd.Timestamp]:
    """Generates `periods` future timestamps following the observed cadence. Prefers a
    pandas-inferred regular frequency (exact calendar-aware stepping, e.g. month-end);
    falls back to the median observed gap for irregular spacing."""
    freq = pd.infer_freq(pd.DatetimeIndex(dates))
    last = dates.iloc[-1]
    if freq is not None:
        return list(pd.date_range(start=last, periods=periods + 1, freq=freq)[1:])
    diffs = dates.diff().dropna()
    step = diffs.median() if not diffs.empty else pd.Timedelta(days=1)
    return [last + step * (i + 1) for i in range(periods)]


def _component_summary(arr: np.ndarray) -> dict:
    clean = arr[~np.isnan(arr)]
    if clean.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(clean.size),
        "mean": _round(float(np.mean(clean))),
        "std": _round(float(np.std(clean, ddof=1))) if clean.size > 1 else 0.0,
        "min": _round(float(np.min(clean))),
        "max": _round(float(np.max(clean))),
    }


def _safe_ic(fitted) -> tuple[float | None, float | None]:
    aic = getattr(fitted, "aic", None)
    bic = getattr(fitted, "bic", None)
    aic = float(aic) if aic is not None and np.isfinite(aic) else None
    bic = float(bic) if bic is not None and np.isfinite(bic) else None
    return aic, bic


def _fmt_date(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")


def _round(value: float | None, ndigits: int = 4) -> float | None:
    if value is None or (isinstance(value, float) and (np.isnan(value) or not np.isfinite(value))):
        return None
    return round(float(value), ndigits)
