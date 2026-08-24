from __future__ import annotations

import pandas as pd

from app.tools.errors import ToolExecutionError
from app.tools.filtering import apply_filters
from app.tools.serialization import dataframe_to_records


def detect_anomalies(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    threshold: float | None = None,
    filters: list[dict] | None = None,
    limit: int = 50,
) -> dict:
    working = apply_filters(df, filters) if filters else df
    if column not in working.columns:
        raise ToolExecutionError(f"Unknown column '{column}'.")
    if method not in {"iqr", "zscore"}:
        raise ToolExecutionError("Method must be 'iqr' or 'zscore'.")

    series = working[column]
    if not pd.api.types.is_numeric_dtype(series):
        raise ToolExecutionError(f"Column '{column}' is not numeric.")
    series = series.dropna()
    if len(series) < 4:
        raise ToolExecutionError("Not enough data points to detect anomalies (need at least 4).")

    if method == "iqr":
        threshold = threshold if threshold is not None else 1.5
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
        mask = (series < lower) | (series > upper)
        bounds = {"lower": round(float(lower), 4), "upper": round(float(upper), 4)}
    else:
        threshold = threshold if threshold is not None else 3.0
        mean, std = series.mean(), series.std()
        if std == 0:
            return {
                "method": method,
                "column": column,
                "threshold": threshold,
                "bounds": None,
                "anomaly_count": 0,
                "anomaly_pct": 0.0,
                "anomalies": [],
            }
        z = (series - mean) / std
        mask = z.abs() > threshold
        bounds = {"lower": round(float(mean - threshold * std), 4), "upper": round(float(mean + threshold * std), 4)}

    anomalous = working.loc[series[mask].index]
    records = dataframe_to_records(anomalous.head(limit))

    return {
        "method": method,
        "column": column,
        "threshold": threshold,
        "bounds": bounds,
        "anomaly_count": int(mask.sum()),
        "anomaly_pct": round(float(mask.mean()) * 100, 2),
        "anomalies": records,
    }
