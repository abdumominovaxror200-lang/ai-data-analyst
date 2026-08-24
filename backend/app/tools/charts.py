from __future__ import annotations

import numpy as np
import pandas as pd

from app.tools.errors import ToolExecutionError
from app.tools.filtering import apply_filters
from app.tools.serialization import dataframe_to_records

_CHART_TYPES = {"line", "bar", "histogram", "scatter", "pie"}
_AGG_FUNCS = {"sum", "mean", "median", "count", "min", "max"}


def generate_chart(
    df: pd.DataFrame,
    chart_type: str,
    x: str,
    y: str | None = None,
    agg_func: str = "sum",
    bins: int = 20,
    filters: list[dict] | None = None,
    top_n: int = 20,
) -> dict:
    working = apply_filters(df, filters) if filters else df
    if chart_type not in _CHART_TYPES:
        raise ToolExecutionError(f"Unsupported chart type '{chart_type}'. Use one of {sorted(_CHART_TYPES)}.")
    if x not in working.columns:
        raise ToolExecutionError(f"Unknown column '{x}'.")
    if working.empty:
        raise ToolExecutionError("No rows match the given filters.")

    if chart_type == "histogram":
        series = working[x].dropna()
        if not pd.api.types.is_numeric_dtype(series):
            raise ToolExecutionError(f"Column '{x}' must be numeric for a histogram.")
        counts, edges = np.histogram(series, bins=bins)
        return {
            "chart_type": chart_type,
            "x_label": x,
            "labels": [f"{edges[i]:.2f}-{edges[i + 1]:.2f}" for i in range(len(edges) - 1)],
            "series": [{"name": x, "values": counts.tolist()}],
        }

    if chart_type == "scatter":
        if not y or y not in working.columns:
            raise ToolExecutionError("Scatter charts require a valid 'y' column.")
        sample = working[[x, y]].dropna().head(2000).rename(columns={x: "x", y: "y"})
        return {
            "chart_type": chart_type,
            "x_label": x,
            "y_label": y,
            "points": dataframe_to_records(sample),
        }

    if not y or y not in working.columns:
        raise ToolExecutionError(f"Chart type '{chart_type}' requires a valid 'y' column.")
    if agg_func not in _AGG_FUNCS:
        raise ToolExecutionError(f"Unsupported aggregation '{agg_func}'. Use one of {sorted(_AGG_FUNCS)}.")
    if agg_func != "count" and not pd.api.types.is_numeric_dtype(working[y]):
        raise ToolExecutionError(f"Column '{y}' is not numeric; cannot compute '{agg_func}'.")

    grouped = working.groupby(x, dropna=False)[y].agg(agg_func)
    grouped = grouped.sort_values(ascending=False).head(top_n)
    if chart_type == "line":
        grouped = grouped.sort_index()

    return {
        "chart_type": chart_type,
        "x_label": x,
        "y_label": y,
        "labels": [str(i) for i in grouped.index],
        "series": [{"name": y, "values": [round(float(v), 4) for v in grouped.values]}],
    }
