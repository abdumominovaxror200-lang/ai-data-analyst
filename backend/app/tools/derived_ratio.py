from __future__ import annotations

import pandas as pd

from app.datasets.metric_registry import MetricRegistry
from app.tools.errors import ToolExecutionError
from app.tools.filtering import apply_filters


def _aggregate(frame: pd.DataFrame, column: str, aggregation: str) -> float:
    if column not in frame.columns:
        raise ToolExecutionError(f"Unknown column '{column}'.")
    if aggregation == "sum":
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ToolExecutionError(f"Column '{column}' must be numeric for sum aggregation.")
        return float(frame[column].sum())
    if aggregation == "count":
        return float(frame[column].count())
    if aggregation == "count_distinct":
        return float(frame[column].nunique(dropna=True))
    raise ToolExecutionError("aggregation must be sum, count, or count_distinct.")


def derived_ratio(
    df: pd.DataFrame,
    metric_registry: MetricRegistry,
    metric_name: str | None = None,
    numerator_column: str | None = None,
    denominator_column: str | None = None,
    numerator_aggregation: str = "sum",
    denominator_aggregation: str = "sum",
    filters: list[dict] | None = None,
    as_percentage: bool = True,
) -> dict:
    """Compute a ratio only from an explicit or registered denominator contract."""
    if metric_name:
        try:
            definition = metric_registry.require_resolved(metric_name)
        except ValueError as exc:
            raise ToolExecutionError(str(exc)) from exc
        numerator_column = definition.numerator_column
        denominator_column = definition.denominator_column
        numerator_aggregation = definition.numerator_aggregation or "sum"
        denominator_aggregation = definition.denominator_aggregation or "sum"
    if not numerator_column or not denominator_column:
        raise ToolExecutionError("Provide metric_name with a resolved registry definition, or explicit numerator_column and denominator_column.")

    working = apply_filters(df, filters) if filters else df
    if working.empty:
        raise ToolExecutionError("No rows match the given filters.")
    numerator = _aggregate(working, numerator_column, numerator_aggregation)
    denominator = _aggregate(working, denominator_column, denominator_aggregation)
    if denominator == 0:
        raise ToolExecutionError("The resolved denominator is zero; the ratio is undefined.")
    ratio = numerator / denominator
    return {
        "metric": metric_name or f"{numerator_column}_per_{denominator_column}",
        "numerator": round(numerator, 6), "denominator": round(denominator, 6),
        "ratio": round(ratio, 8), "ratio_pct": round(ratio * 100, 6) if as_percentage else None,
        "as_percentage": as_percentage, "row_count": int(len(working)),
        "metric_definition": {
            "numerator_column": numerator_column, "denominator_column": denominator_column,
            "numerator_aggregation": numerator_aggregation, "denominator_aggregation": denominator_aggregation,
        },
    }
