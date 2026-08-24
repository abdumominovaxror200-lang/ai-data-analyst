from __future__ import annotations

from typing import Any, Callable

from app.datasets.storage import DatasetRecord
from app.tools import aggregation, anomaly, charts, comparison, correlation, filtering, insights, profiler, report, statistics
from app.tools.errors import ToolExecutionError

FILTERS_SCHEMA = {
    "type": "array",
    "description": "Optional filter conditions applied before analysis.",
    "items": {
        "type": "object",
        "properties": {
            "column": {"type": "string"},
            "op": {"type": "string", "enum": ["==", "!=", ">", ">=", "<", "<=", "in", "contains", "between"]},
            "value": {},
        },
        "required": ["column", "op", "value"],
    },
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "profile_dataset",
            "description": "Get dataset shape, column types, missing values, and duplicate row count.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_data",
            "description": "Get statistical summaries (mean, median, std, min/max, top values) for one or more columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to describe; omit for all numeric columns.",
                    },
                    "filters": FILTERS_SCHEMA,
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_data",
            "description": "Filter rows by conditions and see how many rows match, with a preview.",
            "parameters": {"type": "object", "properties": {"filters": FILTERS_SCHEMA}, "required": ["filters"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "group_and_aggregate",
            "description": "Group rows by a column and aggregate a numeric column (sum, mean, median, count, min, max).",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_by": {"type": "string"},
                    "agg_column": {"type": "string"},
                    "agg_func": {"type": "string", "enum": ["sum", "mean", "median", "count", "min", "max"]},
                    "filters": FILTERS_SCHEMA,
                    "top_n": {"type": "integer"},
                },
                "required": ["group_by", "agg_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_periods",
            "description": "Compare an aggregated value between two date ranges, e.g. this year vs last year.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_column": {"type": "string"},
                    "value_column": {"type": "string"},
                    "current_start": {"type": "string", "description": "ISO date, e.g. 2025-01-01"},
                    "current_end": {"type": "string"},
                    "previous_start": {"type": "string"},
                    "previous_end": {"type": "string"},
                    "agg_func": {"type": "string", "enum": ["sum", "mean", "median", "count", "min", "max"]},
                    "filters": FILTERS_SCHEMA,
                },
                "required": [
                    "date_column",
                    "value_column",
                    "current_start",
                    "current_end",
                    "previous_start",
                    "previous_end",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "correlation_analysis",
            "description": "Compute correlations between numeric columns and rank the strongest relationships.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "method": {"type": "string", "enum": ["pearson", "spearman", "kendall"]},
                    "filters": FILTERS_SCHEMA,
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies",
            "description": "Detect statistical outliers in a numeric column using the IQR or z-score method.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "method": {"type": "string", "enum": ["iqr", "zscore"]},
                    "threshold": {"type": "number"},
                    "filters": FILTERS_SCHEMA,
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_chart",
            "description": "Produce chart-ready data (line, bar, histogram, scatter, pie) for the dashboard to render.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {"type": "string", "enum": ["line", "bar", "histogram", "scatter", "pie"]},
                    "x": {"type": "string"},
                    "y": {"type": "string"},
                    "agg_func": {"type": "string", "enum": ["sum", "mean", "median", "count", "min", "max"]},
                    "bins": {"type": "integer"},
                    "filters": FILTERS_SCHEMA,
                    "top_n": {"type": "integer"},
                },
                "required": ["chart_type", "x"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_business_insights",
            "description": "Compute a bundle of statistics, anomalies, correlations, and data-quality findings for narration.",
            "parameters": {"type": "object", "properties": {"filters": FILTERS_SCHEMA}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": "Generate a full structured business report summarizing key findings for the dataset.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "profile_dataset": lambda record, **p: profiler.profile_dataset(record.df),
    "describe_data": lambda record, **p: statistics.describe_data(record.df, **p),
    "filter_data": lambda record, **p: filtering.filter_data(record.df, **p),
    "group_and_aggregate": lambda record, **p: aggregation.group_and_aggregate(record.df, **p),
    "compare_periods": lambda record, **p: comparison.compare_periods(record.df, **p),
    "correlation_analysis": lambda record, **p: correlation.correlation_analysis(record.df, **p),
    "detect_anomalies": lambda record, **p: anomaly.detect_anomalies(record.df, **p),
    "generate_chart": lambda record, **p: charts.generate_chart(record.df, **p),
    "generate_business_insights": lambda record, **p: insights.generate_business_insights(record.df, **p),
    "generate_report": lambda record, **p: report.generate_report(record.df, record.id, record.original_filename),
}


class ToolRouter:
    def available_tools(self) -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    def execute(self, name: str, record: DatasetRecord, params: dict[str, Any]) -> dict[str, Any]:
        handler = _HANDLERS.get(name)
        if handler is None:
            raise ToolExecutionError(f"Unknown tool '{name}'.")
        return handler(record, **params)
