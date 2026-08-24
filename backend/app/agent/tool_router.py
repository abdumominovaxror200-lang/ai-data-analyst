from __future__ import annotations

from typing import Any, Callable

from app.datasets.storage import DatasetRecord
from app.sql.duckdb_source import DuckDBDataSource
from app.sql.sqlite_source import SQLiteDataSource
from app.tools import (
    aggregation,
    anomaly,
    charts,
    clustering,
    comparison,
    correlation,
    eda,
    filtering,
    forecasting,
    hypothesis,
    insights,
    profiler,
    regression,
    regression_diagnostics,
    report,
    segmentation,
    statistics,
)
from app.tools.errors import ToolExecutionError

# Tool-callable SQL query result cap. Deliberately lower than each DataSource's own
# default (10,000 — sized for direct/API use) because a tool result here is JSON-dumped
# straight into the LLM's conversation context; this project has a documented history of
# 413 "payload too large" failures from oversized tool payloads (see
# app/tools/insights.py, app/tools/forecasting.py, app/tools/eda.py for the same
# discipline applied elsewhere). The underlying SQL engines' read-only enforcement,
# statement validation, and timeout/memory limits are unchanged and unbypassed — this
# only caps how many result *rows* get serialized back to the model.
_SQL_TOOL_MAX_ROWS = 500


def _run_sql_query(record: DatasetRecord, sql: str, engine: str = "duckdb") -> dict[str, Any]:
    """Bridges a DatasetRecord's in-memory DataFrame to the existing, unmodified
    read-only SQL layer (app/sql/duckdb_source.py, sqlite_source.py) for one query.
    Does not alter or bypass any security/resource control in either engine — this
    function only constructs the DataSource (loading `record.df` as a table named
    'dataset') and delegates to its own `execute_query`/`close`."""
    if engine not in ("duckdb", "sqlite"):
        raise ToolExecutionError("engine must be 'duckdb' or 'sqlite'.")
    source_cls = DuckDBDataSource if engine == "duckdb" else SQLiteDataSource
    source = source_cls(record.df, default_table="dataset", max_rows=_SQL_TOOL_MAX_ROWS)
    try:
        result = source.execute_query(sql)
    finally:
        source.close()
    return {
        "engine": engine,
        "columns": result.columns,
        "rows": result.rows,
        "row_count": result.row_count,
        "truncated": result.truncated,
    }


def _explain_sql_query(record: DatasetRecord, sql: str, engine: str = "duckdb") -> dict[str, Any]:
    """Same bridge as `_run_sql_query`, but returns the query plan (EXPLAIN) instead
    of executing it — lets the agent inspect cost before running a potentially
    expensive query."""
    if engine not in ("duckdb", "sqlite"):
        raise ToolExecutionError("engine must be 'duckdb' or 'sqlite'.")
    source_cls = DuckDBDataSource if engine == "duckdb" else SQLiteDataSource
    source = source_cls(record.df, default_table="dataset", max_rows=_SQL_TOOL_MAX_ROWS)
    try:
        result = source.explain(sql)
    finally:
        source.close()
    return {"engine": engine, "columns": result.columns, "plan": result.rows}

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
    # --- Statistical hypothesis testing (Wave 1: hypothesis.py) ---
    {
        "type": "function",
        "function": {
            "name": "t_test",
            "description": (
                "One-sample t-test (column mean vs. a fixed value) or two-sample Welch's t-test "
                "(comparing a numeric column between two groups). Use to test whether an observed "
                "difference is statistically significant, not just a chance result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "group_column": {"type": "string", "description": "Omit for a one-sample test against popmean."},
                    "group_a": {"description": "Required with group_column."},
                    "group_b": {"description": "Required with group_column."},
                    "popmean": {"type": "number", "description": "Required for a one-sample test."},
                    "alpha": {"type": "number", "description": "Significance threshold, default 0.05."},
                    "filters": FILTERS_SCHEMA,
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chi_square_test",
            "description": "Chi-square test of independence between two categorical columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column_a": {"type": "string"},
                    "column_b": {"type": "string"},
                    "alpha": {"type": "number"},
                    "filters": FILTERS_SCHEMA,
                },
                "required": ["column_a", "column_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "anova_test",
            "description": "One-way ANOVA: tests whether a numeric column's mean differs across 3+ groups.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value_column": {"type": "string"},
                    "group_column": {"type": "string"},
                    "alpha": {"type": "number"},
                    "filters": FILTERS_SCHEMA,
                },
                "required": ["value_column", "group_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confidence_interval",
            "description": "Mean and confidence-interval bounds for a numeric column, quantifying estimate uncertainty.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "confidence": {"type": "number", "description": "e.g. 0.95 for a 95% CI."},
                    "filters": FILTERS_SCHEMA,
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "effect_size",
            "description": "Cohen's d effect size between two groups for a numeric column — how large a difference is, not just whether it's significant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "group_column": {"type": "string"},
                    "group_a": {},
                    "group_b": {},
                    "filters": FILTERS_SCHEMA,
                },
                "required": ["column", "group_column", "group_a", "group_b"],
            },
        },
    },
    # --- Regression (Wave 1: regression.py; Wave 2: regression_diagnostics.py) ---
    {
        "type": "function",
        "function": {
            "name": "linear_regression",
            "description": "Ordinary least squares regression: fits target_column on feature_columns, reports coefficients, p-values, and R-squared.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_column": {"type": "string"},
                    "feature_columns": {"type": "array", "items": {"type": "string"}},
                    "alpha": {"type": "number"},
                    "filters": FILTERS_SCHEMA,
                },
                "required": ["target_column", "feature_columns"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "regression_diagnostics",
            "description": (
                "Runs linear_regression plus residual-normality (Shapiro-Wilk), heteroscedasticity "
                "(Breusch-Pagan), and multicollinearity (VIF) checks — use before trusting a "
                "regression's coefficients/p-values at face value."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_column": {"type": "string"},
                    "feature_columns": {"type": "array", "items": {"type": "string"}},
                    "filters": FILTERS_SCHEMA,
                },
                "required": ["target_column", "feature_columns"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "outlier_analysis_multivariate",
            "description": (
                "Detects rows that are jointly unusual across multiple numeric columns at once "
                "(Mahalanobis distance or robust covariance), catching multivariate outliers that "
                "look normal on any single column alone — unlike detect_anomalies (one column at a time)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "method": {"type": "string", "enum": ["mahalanobis", "elliptic_envelope"]},
                    "contamination": {"type": "number", "description": "Expected outlier fraction, default 0.05."},
                    "filters": FILTERS_SCHEMA,
                },
                "required": ["columns"],
            },
        },
    },
    # --- Forecasting (Wave 2: forecasting.py) ---
    {
        "type": "function",
        "function": {
            "name": "train_test_split_timeseries",
            "description": "Chronological (not random) train/test split of a time series — inspect split sizes/dates before forecasting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_column": {"type": "string"},
                    "value_column": {"type": "string"},
                    "test_size": {"type": "number"},
                    "filters": FILTERS_SCHEMA,
                },
                "required": ["date_column", "value_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "decompose_timeseries",
            "description": "Decomposes a time series into trend, seasonal, and residual components; reports seasonality strength.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_column": {"type": "string"},
                    "value_column": {"type": "string"},
                    "period": {"type": "integer", "description": "Seasonal period; omit to auto-infer from date spacing."},
                    "model": {"type": "string", "enum": ["additive", "multiplicative"]},
                    "filters": FILTERS_SCHEMA,
                },
                "required": ["date_column", "value_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forecast",
            "description": (
                "Forecasts future values of a time series (ARIMA or ETS, auto-selected by default) with "
                "prediction intervals. Refuses to forecast further into the future than the observed history."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date_column": {"type": "string"},
                    "value_column": {"type": "string"},
                    "periods": {"type": "integer"},
                    "method": {"type": "string", "enum": ["auto", "arima", "ets"]},
                    "confidence_level": {"type": "number"},
                    "filters": FILTERS_SCHEMA,
                },
                "required": ["date_column", "value_column", "periods"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "backtest_forecast",
            "description": "Measures forecast accuracy (MAE/RMSE/MAPE) on held-out historical data — use to check whether a forecast method is trustworthy on this dataset before relying on it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_column": {"type": "string"},
                    "value_column": {"type": "string"},
                    "method": {"type": "string", "enum": ["auto", "arima", "ets"]},
                    "horizon": {"type": "integer"},
                    "filters": FILTERS_SCHEMA,
                },
                "required": ["date_column", "value_column"],
            },
        },
    },
    # --- Clustering / dimensionality reduction (Wave 2: clustering.py) ---
    {
        "type": "function",
        "function": {
            "name": "kmeans_cluster",
            "description": "K-means clustering over 2+ numeric columns (auto-selects k via silhouette score if not given). Returns cluster sizes and centroids.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "n_clusters": {"type": "integer", "description": "Omit to auto-select between 2 and 8."},
                    "filters": FILTERS_SCHEMA,
                },
                "required": ["columns"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pca_reduce",
            "description": "PCA dimensionality reduction over numeric columns; returns explained variance and per-column loadings per component.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "n_components": {"type": "integer"},
                    "filters": FILTERS_SCHEMA,
                },
                "required": ["columns"],
            },
        },
    },
    # --- Customer segmentation (Wave 2: segmentation.py) ---
    {
        "type": "function",
        "function": {
            "name": "rfm_analysis",
            "description": "Recency/Frequency/Monetary customer segmentation — buckets customers into standard segments (Champions, At Risk, Lost, etc.) with per-segment summaries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_column": {"type": "string"},
                    "date_column": {"type": "string"},
                    "value_column": {"type": "string"},
                    "reference_date": {"type": "string", "description": "ISO date; omit to use the latest date in the data."},
                    "filters": FILTERS_SCHEMA,
                },
                "required": ["customer_column", "date_column", "value_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cohort_analysis",
            "description": "Cohort retention (or cohort total-value) table: groups customers by their first-purchase period and tracks them over subsequent periods.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_column": {"type": "string"},
                    "date_column": {"type": "string"},
                    "value_column": {"type": "string", "description": "Omit for retention-rate mode; provide for total-value mode."},
                    "period": {"type": "string", "enum": ["D", "W", "M", "Q", "Y"]},
                    "filters": FILTERS_SCHEMA,
                },
                "required": ["customer_column", "date_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "churn_risk_analysis",
            "description": "Classifies customers as active/at_risk/churned based on recency vs. their typical purchase cadence (threshold auto-inferred if not given).",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_column": {"type": "string"},
                    "date_column": {"type": "string"},
                    "reference_date": {"type": "string"},
                    "churn_threshold_days": {"type": "number"},
                    "filters": FILTERS_SCHEMA,
                },
                "required": ["customer_column", "date_column"],
            },
        },
    },
    # --- Automated EDA / profiling extensions (Wave 2: eda.py) ---
    {
        "type": "function",
        "function": {
            "name": "automated_eda",
            "description": (
                "Full automatic exploratory pass over the whole dataset: schema, missingness, "
                "duplicates, cardinality, distributions, outliers, correlations, and a prioritized "
                "plain-language list of potential problems. The best first tool to call for an "
                "open-ended 'analyze this dataset' request."
            ),
            "parameters": {"type": "object", "properties": {"filters": FILTERS_SCHEMA}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_cardinality",
            "description": "Classifies every column's cardinality pattern (constant, ID-like, boolean-like, high/low-cardinality categorical, etc.).",
            "parameters": {"type": "object", "properties": {"filters": FILTERS_SCHEMA}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_distributions",
            "description": "Skewness/kurtosis for numeric columns and category-balance/entropy for categorical columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {"type": "array", "items": {"type": "string"}, "description": "Omit to analyze every column."},
                    "filters": FILTERS_SCHEMA,
                },
            },
        },
    },
    # --- Read-only SQL (Wave 1: app/sql/, bridged here — see _run_sql_query) ---
    {
        "type": "function",
        "function": {
            "name": "run_sql_query",
            "description": (
                "Runs a read-only SQL SELECT query against the dataset (available as a table named "
                "'dataset'). Use for joins, window functions, CTEs, or aggregations that are awkward "
                "to express with the other tools. Only SELECT is permitted — no writes, DDL, or "
                "multi-statement queries; results are capped and time/memory-limited."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A single SELECT statement. Table name: dataset."},
                    "engine": {"type": "string", "enum": ["duckdb", "sqlite"], "description": "Default duckdb."},
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_sql_query",
            "description": "Returns the query plan for a SELECT query without executing it, to check cost before running something expensive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                    "engine": {"type": "string", "enum": ["duckdb", "sqlite"]},
                },
                "required": ["sql"],
            },
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
    "t_test": lambda record, **p: hypothesis.t_test(record.df, **p),
    "chi_square_test": lambda record, **p: hypothesis.chi_square_test(record.df, **p),
    "anova_test": lambda record, **p: hypothesis.anova_test(record.df, **p),
    "confidence_interval": lambda record, **p: hypothesis.confidence_interval(record.df, **p),
    "effect_size": lambda record, **p: hypothesis.effect_size(record.df, **p),
    "linear_regression": lambda record, **p: regression.linear_regression(record.df, **p),
    "regression_diagnostics": lambda record, **p: regression_diagnostics.regression_diagnostics(record.df, **p),
    "outlier_analysis_multivariate": lambda record, **p: regression_diagnostics.outlier_analysis_multivariate(record.df, **p),
    "train_test_split_timeseries": lambda record, **p: forecasting.train_test_split_timeseries(record.df, **p),
    "decompose_timeseries": lambda record, **p: forecasting.decompose_timeseries(record.df, **p),
    "forecast": lambda record, **p: forecasting.forecast(record.df, **p),
    "backtest_forecast": lambda record, **p: forecasting.backtest_forecast(record.df, **p),
    "kmeans_cluster": lambda record, **p: clustering.kmeans_cluster(record.df, **p),
    "pca_reduce": lambda record, **p: clustering.pca_reduce(record.df, **p),
    "rfm_analysis": lambda record, **p: segmentation.rfm_analysis(record.df, **p),
    "cohort_analysis": lambda record, **p: segmentation.cohort_analysis(record.df, **p),
    "churn_risk_analysis": lambda record, **p: segmentation.churn_risk_analysis(record.df, **p),
    "automated_eda": lambda record, **p: eda.automated_eda(record.df, **p),
    "analyze_cardinality": lambda record, **p: eda.analyze_cardinality(record.df, **p),
    "analyze_distributions": lambda record, **p: eda.analyze_distributions(record.df, **p),
    "run_sql_query": lambda record, **p: _run_sql_query(record, **p),
    "explain_sql_query": lambda record, **p: _explain_sql_query(record, **p),
}


class ToolRouter:
    def available_tools(self) -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    def execute(self, name: str, record: DatasetRecord, params: dict[str, Any]) -> dict[str, Any]:
        handler = _HANDLERS.get(name)
        if handler is None:
            raise ToolExecutionError(f"Unknown tool '{name}'.")
        return handler(record, **params)
