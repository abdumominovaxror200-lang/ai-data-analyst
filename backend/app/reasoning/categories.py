"""Tool catalog filtering (Phase 3B.3).

32 tools are now registered in `app.agent.tool_router.TOOL_SCHEMAS` (Phase 3A). Handing
all 32 to a single planning/execution completion invites exactly the failure mode the
user flagged: a "why did revenue fall" or "is X significantly different" question
getting a `forecast` or `pca_reduce` call for no good reason, just because it was in
the list.

This module is the enforcement point. It does not touch `tool_router.py` (still the
single source of truth for what a tool *is*) — it only classifies each already-
registered tool name into one analytical capability category, and offers a filtered
subset of `TOOL_SCHEMAS` for a given set of categories. The reasoning orchestrator's
planner LLM call chooses categories (not raw tools); the actual tool-execution phase
then only ever sees the filtered schema list — so even a hallucinated or
overly-eager category choice cannot make an out-of-category tool reachable, and a
planner mistake is bounded to "wrong category chosen," never "arbitrary tool chosen."
"""

from __future__ import annotations

from enum import Enum

from app.agent.tool_router import TOOL_SCHEMAS


class ToolCategory(str, Enum):
    DATA_PROFILING = "DATA_PROFILING"
    SQL = "SQL"
    EDA = "EDA"
    STATISTICS = "STATISTICS"
    REGRESSION = "REGRESSION"
    FORECASTING = "FORECASTING"
    CLUSTERING = "CLUSTERING"
    SEGMENTATION = "SEGMENTATION"
    DATA_QUALITY = "DATA_QUALITY"
    GENERAL_ANALYSIS = "GENERAL_ANALYSIS"


CATEGORY_DESCRIPTIONS: dict[str, str] = {
    ToolCategory.DATA_PROFILING: "Dataset shape, column types, and coverage — what the data looks like.",
    ToolCategory.SQL: "Ad hoc read-only SQL queries (joins, CTEs, window functions) for questions the other tools can't express directly.",
    ToolCategory.EDA: "Broad exploratory analysis: correlations, distribution shape, automated first-pass review.",
    ToolCategory.STATISTICS: "Hypothesis testing (t-test, chi-square, ANOVA), confidence intervals, effect size — 'is this difference/relationship statistically real?'",
    ToolCategory.REGRESSION: "Linear regression and its diagnostics (normality, heteroscedasticity, multicollinearity, multivariate outliers) — 'what predicts this metric, and can I trust the model?'",
    ToolCategory.FORECASTING: "Time-series decomposition, forecasting, and backtesting — 'what will this metric do next?'",
    ToolCategory.CLUSTERING: "K-means clustering and PCA — 'what natural groupings or dimensions exist in the data?'",
    ToolCategory.SEGMENTATION: "Customer-centric segmentation: RFM, cohort retention, churn risk.",
    ToolCategory.DATA_QUALITY: "Cardinality/identifier detection and univariate anomaly detection — 'is this column/value trustworthy or unusual?'",
    ToolCategory.GENERAL_ANALYSIS: "Filtering, grouping/aggregation, period comparison, charts, and bundled business-insight/report generation — general-purpose lookups.",
}

# Every one of the 32 tools registered in tool_router.py must appear here exactly
# once -- enforced by test_categories.py's coverage test, so a future new tool that
# forgets to update this map fails loudly instead of silently landing in no category.
TOOL_CATEGORY_MAP: dict[str, ToolCategory] = {
    "profile_dataset": ToolCategory.DATA_PROFILING,
    "run_sql_query": ToolCategory.SQL,
    "explain_sql_query": ToolCategory.SQL,
    "automated_eda": ToolCategory.EDA,
    "analyze_distributions": ToolCategory.EDA,
    "correlation_analysis": ToolCategory.EDA,
    "t_test": ToolCategory.STATISTICS,
    "chi_square_test": ToolCategory.STATISTICS,
    "anova_test": ToolCategory.STATISTICS,
    "confidence_interval": ToolCategory.STATISTICS,
    "effect_size": ToolCategory.STATISTICS,
    "linear_regression": ToolCategory.REGRESSION,
    "regression_diagnostics": ToolCategory.REGRESSION,
    "outlier_analysis_multivariate": ToolCategory.REGRESSION,
    "train_test_split_timeseries": ToolCategory.FORECASTING,
    "decompose_timeseries": ToolCategory.FORECASTING,
    "forecast": ToolCategory.FORECASTING,
    "backtest_forecast": ToolCategory.FORECASTING,
    "kmeans_cluster": ToolCategory.CLUSTERING,
    "pca_reduce": ToolCategory.CLUSTERING,
    "rfm_analysis": ToolCategory.SEGMENTATION,
    "cohort_analysis": ToolCategory.SEGMENTATION,
    "churn_risk_analysis": ToolCategory.SEGMENTATION,
    "analyze_cardinality": ToolCategory.DATA_QUALITY,
    "detect_anomalies": ToolCategory.DATA_QUALITY,
    "duplicate_analysis": ToolCategory.DATA_QUALITY,
    "data_quality_report": ToolCategory.DATA_QUALITY,
    "describe_data": ToolCategory.GENERAL_ANALYSIS,
    "filter_data": ToolCategory.GENERAL_ANALYSIS,
    "group_and_aggregate": ToolCategory.GENERAL_ANALYSIS,
    "compare_periods": ToolCategory.GENERAL_ANALYSIS,
    "generate_chart": ToolCategory.GENERAL_ANALYSIS,
    "generate_business_insights": ToolCategory.GENERAL_ANALYSIS,
    "generate_report": ToolCategory.GENERAL_ANALYSIS,
    "contribution_analysis": ToolCategory.GENERAL_ANALYSIS,
    "executive_summary": ToolCategory.GENERAL_ANALYSIS,
    "correlation_heatmap_data": ToolCategory.GENERAL_ANALYSIS,
    "boxplot_data": ToolCategory.GENERAL_ANALYSIS,
    "pareto_chart_data": ToolCategory.GENERAL_ANALYSIS,
}

# Never-empty, never-full-32 fallback: used when the planner names zero valid
# categories (parse failure, degenerate output) so the execution phase still has
# *something* safe and generically useful to work with, rather than either nothing
# (agent can't do anything) or everything (defeats the point of this module).
DEFAULT_FALLBACK_CATEGORIES: list[str] = [ToolCategory.DATA_PROFILING, ToolCategory.GENERAL_ANALYSIS]


def valid_categories(names: list[str]) -> list[str]:
    """Filters `names` down to real ToolCategory values, dropping anything a planner
    LLM call hallucinated. Never trust a free-form category name blindly."""
    known = {c.value for c in ToolCategory}
    return [n for n in names if n in known]


def tool_names_for_categories(categories: list[str]) -> set[str]:
    valid = set(valid_categories(categories)) or set(DEFAULT_FALLBACK_CATEGORIES)
    return {name for name, cat in TOOL_CATEGORY_MAP.items() if cat.value in valid}


def filtered_tool_schemas(categories: list[str]) -> list[dict]:
    """The actual enforcement: returns only the TOOL_SCHEMAS entries whose tool name
    falls in one of the resolved categories. This -- not the planner's free-text
    `tools_required` -- is what actually gets passed as the `tools` argument to any
    LLM completion during the execution phase."""
    allowed = tool_names_for_categories(categories)
    return [s for s in TOOL_SCHEMAS if s["function"]["name"] in allowed]


def category_catalog_text() -> str:
    """Compact, human-readable category list for the planner prompt -- deliberately
    NOT the full 32-tool JSON schema list, per Phase 3B.3."""
    lines = []
    for cat in ToolCategory:
        tool_names = sorted(n for n, c in TOOL_CATEGORY_MAP.items() if c == cat)
        lines.append(f"- {cat.value}: {CATEGORY_DESCRIPTIONS[cat]} (tools: {', '.join(tool_names)})")
    return "\n".join(lines)
