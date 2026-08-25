from __future__ import annotations

from app.agent.tool_router import TOOL_SCHEMAS
from app.reasoning.categories import (
    DEFAULT_FALLBACK_CATEGORIES,
    TOOL_CATEGORY_MAP,
    ToolCategory,
    filtered_tool_schemas,
    tool_names_for_categories,
    valid_categories,
)


def test_every_registered_tool_is_mapped_to_exactly_one_category():
    real_tool_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert set(TOOL_CATEGORY_MAP.keys()) == real_tool_names


def test_valid_categories_drops_hallucinated_names():
    assert valid_categories(["STATISTICS", "NOT_A_REAL_CATEGORY", "FORECASTING"]) == ["STATISTICS", "FORECASTING"]


def test_empty_or_fully_invalid_categories_fall_back_to_the_safe_default():
    assert tool_names_for_categories([]) == tool_names_for_categories(DEFAULT_FALLBACK_CATEGORIES)
    assert tool_names_for_categories(["GARBAGE"]) == tool_names_for_categories(DEFAULT_FALLBACK_CATEGORIES)


def test_statistics_question_filtering_excludes_forecasting_and_clustering_tools():
    """The exact example from the Phase 3B spec: a significance question should never
    see forecast/clustering/PCA in its filtered catalog."""
    schemas = filtered_tool_schemas(["STATISTICS"])
    names = {s["function"]["name"] for s in schemas}
    assert "t_test" in names
    assert "forecast" not in names
    assert "kmeans_cluster" not in names
    assert "pca_reduce" not in names


def test_forecasting_question_filtering_yields_only_forecasting_tools():
    schemas = filtered_tool_schemas(["FORECASTING"])
    names = {s["function"]["name"] for s in schemas}
    assert names == {"train_test_split_timeseries", "decompose_timeseries", "forecast", "backtest_forecast"}


def test_sql_question_filtering_yields_only_sql_tools():
    schemas = filtered_tool_schemas(["SQL"])
    names = {s["function"]["name"] for s in schemas}
    assert names == {"run_sql_query", "explain_sql_query"}


def test_multiple_categories_union_correctly():
    schemas = filtered_tool_schemas(["STATISTICS", "REGRESSION"])
    names = {s["function"]["name"] for s in schemas}
    assert "t_test" in names
    assert "linear_regression" in names
    assert "forecast" not in names


def test_category_catalog_covers_all_ten_categories():
    from app.reasoning.categories import category_catalog_text

    text = category_catalog_text()
    for cat in ToolCategory:
        assert cat.value in text
