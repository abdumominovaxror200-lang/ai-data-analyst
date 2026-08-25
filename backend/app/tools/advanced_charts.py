from __future__ import annotations

import pandas as pd

from app.tools.errors import ToolExecutionError
from app.tools.filtering import apply_filters

_CORR_METHODS = {"pearson", "spearman", "kendall"}

# Payload-size caps. This project has a documented, real history of 413
# "payload too large" tool-result failures (see .agent/decisions.md); every
# cap below exists to keep a single tool result bounded and JSON-cheap
# regardless of how large the underlying dataset is.
_MAX_HEATMAP_COLUMNS = 50  # 50x50 = 2,500 cells; also the practical limit of a readable heatmap
_MAX_GROUPS_HARD_CAP = 100  # hard ceiling on `max_groups`, independent of what the caller requests
_MAX_OUTLIERS_PER_GROUP = 20  # never return every outlier point in a group, only a bounded sample
_MAX_PARETO_TOP_N = 50  # hard ceiling on `top_n`, independent of what the caller requests

# Matches app.tools.anomaly.detect_anomalies's IQR convention exactly
# (method="iqr", default threshold=1.5) so outlier bounds are consistent
# across every tool in this project that reports IQR fences.
_IQR_FENCE_MULTIPLIER = 1.5


def correlation_heatmap_data(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "pearson",
    filters: list[dict] | None = None,
) -> dict:
    """Full pairwise correlation matrix, shaped for heatmap rendering.

    This is a different shape for a different purpose than
    `app.tools.correlation.correlation_analysis`: that tool returns only the
    top-N strongest pairs (bounded, meant for narration — "these two columns
    are most correlated"). A heatmap needs the *complete* grid to render,
    including weak/near-zero pairs and the diagonal, or the visualization is
    just wrong. This function is not a wrapper around `correlation_analysis`
    for that reason, but it does reuse the same computation convention: the
    same `method` enum (pearson/spearman/kendall) and the same
    `DataFrame.corr(method=..., numeric_only=True)` call, rather than
    reimplementing correlation math a second way.

    Output shape: `columns` (the ordered axis label list) plus `cells`, a
    flat list of `{"x": col_a, "y": col_b, "value": r}` dicts covering every
    (row, column) pair, including the diagonal (self-correlation == 1.0) and
    both triangles. A flat cell list was chosen over a 2D list-of-lists
    matrix because it maps directly onto what JS heatmap/matrix chart
    primitives actually expect as input (e.g. Chart.js's matrix controller
    takes `{x, y, v}` records; ECharts' heatmap series takes coordinate/value
    triples) — the frontend can feed `cells` straight into a chart with no
    reshape step. A 2D array would still need the client to zip rows/columns
    back to axis labels by index, which is an easy off-by-one bug to
    introduce for no benefit here.

    Every correlation value is rounded to 4 decimal places, matching
    `correlation_analysis`'s rounding convention.

    Capped at `_MAX_HEATMAP_COLUMNS` (50) numeric columns. Above that the
    matrix is rejected with `ToolExecutionError` rather than silently
    truncated — dropping columns out of a matrix would leave the remaining
    cells looking like "no relationship" for pairs that were simply never
    computed, which is worse than refusing outright. Narrow `columns` to fit
    under the cap instead.
    """
    working = apply_filters(df, filters) if filters else df
    if method not in _CORR_METHODS:
        raise ToolExecutionError(f"Method must be one of {sorted(_CORR_METHODS)}.")
    if working.empty:
        raise ToolExecutionError("No rows match the given filters.")

    if columns:
        missing = [c for c in columns if c not in working.columns]
        if missing:
            raise ToolExecutionError(f"Unknown column(s): {', '.join(missing)}")
        numeric_df = working[columns].select_dtypes(include="number")
    else:
        numeric_df = working.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        raise ToolExecutionError("At least two numeric columns are required for a correlation heatmap.")
    if numeric_df.shape[1] > _MAX_HEATMAP_COLUMNS:
        raise ToolExecutionError(
            f"{numeric_df.shape[1]} numeric columns exceeds the heatmap cap of "
            f"{_MAX_HEATMAP_COLUMNS}. Pass a smaller 'columns' list to narrow the matrix."
        )

    corr = numeric_df.corr(method=method, numeric_only=True)
    cols = list(corr.columns)
    cells = []
    for a in cols:
        for b in cols:
            value = corr.loc[a, b]
            cells.append({"x": a, "y": b, "value": None if pd.isna(value) else round(float(value), 4)})

    return {
        "method": method,
        "columns": cols,
        "cells": cells,
    }


def boxplot_data(
    df: pd.DataFrame,
    value_column: str,
    group_column: str | None = None,
    filters: list[dict] | None = None,
    max_groups: int = 20,
) -> dict:
    """Five-number summary + IQR outlier points, shaped for box-plot rendering.

    Returns one "box" for `value_column` overall (`group_column=None`), or
    one box per distinct value of `group_column`. Each box carries min, Q1,
    median, Q3, max, the IQR fences, and a bounded sample of outlier points.

    Outlier detection uses the exact same IQR convention already established
    by `app.tools.anomaly.detect_anomalies` (method="iqr"): fences at
    `Q1 - 1.5*IQR` and `Q3 + 1.5*IQR` (`_IQR_FENCE_MULTIPLIER`, matching
    `detect_anomalies`'s default `threshold`). This is inlined rather than
    calling `detect_anomalies` per group, because that function hard-refuses
    below 4 data points and operates on one ungrouped column at a time — both
    reasonable for its own use case, but a boxplot must still render small
    groups gracefully (an empty outlier list, not a group-killing exception).
    The fence formula itself is identical, and `test_advanced_charts.py`
    cross-checks this module's bounds directly against `detect_anomalies`'s
    bounds on the same data to prove the two stay consistent.

    Groups are capped at `max_groups` (default 20, hard ceiling
    `_MAX_GROUPS_HARD_CAP`=100 regardless of what's requested), keeping the
    largest groups by non-null data-point count and dropping the rest —
    `groups_truncated` reports whether that happened. Outlier points within
    each box are capped at `_MAX_OUTLIERS_PER_GROUP` (20); `outlier_count`
    still reports the true, uncapped count so truncation is never silent.
    """
    working = apply_filters(df, filters) if filters else df
    if value_column not in working.columns:
        raise ToolExecutionError(f"Unknown column '{value_column}'.")
    if not pd.api.types.is_numeric_dtype(working[value_column]):
        raise ToolExecutionError(f"Column '{value_column}' must be numeric for a box plot.")
    if group_column is not None and group_column not in working.columns:
        raise ToolExecutionError(f"Unknown column '{group_column}'.")
    if working[value_column].dropna().empty:
        raise ToolExecutionError(f"Column '{value_column}' has no non-null values.")

    effective_max_groups = min(max_groups, _MAX_GROUPS_HARD_CAP)

    if group_column is None:
        group_frames = [("All", working)]
        total_group_count = 1
        groups_truncated = False
    else:
        sizes = working.groupby(group_column, dropna=False)[value_column].count().sort_values(ascending=False)
        total_group_count = int(len(sizes))
        selected_keys = sizes.head(effective_max_groups).index.tolist()
        groups_truncated = total_group_count > len(selected_keys)
        group_frames = []
        for key in selected_keys:
            subset = working[working[group_column].isna()] if pd.isna(key) else working[working[group_column] == key]
            group_frames.append((str(key), subset))

    boxes = []
    for name, gdf in group_frames:
        series = gdf[value_column].dropna()
        if series.empty:
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        median = float(series.quantile(0.5))
        iqr = q3 - q1
        lower_fence = q1 - _IQR_FENCE_MULTIPLIER * iqr
        upper_fence = q3 + _IQR_FENCE_MULTIPLIER * iqr
        outlier_mask = (series < lower_fence) | (series > upper_fence)
        outlier_values = sorted(series[outlier_mask].tolist())
        boxes.append(
            {
                "group": name,
                "count": int(len(series)),
                "min": round(float(series.min()), 4),
                "q1": round(q1, 4),
                "median": round(median, 4),
                "q3": round(q3, 4),
                "max": round(float(series.max()), 4),
                "lower_fence": round(lower_fence, 4),
                "upper_fence": round(upper_fence, 4),
                "outlier_count": int(outlier_mask.sum()),
                "outliers": [round(float(v), 4) for v in outlier_values[:_MAX_OUTLIERS_PER_GROUP]],
            }
        )

    if not boxes:
        raise ToolExecutionError("No group had any non-null data points to summarize.")

    return {
        "value_column": value_column,
        "group_column": group_column,
        "max_groups": effective_max_groups,
        "total_group_count": total_group_count,
        "groups_truncated": groups_truncated,
        "boxes": boxes,
    }


def pareto_chart_data(
    df: pd.DataFrame,
    dimension_column: str,
    value_column: str,
    filters: list[dict] | None = None,
    top_n: int = 15,
) -> dict:
    """Pareto (80/20 cumulative-contribution) chart data.

    Aggregates `value_column` (sum) by `dimension_column`, sorts descending,
    and returns each category's value, its percentage of the grand total,
    and the running cumulative percentage — the cumulative series is the
    defining feature of a Pareto chart and is guaranteed monotonically
    non-decreasing, ending at exactly 100% (before rounding).

    Individually listed categories are capped at `top_n` (default 15, hard
    ceiling `_MAX_PARETO_TOP_N`=50 regardless of what's requested); the
    remainder is folded into one explicit `"Other"` bucket rather than
    silently dropped, the same "bundle the tail" pattern used by waterfall
    charts elsewhere in this project — so `pct_of_total` values still sum to
    100 and the cumulative line stays accurate even when the underlying
    dimension has far more distinct values than `top_n`.

    Validates that `dimension_column` is actually a meaningful categorical
    dimension for this kind of chart, not an identifier: if every row has a
    distinct value (the classic "customer_id as a Pareto dimension" mistake —
    one bar per row is not a Pareto chart, it's a data dump) or if more than
    95% of values are distinct, `ToolExecutionError` is raised explaining why,
    rather than returning a technically-correct but meaningless chart.

    Also requires non-negative aggregated values (a Pareto chart assumes
    non-negative per-category contributions summing to a meaningful whole —
    mixed-sign values would make "percentage of total" and the monotonic
    cumulative-percentage guarantee both break down) and a non-zero total.
    """
    working = apply_filters(df, filters) if filters else df
    if dimension_column not in working.columns:
        raise ToolExecutionError(f"Unknown column '{dimension_column}'.")
    if value_column not in working.columns:
        raise ToolExecutionError(f"Unknown column '{value_column}'.")
    if not pd.api.types.is_numeric_dtype(working[value_column]):
        raise ToolExecutionError(f"Column '{value_column}' must be numeric for a Pareto chart.")
    if working.empty:
        raise ToolExecutionError("No rows match the given filters.")

    non_null = working.dropna(subset=[value_column])
    if non_null.empty:
        raise ToolExecutionError(f"Column '{value_column}' has no non-null values.")

    n_rows = int(len(non_null))
    n_unique = int(non_null[dimension_column].nunique(dropna=True))
    if n_rows > 1 and n_unique == n_rows:
        raise ToolExecutionError(
            f"Column '{dimension_column}' has a distinct value for every row "
            f"({n_unique} unique of {n_rows}) — it looks like an identifier column, "
            "not a categorical dimension. Pareto analysis needs repeated categories "
            "to aggregate over; pick a lower-cardinality column instead."
        )
    if n_rows > 1 and (n_unique / n_rows) > 0.95:
        raise ToolExecutionError(
            f"Column '{dimension_column}' has {n_unique} distinct values across "
            f"{n_rows} rows ({n_unique / n_rows:.0%} unique) — too high-cardinality to "
            "be a meaningful Pareto dimension. Pick a column with genuinely repeated "
            "categories (e.g. product, category, region) instead."
        )

    grouped = non_null.groupby(dimension_column, dropna=False)[value_column].sum()
    if (grouped < 0).any():
        raise ToolExecutionError(
            f"Column '{value_column}' has negative aggregated values for at least one "
            f"category of '{dimension_column}'; Pareto analysis assumes non-negative "
            "per-category contributions."
        )
    total = float(grouped.sum())
    if total == 0:
        raise ToolExecutionError(f"Total of '{value_column}' is zero; Pareto percentages are undefined.")

    effective_top_n = min(top_n, _MAX_PARETO_TOP_N)
    grouped = grouped.sort_values(ascending=False)
    top = grouped.head(effective_top_n)
    remainder = grouped.iloc[effective_top_n:]

    categories = []
    cumulative_value = 0.0
    for label, value in top.items():
        cumulative_value += float(value)
        categories.append(
            {
                "category": str(label),
                "value": round(float(value), 4),
                "pct_of_total": round(float(value) / total * 100, 2),
                "cumulative_pct": round(cumulative_value / total * 100, 2),
            }
        )

    other_bucket_included = len(remainder) > 0
    if other_bucket_included:
        other_value = float(remainder.sum())
        cumulative_value += other_value
        categories.append(
            {
                "category": "Other",
                "value": round(other_value, 4),
                "pct_of_total": round(other_value / total * 100, 2),
                "cumulative_pct": round(cumulative_value / total * 100, 2),
            }
        )

    return {
        "dimension_column": dimension_column,
        "value_column": value_column,
        "total": round(total, 4),
        "top_n": effective_top_n,
        "other_bucket_included": other_bucket_included,
        "other_category_count": int(len(remainder)),
        "categories": categories,
    }
