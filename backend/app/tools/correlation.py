from __future__ import annotations

import pandas as pd

from app.tools.errors import ToolExecutionError
from app.tools.filtering import apply_filters


def correlation_analysis(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "pearson",
    filters: list[dict] | None = None,
) -> dict:
    working = apply_filters(df, filters) if filters else df
    if method not in {"pearson", "spearman", "kendall"}:
        raise ToolExecutionError("Method must be one of pearson, spearman, kendall.")

    if columns:
        missing = [c for c in columns if c not in working.columns]
        if missing:
            raise ToolExecutionError(f"Unknown column(s): {', '.join(missing)}")
        numeric_df = working[columns].select_dtypes(include="number")
    else:
        numeric_df = working.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        raise ToolExecutionError("At least two numeric columns are required for correlation analysis.")

    corr = numeric_df.corr(method=method, numeric_only=True)
    cols = list(corr.columns)
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            value = corr.loc[a, b]
            if pd.isna(value):
                continue
            pairs.append({"column_a": a, "column_b": b, "correlation": round(float(value), 4)})
    pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)

    matrix = {
        a: {b: (None if pd.isna(v) else round(float(v), 4)) for b, v in row.items()}
        for a, row in corr.to_dict(orient="index").items()
    }

    return {
        "method": method,
        "columns": cols,
        "matrix": matrix,
        "strongest_pairs": pairs[:10],
    }
