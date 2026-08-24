from __future__ import annotations

import pandas as pd

from app.tools.anomaly import detect_anomalies
from app.tools.correlation import correlation_analysis
from app.tools.errors import ToolExecutionError
from app.tools.filtering import apply_filters
from app.tools.profiler import profile_dataset
from app.tools.statistics import describe_data


def generate_business_insights(df: pd.DataFrame, filters: list[dict] | None = None) -> dict:
    working = apply_filters(df, filters) if filters else df
    if working.empty:
        raise ToolExecutionError("No rows match the given filters.")

    profile = profile_dataset(working)
    numeric_cols = profile["numeric_columns"]
    stats = describe_data(working, numeric_cols) if numeric_cols else {"row_count": len(working), "columns": {}}

    anomaly_findings = []
    for col in numeric_cols[:8]:
        try:
            result = detect_anomalies(working, col, method="iqr")
        except ToolExecutionError:
            continue
        if result["anomaly_count"] > 0:
            anomaly_findings.append({"column": col, **result})

    correlation_findings = None
    if len(numeric_cols) >= 2:
        try:
            correlation_findings = correlation_analysis(working, numeric_cols)
        except ToolExecutionError:
            correlation_findings = None

    data_quality_issues = [c["name"] for c in profile["column_info"] if c["missing_pct"] > 5]
    if profile["duplicate_rows"] > 0:
        data_quality_issues.append(f"{profile['duplicate_rows']} duplicate rows")

    return {
        "profile_summary": {
            "rows": profile["rows"],
            "columns": profile["columns"],
            "numeric_columns": numeric_cols,
            "categorical_columns": profile["categorical_columns"],
        },
        "statistics": stats,
        "anomalies": anomaly_findings,
        "top_correlations": (correlation_findings or {}).get("strongest_pairs", []),
        "data_quality_issues": data_quality_issues,
    }
