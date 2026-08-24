from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from app.tools.insights import generate_business_insights
from app.tools.profiler import profile_dataset


def generate_report(df: pd.DataFrame, dataset_id: str, filename: str) -> dict:
    profile = profile_dataset(df)
    insights = generate_business_insights(df)

    findings = []
    for issue in insights["data_quality_issues"]:
        findings.append(f"Data quality: {issue}")
    for anomaly in insights["anomalies"][:5]:
        findings.append(
            f"{anomaly['anomaly_count']} anomalies detected in '{anomaly['column']}' "
            f"({anomaly['anomaly_pct']}% of values, {anomaly['method'].upper()} method)."
        )
    for pair in insights["top_correlations"][:5]:
        findings.append(f"'{pair['column_a']}' and '{pair['column_b']}' are correlated (r={pair['correlation']}).")

    return {
        "dataset_id": dataset_id,
        "filename": filename,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overview": {
            "rows": profile["rows"],
            "columns": profile["columns"],
            "missing_total": profile["missing_total"],
            "duplicate_rows": profile["duplicate_rows"],
        },
        "statistics": insights["statistics"],
        "anomalies": insights["anomalies"],
        "correlations": insights["top_correlations"],
        "key_findings": findings or ["No significant data quality issues or anomalies detected."],
    }
