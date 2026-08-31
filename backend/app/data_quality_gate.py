from __future__ import annotations

import pandas as pd

from app.reasoning.contracts import ColumnCoverage, DataCaveats, Limitation
from app.tools.data_quality import data_quality_report
from app.tools.profiler import profile_dataset


def evaluate_data_quality(df: pd.DataFrame) -> tuple[DataCaveats, list[Limitation]]:
    """Run the mandatory, provider-free quality gate without changing the data."""
    profile = profile_dataset(df)
    report = data_quality_report(df)
    total = int(profile["rows"])
    coverage = [
        ColumnCoverage(
            column=item["name"],
            non_null_rows=total - int(item["missing_count"]),
            total_rows=total,
            coverage_pct=round(100.0 - float(item["missing_pct"]), 2),
        )
        for item in profile["column_info"]
    ]
    mixed = report.get("mixed_type_columns", [])
    caveats = DataCaveats(
        column_coverage=coverage,
        duplicate_row_count=int(report["duplicates"]["duplicate_row_count"]),
        duplicate_pct=float(report["duplicates"]["duplicate_pct"]),
        actual_date_ranges=profile.get("date_ranges", {}),
        type_anomalies=[str(item.get("message", item.get("column", "Mixed-type anomaly"))) for item in mixed],
    )

    limitations: list[Limitation] = []
    below_50 = [item for item in coverage if item.coverage_pct < 50]
    below_80 = [item for item in coverage if 50 <= item.coverage_pct < 80]
    if below_50:
        details = ", ".join(f"{item.column} ({item.coverage_pct}%)" for item in below_50)
        limitations.append(Limitation(category="insufficient_coverage", severity="blocks_conclusion", text=f"Data coverage is below 50%: {details}."))
    elif below_80:
        details = ", ".join(f"{item.column} ({item.coverage_pct}%)" for item in below_80)
        limitations.append(Limitation(category="insufficient_coverage", severity="reduces_confidence", text=f"Data coverage is below 80%: {details}."))
    if caveats.duplicate_pct > 5:
        limitations.append(Limitation(category="insufficient_coverage", severity="reduces_confidence", text=f"Exact duplicate rows are {caveats.duplicate_pct}% of the dataset ({caveats.duplicate_row_count} rows)."))
    return caveats, limitations
