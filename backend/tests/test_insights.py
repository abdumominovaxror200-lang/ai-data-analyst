from __future__ import annotations

import pandas as pd
import pytest

from app.tools.errors import ToolExecutionError
from app.tools.insights import generate_business_insights
from app.tools.report import generate_report


@pytest.fixture
def df_with_anomalies() -> pd.DataFrame:
    values = [100, 102, 98, 101, 99, 100, 103, 97, 100, 5000]
    return pd.DataFrame({"revenue": values, "region": ["North", "South"] * 5})


def test_generate_business_insights_summarizes_anomalies_without_raw_rows(df_with_anomalies):
    """Regression test: a real Groq benchmark run hit a 413 'payload too large' error
    because this bundle used to embed up to 50 raw anomalous rows per numeric column.
    The bundle must stay summary-only — counts/bounds, not row dumps."""
    result = generate_business_insights(df_with_anomalies)
    revenue_finding = next(a for a in result["anomalies"] if a["column"] == "revenue")

    assert revenue_finding["anomaly_count"] == 1
    assert "bounds" in revenue_finding
    assert "anomalies" not in revenue_finding  # the raw row list must not be embedded here


def test_generate_business_insights_rejects_empty_filter_result(df_with_anomalies):
    with pytest.raises(ToolExecutionError):
        generate_business_insights(df_with_anomalies, filters=[{"column": "region", "op": "==", "value": "nope"}])


def test_generate_report_includes_key_findings_and_overview(df_with_anomalies):
    report = generate_report(df_with_anomalies, dataset_id="ds-1", filename="test.csv")
    assert report["overview"]["rows"] == len(df_with_anomalies)
    assert any("revenue" in finding for finding in report["key_findings"])
