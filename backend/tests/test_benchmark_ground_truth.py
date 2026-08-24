"""Regression tests for the deterministic ground truth behind the 7-question manual
benchmark run documented in .agent/benchmark_status.md and formalized in
backend/tests/benchmark/questions.json.

The benchmark itself was run against a live Groq LLM and can't be replayed
deterministically here (see BENCHMARK-ENGINEER's future automated runner), but the
NUMBERS the LLM's answers were checked against are 100% deterministic - they come
straight out of `profile_dataset` / `compare_periods` run against the seeded demo
dataset (`data/demo/sales_data.xlsx`). This file pins those numbers so a future change
to a tool, to pandas, or to the demo dataset itself fails loudly here instead of
silently invalidating the benchmark's ground truth.

Every number below was independently re-derived from the real file for this task
(not copied blind from the task description) - see the QA-ENGINEER report for the
verification commands.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.datasets.storage import DatasetStore
from app.tools.comparison import compare_periods
from app.tools.profiler import profile_dataset

DEMO_DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "demo" / "sales_data.xlsx"


@pytest.fixture(scope="module")
def demo_df() -> pd.DataFrame:
    """Loads the real demo dataset through the exact same code path production uses
    (DatasetStore._parse -> pd.read_excel + datetime inference), so this test reflects
    what the running app actually sees, not a hand-rolled re-read of the file."""
    if not DEMO_DATASET_PATH.exists():
        pytest.skip(f"Demo dataset not found at {DEMO_DATASET_PATH}")
    content = DEMO_DATASET_PATH.read_bytes()
    return DatasetStore._parse(content, ".xlsx")


@pytest.fixture(scope="module")
def demo_profile(demo_df: pd.DataFrame) -> dict:
    return profile_dataset(demo_df)


class TestDemoDatasetShape:
    """Ground truth fact #4 and #6 from benchmark_status.md: 4,000 rows (not the
    10,000,000 a misleading benchmark question claimed), and full date coverage.

    Note: this worktree's `profile_dataset` (backend/app/tools/profiler.py, owned by
    EDA-ANALYST, not modified here) predates the `date_ranges`/`text_columns` keys
    that a later commit on main adds - it only buckets numeric/categorical/
    datetime/boolean columns at the top level, with high-cardinality non-numeric
    columns (role == "text") left out of every bucket entirely and only visible via
    per-column `column_info`. These tests are written against what this worktree's
    tool actually returns, verified directly rather than assumed."""

    def test_row_and_column_count(self, demo_profile: dict) -> None:
        assert demo_profile["rows"] == 4000
        assert demo_profile["columns"] == 11

    def test_date_coverage(self, demo_df: pd.DataFrame) -> None:
        # This worktree's profile_dataset does not surface a date_ranges dict, so
        # date coverage is verified directly against the loaded dataframe (the same
        # data profile_dataset itself reads from) and cross-checked against
        # compare_periods, which is exercised independently below.
        dates = pd.to_datetime(demo_df["date"], errors="coerce")
        assert dates.min().strftime("%Y-%m-%d") == "2024-01-01"
        assert dates.max().strftime("%Y-%m-%d") == "2025-12-31"

    def test_has_customer_id_but_no_explicit_segment_column(self, demo_df: pd.DataFrame, demo_profile: dict) -> None:
        assert "customer_id" in demo_df.columns
        assert demo_df["customer_id"].nunique() == 1144
        # customer_id is high-cardinality (mostly-unique IDs), so profile_dataset's
        # role classifier buckets it as "text" - which this worktree's version drops
        # from every named top-level bucket (numeric/categorical/date/boolean).
        # Confirm that directly via column_info's per-column role field instead.
        customer_id_info = next(c for c in demo_profile["column_info"] if c["name"] == "customer_id")
        assert customer_id_info["role"] == "text"
        assert customer_id_info["unique_count"] == 1144
        assert "customer_id" not in demo_profile["numeric_columns"]
        assert "customer_id" not in demo_profile["categorical_columns"]
        assert "customer_id" not in demo_profile["date_columns"]

        all_named_columns = {
            *demo_profile["numeric_columns"],
            *demo_profile["categorical_columns"],
            *demo_profile["date_columns"],
            *demo_profile["boolean_columns"],
        }
        assert not any("segment" in col.lower() for col in all_named_columns)
        assert not any("segment" in c["name"].lower() for c in demo_profile["column_info"])

    def test_no_marketing_or_conversion_column(self, demo_df: pd.DataFrame) -> None:
        lowered = {str(c).lower() for c in demo_df.columns}
        assert not any("campaign" in c or "conversion" in c or "marketing" in c for c in lowered)


class TestMissingValuesGroundTruth:
    """Ground truth fact #1: 25 missing values total, all in customer_id, 0.62%."""

    def test_missing_total(self, demo_profile: dict) -> None:
        assert demo_profile["missing_total"] == 25

    def test_missing_values_are_all_in_customer_id(self, demo_profile: dict) -> None:
        columns_with_missing = [c for c in demo_profile["column_info"] if c["missing_count"] > 0]
        assert len(columns_with_missing) == 1
        assert columns_with_missing[0]["name"] == "customer_id"
        assert columns_with_missing[0]["missing_count"] == 25
        assert columns_with_missing[0]["missing_pct"] == 0.62


class TestCompareQ2VsQ1_2025(object):
    """Ground truth fact #2: the exact 'false premise' benchmark case. A benchmark
    question claimed revenue fell 18% quarter-over-quarter; the real figure is -5.15%."""

    @pytest.fixture(scope="class")
    def result(self, demo_df: pd.DataFrame) -> dict:
        return compare_periods(
            demo_df,
            date_column="date",
            value_column="revenue",
            current_start="2025-04-01",
            current_end="2025-06-30",
            previous_start="2025-01-01",
            previous_end="2025-03-31",
            agg_func="sum",
        )

    def test_current_period(self, result: dict) -> None:
        assert result["current_period"]["value"] == pytest.approx(216772.38)
        assert result["current_period"]["n"] == 566

    def test_previous_period(self, result: dict) -> None:
        assert result["previous_period"]["value"] == pytest.approx(228554.00)
        assert result["previous_period"]["n"] == 541

    def test_delta_and_pct_change(self, result: dict) -> None:
        assert result["delta"] == pytest.approx(-11781.62)
        assert result["pct_change"] == pytest.approx(-5.15)
        # The specific false claim under test: revenue did NOT fall 18%.
        assert result["pct_change"] != pytest.approx(-18.0, abs=1.0)


class TestCompareFullYear2025Vs2024:
    """Ground truth fact #3: full-year comparison. A benchmark question presupposed a
    revenue decline; revenue actually grew +1.7%."""

    @pytest.fixture(scope="class")
    def result(self, demo_df: pd.DataFrame) -> dict:
        return compare_periods(
            demo_df,
            date_column="date",
            value_column="revenue",
            current_start="2025-01-01",
            current_end="2025-12-31",
            previous_start="2024-01-01",
            previous_end="2024-12-31",
            agg_func="sum",
        )

    def test_current_period(self, result: dict) -> None:
        assert result["current_period"]["value"] == pytest.approx(765856.82)
        assert result["current_period"]["n"] == 1993

    def test_previous_period(self, result: dict) -> None:
        assert result["previous_period"]["value"] == pytest.approx(753086.13)
        assert result["previous_period"]["n"] == 2007

    def test_delta_and_pct_change_show_growth_not_decline(self, result: dict) -> None:
        assert result["delta"] == pytest.approx(12770.69)
        assert result["pct_change"] == pytest.approx(1.7)
        # The specific false premise under test: revenue GREW, it did not decline.
        assert result["delta"] > 0
        assert result["pct_change"] > 0
