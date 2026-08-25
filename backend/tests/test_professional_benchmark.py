"""Phase 3C professional analyst benchmark: schema validation + real, measured scoring
run (BENCHMARK-ENGINEER).

Two things happen here, deliberately kept in one file so a single `pytest` run always
produces a fresh, trustworthy result:

1. Schema-sanity tests (mirrors `tests/reasoning/test_reasoning_benchmark_fixtures.py`'s
   pattern) -- prove `professional_benchmark.json` is well-formed *before* trusting a
   scoring run against it: every case has the required keys, references real
   `ToolCategory`/classification/causal-behavior values, category coverage meets the
   5+-per-category requirement, and the scripted-case count meets the 25+ requirement.
2. A real scoring run: every case is driven through `tests.benchmark.scoring.run_case`
   against the real `ReasoningOrchestrator` (scripted cases) or the real
   `premise_validator` (deterministic-only cases), using the real primary dataset
   (`data/demo/sales_data.xlsx`) or its date-aggregated view (see `_daily_record`
   below -- required because the forecasting tools reject multiple rows sharing one
   date, so a real forecasting workflow against this dataset needs pre-aggregation).
   `scoring.summarize()` produces the actual report, written to
   `tests/benchmark/professional_benchmark_results.json` as a side effect so the exact
   measured numbers are available afterward without re-running anything.

Per the project's explicit instruction: this file does NOT hard-assert a 100% (or any
other invented) pass rate. It asserts a reasonable minimum (`overall_score_pct >= 60.0`)
so a real regression still fails the suite loudly, while leaving room for the benchmark
to honestly report gaps rather than being tuned to always pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.datasets.storage import DatasetRecord
from app.reasoning.categories import ToolCategory
from tests.benchmark.scoring import run_case, summarize

_BENCH_DIR = Path(__file__).resolve().parent / "benchmark"
_FIXTURE_PATH = _BENCH_DIR / "professional_benchmark.json"
_RESULTS_PATH = _BENCH_DIR / "professional_benchmark_results.json"
_DEMO_XLSX_PATH = Path(__file__).resolve().parents[2] / "data" / "demo" / "sales_data.xlsx"

_REQUIRED_CASE_KEYS = {
    "case_id",
    "category",
    "user_question",
    "expected_tool_category",
    "required_constraints",
    "expected_classifications",
    "expected_limitations",
    "expected_causal_behavior",
}
_VALID_CLASSIFICATIONS = {"FACT", "CALCULATED_RESULT", "STATISTICAL_RESULT", "HYPOTHESIS", "ASSUMPTION", "UNKNOWN"}
_VALID_LIMITATION_CATEGORIES = {
    "missing_data",
    "insufficient_coverage",
    "sample_size",
    "unavailable_capability",
    "methodological",
    "resource_limit",
    "other",
}
_VALID_LIMITATION_SEVERITIES = {"blocks_conclusion", "reduces_confidence", "minor"}
_VALID_CAUSAL_BEHAVIORS = {
    "not_applicable",
    "must_hedge_unless_causal_hypothesis_supported",
    "must_generate_2_to_4_competing_hypotheses",
}
_REQUIRED_CATEGORIES = {
    "data_understanding",
    "eda",
    "sql",
    "statistics",
    "forecasting",
    "business_diagnosis",
    "segmentation",
    "data_quality",
    "executive_reporting",
    "reasoning_traps",
}
_MIN_CASES_PER_CATEGORY = 5
_MIN_TOTAL_CASES = 50
_MIN_SCRIPTED_CASES = 25
_MIN_OVERALL_SCORE_PCT = 60.0


# --- fixtures ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_data() -> dict:
    with open(_FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def primary_df() -> pd.DataFrame:
    return pd.read_excel(_DEMO_XLSX_PATH)


@pytest.fixture(scope="module")
def primary_record(primary_df: pd.DataFrame) -> DatasetRecord:
    return DatasetRecord(
        id="professional-benchmark-primary",
        original_filename="sales_data.xlsx",
        extension=".xlsx",
        uploaded_at=pd.Timestamp.utcnow(),
        df=primary_df,
        stored_path="unused",
    )


@pytest.fixture(scope="module")
def primary_daily_record(primary_df: pd.DataFrame) -> DatasetRecord:
    """Date-aggregated view of the primary dataset (one row per date, summed
    revenue/quantity/cost/profit) -- required for forecasting-category cases. The raw
    transaction-level dataset has ~5.5 rows per date on average, and every forecasting
    tool (`forecast`, `decompose_timeseries`, `backtest_forecast`,
    `train_test_split_timeseries`) raises `ToolExecutionError` on a date column with
    duplicate timestamps (verified directly against `app.tools.forecasting` while
    authoring these cases) -- so a real forecasting workflow against this dataset
    needs this aggregation step first, same as a competent analyst would do."""
    daily = primary_df.groupby("date", as_index=False).agg(
        revenue=("revenue", "sum"),
        quantity=("quantity", "sum"),
        profit=("profit", "sum"),
        cost=("cost", "sum"),
    )
    return DatasetRecord(
        id="professional-benchmark-primary-daily",
        original_filename="sales_data_daily.csv",
        extension=".csv",
        uploaded_at=pd.Timestamp.utcnow(),
        df=daily,
        stored_path="unused",
    )


@pytest.fixture(scope="module")
def records(primary_record: DatasetRecord, primary_daily_record: DatasetRecord) -> dict[str, DatasetRecord]:
    return {"primary": primary_record, "primary_daily": primary_daily_record}


# --- schema-sanity tests (fast, no orchestrator run) ---------------------------------


def test_fixture_file_is_valid_json_with_cases(fixture_data):
    assert isinstance(fixture_data["cases"], list)
    assert len(fixture_data["cases"]) >= _MIN_TOTAL_CASES


def test_every_case_has_all_required_fields(fixture_data):
    for case in fixture_data["cases"]:
        missing = _REQUIRED_CASE_KEYS - set(case.keys())
        assert not missing, f"case {case.get('case_id')} missing fields: {missing}"


def test_case_ids_are_unique(fixture_data):
    ids = [c["case_id"] for c in fixture_data["cases"]]
    assert len(ids) == len(set(ids))


def test_expected_tool_categories_are_real_categories(fixture_data):
    valid = {c.value for c in ToolCategory}
    for case in fixture_data["cases"]:
        for cat in case["expected_tool_category"]:
            assert cat in valid, f"case {case['case_id']} references unknown category {cat!r}"


def test_expected_classifications_are_real(fixture_data):
    for case in fixture_data["cases"]:
        for classification in case["expected_classifications"]:
            assert classification in _VALID_CLASSIFICATIONS, (
                f"case {case['case_id']} has unknown classification {classification!r}"
            )


def test_expected_limitations_are_well_formed(fixture_data):
    for case in fixture_data["cases"]:
        for lim in case["expected_limitations"]:
            assert lim.get("category") in _VALID_LIMITATION_CATEGORIES, (
                f"case {case['case_id']} has unknown limitation category {lim.get('category')!r}"
            )
            assert lim.get("severity") in _VALID_LIMITATION_SEVERITIES, (
                f"case {case['case_id']} has unknown limitation severity {lim.get('severity')!r}"
            )


def test_expected_causal_behavior_is_valid(fixture_data):
    for case in fixture_data["cases"]:
        assert case["expected_causal_behavior"] in _VALID_CAUSAL_BEHAVIORS, (
            f"case {case['case_id']} has unknown causal behavior {case['expected_causal_behavior']!r}"
        )


def test_at_least_five_cases_per_required_category(fixture_data):
    counts: dict[str, int] = {}
    for case in fixture_data["cases"]:
        counts[case["category"]] = counts.get(case["category"], 0) + 1
    missing = _REQUIRED_CATEGORIES - set(counts)
    assert not missing, f"benchmark missing coverage for categories: {missing}"
    under_covered = {cat: n for cat, n in counts.items() if cat in _REQUIRED_CATEGORIES and n < _MIN_CASES_PER_CATEGORY}
    assert not under_covered, f"categories with fewer than {_MIN_CASES_PER_CATEGORY} cases: {under_covered}"


def test_at_least_25_cases_have_a_script(fixture_data):
    scripted = sum(1 for c in fixture_data["cases"] if c.get("script"))
    assert scripted >= _MIN_SCRIPTED_CASES, f"only {scripted} scripted cases, need >= {_MIN_SCRIPTED_CASES}"


def test_every_case_dataset_reference_is_known(fixture_data):
    known = {"primary", "primary_daily"}
    for case in fixture_data["cases"]:
        assert case.get("dataset", "primary") in known, f"case {case['case_id']} references unknown dataset {case.get('dataset')!r}"


def test_primary_dataset_facts_match_the_real_demo_file(fixture_data, primary_df):
    facts = fixture_data["primary_dataset_facts"]
    assert len(primary_df) == facts["rows"]
    assert list(primary_df.columns) == facts["column_names"]


def test_no_pre_claimed_benchmark_score_in_the_fixture_itself(fixture_data):
    """The fixture file describes cases, not results -- the actual score is only ever
    produced by actually running the cases (see test_full_benchmark_run below), never
    hand-entered into the fixture file itself."""
    assert "score" not in fixture_data
    assert "pass_rate" not in fixture_data
    assert "overall_score_pct" not in fixture_data
    for case in fixture_data["cases"]:
        assert "actual_result" not in case
        assert "verdict" not in case


# --- the real scoring run --------------------------------------------------------------


@pytest.fixture(scope="module")
def benchmark_report(fixture_data, records) -> dict:
    """Runs every case in the fixture through the real scoring.run_case() (which
    drives the real, unmodified ReasoningOrchestrator for scripted cases, and the
    real, unmodified premise_validator for deterministic-only cases), then
    summarizes. Computed once per test session (module-scoped) and reused by every
    test below plus the results-file side effect, so the suite doesn't re-run the
    (non-trivial: real ARIMA fits, real regressions, real clustering) orchestrator
    work multiple times for what is fundamentally one benchmark pass."""
    cases = fixture_data["cases"]
    cases_by_id = {c["case_id"]: c for c in cases}
    results = []
    for case in cases:
        record = records[case.get("dataset", "primary")]
        results.append(run_case(case, record))
    return summarize(results, cases_by_id)


def test_full_benchmark_run_covers_every_case(fixture_data, benchmark_report):
    assert benchmark_report["total_tasks"] == len(fixture_data["cases"])
    assert benchmark_report["total_tasks"] >= _MIN_TOTAL_CASES


def test_full_benchmark_run_report_has_the_expected_shape(benchmark_report):
    for key in ("total_tasks", "passed", "partial", "failed", "overall_score_pct", "category_scores_pct", "cases"):
        assert key in benchmark_report
    assert benchmark_report["passed"] + benchmark_report["partial"] + benchmark_report["failed"] == benchmark_report["total_tasks"]
    assert set(benchmark_report["category_scores_pct"]) == _REQUIRED_CATEGORIES
    assert len(benchmark_report["cases"]) == benchmark_report["total_tasks"]
    for entry in benchmark_report["cases"]:
        assert entry["verdict"] in ("PASS", "PARTIAL", "FAIL")


def test_full_benchmark_run_meets_a_reasonable_minimum_pass_rate(benchmark_report):
    """Deliberately NOT asserting 100% (or any other invented ceiling) -- a real
    regression in the reasoning pipeline must still fail this test loudly, but the
    benchmark is allowed to honestly report a real gap rather than being tuned to
    always pass. See the module docstring and the task's own honesty requirement."""
    assert benchmark_report["overall_score_pct"] >= _MIN_OVERALL_SCORE_PCT, (
        f"overall_score_pct={benchmark_report['overall_score_pct']} fell below the "
        f"{_MIN_OVERALL_SCORE_PCT}% minimum -- see "
        f"{_RESULTS_PATH} for the full per-case breakdown."
    )


def test_full_benchmark_run_writes_the_real_measured_results_to_disk(benchmark_report):
    """Side effect: the real, measured summarize() output is written to disk so the
    orchestrator (or anyone) can read the actual numbers afterward without re-running
    the (slow: real statistical/ML tool executions) benchmark."""
    _RESULTS_PATH.write_text(json.dumps(benchmark_report, indent=2), encoding="utf-8")
    assert _RESULTS_PATH.exists()
    reloaded = json.loads(_RESULTS_PATH.read_text(encoding="utf-8"))
    assert reloaded["total_tasks"] == benchmark_report["total_tasks"]
