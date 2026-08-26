"""Master Mission final deliverable: the unified 100+-case benchmark requested across
all four master-mission messages (the original 24-category ask, the "at least 100
cases" addition, and the professional-analyst-capability-map coverage requirement).

Mirrors `test_professional_benchmark.py`'s proven two-part structure exactly (schema
sanity, then a real scoring run against the real `ReasoningOrchestrator` /
`premise_validator`) against a SEPARATE fixture file (`final_100_cases.json`,
102 cases across 24 categories) rather than modifying the existing, already-passing
`professional_benchmark.json` -- this is a new, broader benchmark, not a replacement.

Forecasting-category cases (and any other case whose scripted tool call needs one row
per date) use the `primary_daily` dataset view, for the same documented reason
`test_professional_benchmark.py` does: the raw transaction-level dataset has ~3.6 rows
per date on average, and every forecasting tool raises `ToolExecutionError` on
duplicate-timestamp date columns.

Per the project's explicit honesty requirement: this file does NOT hard-assert a 100%
(or any other invented) pass rate, and a scripted 100/100 result -- if it happened --
would NOT be described anywhere as "professional analyst level" (that phrase requires
real-LLM evidence, which this deterministic suite deliberately does not claim to be).
The minimum floor below is a regression guard, not a target.
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
_FIXTURE_PATH = _BENCH_DIR / "final_100_cases.json"
_RESULTS_PATH = _BENCH_DIR / "final_100_cases_results.json"
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
    "data_profiling",
    "data_cleaning_quality",
    "eda_distributions",
    "outlier_anomaly",
    "sql_analysis",
    "aggregation_grouping",
    "statistical_testing",
    "correlation",
    "regression_diagnostics",
    "forecasting_timeseries",
    "clustering_pca",
    "segmentation_rfm_cohort",
    "business_kpi_contribution_pareto",
    "executive_summary_recommendations",
    "visualization_chart_selection",
    "causal_reasoning",
    "bias_traps",
    "ambiguous_questions",
    "insufficient_data",
    "contradictory_data",
    "adversarial_prompt_injection",
    "large_data_scale_awareness",
    "numerical_sanity_checks",
    "recommendation_grounding",
}
_MIN_CASES_PER_CATEGORY = 3
_MIN_TOTAL_CASES = 100
_MIN_SCRIPTED_CASES = 60
_MIN_OVERALL_SCORE_PCT = 80.0  # regression guard, not a target -- see module docstring.
# Real measured score at the time this was set: 99.0% (101/102 PASS, 1 PARTIAL). The
# floor is deliberately well below that (matching the ~20-point margin
# test_professional_benchmark.py's own 60.0 floor leaves below its measured 100.0%)
# so a real regression still fails loudly without the threshold being brittle to
# normal day-to-day variance in a scripted-but-real orchestrator run.


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
        id="final-100-benchmark-primary",
        original_filename="sales_data.xlsx",
        extension=".xlsx",
        uploaded_at=pd.Timestamp.utcnow(),
        df=primary_df,
        stored_path="unused",
    )


@pytest.fixture(scope="module")
def primary_daily_record(primary_df: pd.DataFrame) -> DatasetRecord:
    """Date-aggregated view -- required for forecasting-category cases. See module
    docstring and the identical fixture in test_professional_benchmark.py."""
    daily = primary_df.groupby("date", as_index=False).agg(
        revenue=("revenue", "sum"),
        quantity=("quantity", "sum"),
        profit=("profit", "sum"),
        cost=("cost", "sum"),
    )
    return DatasetRecord(
        id="final-100-benchmark-primary-daily",
        original_filename="sales_data_daily.csv",
        extension=".csv",
        uploaded_at=pd.Timestamp.utcnow(),
        df=daily,
        stored_path="unused",
    )


@pytest.fixture(scope="module")
def tiny_record(primary_daily_record: DatasetRecord) -> DatasetRecord:
    """First 4 rows of the daily-aggregated view -- genuinely below every forecasting
    tool's `_MIN_POINTS` (10) minimum, so a case that needs the REAL `ToolExecutionError`
    refusal path (not just a scripted claim of one) has a dataset that actually
    triggers it. See insuf1 in final_100_cases.json."""
    return DatasetRecord(
        id="final-100-benchmark-tiny",
        original_filename="sales_data_daily_tiny.csv",
        extension=".csv",
        uploaded_at=pd.Timestamp.utcnow(),
        df=primary_daily_record.df.head(4).reset_index(drop=True),
        stored_path="unused",
    )


@pytest.fixture(scope="module")
def records(
    primary_record: DatasetRecord, primary_daily_record: DatasetRecord, tiny_record: DatasetRecord
) -> dict[str, DatasetRecord]:
    return {"primary": primary_record, "primary_daily": primary_daily_record, "tiny": tiny_record}


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


def test_every_required_category_has_minimum_coverage(fixture_data):
    counts: dict[str, int] = {}
    for case in fixture_data["cases"]:
        counts[case["category"]] = counts.get(case["category"], 0) + 1
    missing = _REQUIRED_CATEGORIES - set(counts)
    assert not missing, f"benchmark missing coverage for categories: {missing}"
    under_covered = {cat: n for cat, n in counts.items() if cat in _REQUIRED_CATEGORIES and n < _MIN_CASES_PER_CATEGORY}
    assert not under_covered, f"categories with fewer than {_MIN_CASES_PER_CATEGORY} cases: {under_covered}"


def test_at_least_60_cases_have_a_script(fixture_data):
    scripted = sum(1 for c in fixture_data["cases"] if c.get("script"))
    assert scripted >= _MIN_SCRIPTED_CASES, f"only {scripted} scripted cases, need >= {_MIN_SCRIPTED_CASES}"


def test_every_case_dataset_reference_is_known(fixture_data):
    known = {"primary", "primary_daily", "tiny"}
    for case in fixture_data["cases"]:
        assert case.get("dataset", "primary") in known, f"case {case['case_id']} references unknown dataset {case.get('dataset')!r}"


def test_primary_dataset_facts_match_the_real_demo_file(fixture_data, primary_df):
    facts = fixture_data["primary_dataset_facts"]
    assert len(primary_df) == facts["rows"]
    assert list(primary_df.columns) == facts["column_names"]


def test_no_pre_claimed_benchmark_score_in_the_fixture_itself(fixture_data):
    assert "score" not in fixture_data
    assert "pass_rate" not in fixture_data
    assert "overall_score_pct" not in fixture_data
    for case in fixture_data["cases"]:
        assert "actual_result" not in case
        assert "verdict" not in case


# --- the real scoring run --------------------------------------------------------------


@pytest.fixture(scope="module")
def benchmark_report(fixture_data, records) -> dict:
    """Runs every case through the real, unmodified scoring.run_case() -- see
    test_professional_benchmark.py's identical fixture for the full rationale.
    Module-scoped: this is a genuinely slow real run (real statistical tests,
    regressions, ARIMA fits, clustering across 102 cases), computed once."""
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
    """Deliberately NOT asserting 100% (or any other invented ceiling) -- see module
    docstring. A real regression must still fail this test loudly."""
    assert benchmark_report["overall_score_pct"] >= _MIN_OVERALL_SCORE_PCT, (
        f"overall_score_pct={benchmark_report['overall_score_pct']} fell below the "
        f"{_MIN_OVERALL_SCORE_PCT}% minimum -- see {_RESULTS_PATH} for the full per-case breakdown."
    )


def test_full_benchmark_run_writes_the_real_measured_results_to_disk(benchmark_report):
    _RESULTS_PATH.write_text(json.dumps(benchmark_report, indent=2), encoding="utf-8")
    assert _RESULTS_PATH.exists()
    reloaded = json.loads(_RESULTS_PATH.read_text(encoding="utf-8"))
    assert reloaded["total_tasks"] == benchmark_report["total_tasks"]
