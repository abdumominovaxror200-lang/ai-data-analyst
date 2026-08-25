"""Phase 4 REAL-LLM-BENCHMARK-ENGINEER: real, live-Groq benchmark run.

Unlike `tests/test_professional_benchmark.py` (which drives a scripted `MockProvider`
standing in for "a plausible LLM response"), every test in this file that reaches
`benchmark_report` makes REAL network calls to the project's configured LLM provider
(Groq, `openai/gpt-oss-120b`) via the real, unmodified `ReasoningOrchestrator`. This
file must NEVER run as part of an ordinary `pytest -q` pass -- the whole module is
gated behind `pytestmark = skip_unless_real_llm` (see `runner.py`), which only lifts
when a human explicitly sets `RUN_REAL_LLM_BENCHMARK=1` in the environment.

Two things happen here, same "one file, one trustworthy run" pattern as the scripted
benchmark:

1. Schema-sanity tests (fast, no network calls) -- prove `real_llm_cases.json` is
   well-formed *before* trusting a live run against it: every case has the required
   keys, none of them smuggle in a `"script"` key (this module reuses
   `validate_real_case_schema` from `runner.py` for exactly that check), category
   coverage meets the 12-required-category / 2+-per-category bar, and referenced
   `ToolCategory`/classification/limitation/causal-behavior values are all real.
2. A real scoring run: every case is driven through `runner.run_real_case()` against
   the real `ReasoningOrchestrator` and real Groq calls, using either the real primary
   dataset (`data/demo/sales_data.xlsx`), its date-aggregated view (required for the
   forecasting-category cases, since those tools reject duplicate per-date rows), or
   one of two small synthetic DataFrames built directly below for the two
   insufficient-data cases that need a guaranteed-small sample regardless of what the
   real 4000-row file happens to contain. `scoring.summarize()` produces the actual
   report, written to `real_llm_results.json` as a side effect.

Per the task's explicit instruction: this file does NOT hard-assert a high pass rate.
A real model WILL sometimes disagree with authored expectations in ways worth seeing,
not hiding behind a lenient threshold tuned to always pass -- so the only assertions
here are structural sanity (right number of cases ran, report has the right shape),
never a minimum overall_score_pct.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.datasets.storage import DatasetRecord
from app.reasoning.categories import ToolCategory
from tests.benchmark.real_llm.runner import run_real_case, skip_unless_real_llm, validate_real_case_schema
from tests.benchmark.scoring import summarize

pytestmark = skip_unless_real_llm

_REAL_LLM_DIR = Path(__file__).resolve().parent
_FIXTURE_PATH = _REAL_LLM_DIR / "real_llm_cases.json"
_RESULTS_PATH = _REAL_LLM_DIR / "real_llm_results.json"
_DEMO_XLSX_PATH = Path(__file__).resolve().parents[4] / "data" / "demo" / "sales_data.xlsx"

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
    "descriptive_analysis",
    "sql_analysis",
    "statistical_testing",
    "regression",
    "forecasting",
    "clustering_segmentation",
    "eda",
    "data_quality",
    "ambiguous_questions",
    "causal_traps",
    "insufficient_data",
    "business_recommendations",
}
_MIN_CASES_PER_CATEGORY = 2
_MIN_TOTAL_CASES = 30


# --- fixtures: case data ------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_data() -> dict:
    with open(_FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


# --- fixtures: datasets ---------------------------------------------------------------


@pytest.fixture(scope="module")
def primary_df() -> pd.DataFrame:
    return pd.read_excel(_DEMO_XLSX_PATH)


@pytest.fixture(scope="module")
def primary_record(primary_df: pd.DataFrame) -> DatasetRecord:
    return DatasetRecord(
        id="real-llm-benchmark-primary",
        original_filename="sales_data.xlsx",
        extension=".xlsx",
        uploaded_at=pd.Timestamp.utcnow(),
        df=primary_df,
        stored_path="unused",
    )


@pytest.fixture(scope="module")
def primary_daily_record(primary_df: pd.DataFrame) -> DatasetRecord:
    """Date-aggregated view (one row per date) -- required for forecasting-category
    cases. Same reasoning as test_professional_benchmark.py's identical fixture: the
    raw transaction-level file has ~5.5 rows/date on average, and every forecasting
    tool raises ToolExecutionError on a date column with duplicate timestamps."""
    daily = primary_df.groupby("date", as_index=False).agg(
        revenue=("revenue", "sum"),
        quantity=("quantity", "sum"),
        profit=("profit", "sum"),
        cost=("cost", "sum"),
    )
    return DatasetRecord(
        id="real-llm-benchmark-primary-daily",
        original_filename="sales_data_daily.csv",
        extension=".csv",
        uploaded_at=pd.Timestamp.utcnow(),
        df=daily,
        stored_path="unused",
    )


@pytest.fixture(scope="module")
def tiny_forecast_record() -> DatasetRecord:
    """6 daily points -- deliberately below app/tools/forecasting.py's _MIN_POINTS=10
    floor, so every FORECASTING-category tool (forecast/decompose_timeseries/
    backtest_forecast/train_test_split_timeseries) refuses with a real
    ToolExecutionError. Used by case id_01 to verify the real, load-bearing
    insufficient-data refusal path -- not a filtered slice of the real dataset,
    because that would not reliably guarantee fewer than 10 points."""
    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    revenue = [980.0, 1050.0, 1010.0, 1100.0, 1020.0, 1080.0]
    df = pd.DataFrame({"date": dates, "revenue": revenue})
    return DatasetRecord(
        id="real-llm-benchmark-tiny-forecast",
        original_filename="tiny_forecast.csv",
        extension=".csv",
        uploaded_at=pd.Timestamp.utcnow(),
        df=df,
        stored_path="unused",
    )


@pytest.fixture(scope="module")
def tiny_pilot_record() -> DatasetRecord:
    """6 rows, same 11-column schema as the real primary dataset (region='Pilot') --
    used by case id_02 to verify app/reasoning/verifier.py's automatic 'sample_size'
    Limitation (fires when a STATISTICAL_RESULT Evidence reports sample_size < 10).
    A synthetic DataFrame, not a filtered slice of the real data, so the sample size
    is exactly and reliably 6 regardless of what the real file happens to contain."""
    rng = np.random.default_rng(7)
    n = 6
    dates = pd.date_range("2026-02-01", periods=n, freq="D")
    revenue = rng.normal(450, 60, n).round(2)
    unit_price = rng.normal(45, 5, n).round(2)
    quantity = rng.integers(1, 10, n)
    cost = (revenue * 0.55).round(2)
    df = pd.DataFrame(
        {
            "date": dates,
            "product": ["Pilot Widget"] * n,
            "category": ["Home & Office"] * n,
            "region": ["Pilot"] * n,
            "salesperson": ["H. Pilotova"] * n,
            "quantity": quantity,
            "unit_price": unit_price,
            "revenue": revenue,
            "cost": cost,
            "profit": (revenue - cost).round(2),
            "customer_id": [f"PILOT-{i}" for i in range(n)],
        }
    )
    return DatasetRecord(
        id="real-llm-benchmark-tiny-pilot",
        original_filename="tiny_pilot.csv",
        extension=".csv",
        uploaded_at=pd.Timestamp.utcnow(),
        df=df,
        stored_path="unused",
    )


@pytest.fixture(scope="module")
def records(
    primary_record: DatasetRecord,
    primary_daily_record: DatasetRecord,
    tiny_forecast_record: DatasetRecord,
    tiny_pilot_record: DatasetRecord,
) -> dict[str, DatasetRecord]:
    return {
        "primary": primary_record,
        "primary_daily": primary_daily_record,
        "tiny_forecast": tiny_forecast_record,
        "tiny_pilot": tiny_pilot_record,
    }


# --- schema-sanity tests (fast, no network calls) ------------------------------------


def test_fixture_file_is_valid_json_with_cases(fixture_data):
    assert isinstance(fixture_data["cases"], list)
    assert len(fixture_data["cases"]) >= _MIN_TOTAL_CASES


def test_every_case_has_all_required_fields(fixture_data):
    for case in fixture_data["cases"]:
        missing = _REQUIRED_CASE_KEYS - set(case.keys())
        assert not missing, f"case {case.get('case_id')} missing fields: {missing}"


def test_no_case_defines_a_script(fixture_data):
    """The defining difference from the scripted benchmark: real-LLM cases must NOT
    include a 'script' key -- there is no MockProvider to script. Reuses the runner's
    own validator so this test and the live run enforce the identical rule."""
    for case in fixture_data["cases"]:
        validate_real_case_schema(case)  # raises ValueError if 'script' is present or a key is missing


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


def test_all_12_required_categories_are_covered_with_a_minimum_each(fixture_data):
    counts: dict[str, int] = {}
    for case in fixture_data["cases"]:
        counts[case["category"]] = counts.get(case["category"], 0) + 1
    missing = _REQUIRED_CATEGORIES - set(counts)
    assert not missing, f"benchmark missing coverage for categories: {missing}"
    under_covered = {cat: n for cat, n in counts.items() if cat in _REQUIRED_CATEGORIES and n < _MIN_CASES_PER_CATEGORY}
    assert not under_covered, f"categories with fewer than {_MIN_CASES_PER_CATEGORY} cases: {under_covered}"


def test_every_case_dataset_reference_is_known(fixture_data):
    known = {"primary", "primary_daily", "tiny_forecast", "tiny_pilot"}
    for case in fixture_data["cases"]:
        assert case.get("dataset", "primary") in known, f"case {case['case_id']} references unknown dataset {case.get('dataset')!r}"


def test_no_pre_claimed_benchmark_score_in_the_fixture_itself(fixture_data):
    """Same honesty rule as the scripted benchmark's fixture: the fixture describes
    cases, not results. The actual score is only ever produced by actually running the
    cases live (see test_full_live_benchmark_run below)."""
    assert "score" not in fixture_data
    assert "pass_rate" not in fixture_data
    assert "overall_score_pct" not in fixture_data
    for case in fixture_data["cases"]:
        assert "actual_result" not in case
        assert "verdict" not in case


def test_primary_dataset_facts_match_the_real_demo_file(fixture_data, primary_df):
    facts = fixture_data["primary_dataset_facts"]
    assert len(primary_df) == facts["rows"]
    assert list(primary_df.columns) == facts["column_names"]


def test_tiny_fixtures_are_actually_below_the_thresholds_they_claim_to_test(tiny_forecast_record, tiny_pilot_record):
    """Sanity-checks the synthetic fixtures themselves, independent of any live call:
    tiny_forecast_record must have fewer than forecasting.py's 10-point floor, and
    tiny_pilot_record must have fewer than verifier.py's 10-sample threshold."""
    assert len(tiny_forecast_record.df) < 10
    assert len(tiny_pilot_record.df) < 10


# --- the real, live scoring run -------------------------------------------------------


@pytest.fixture(scope="module")
def benchmark_report(fixture_data, records) -> dict:
    """Runs every case in the fixture through the REAL `run_real_case()` -- real Groq
    calls, real tool execution, real `ReasoningOrchestrator` -- then summarizes with
    the identical `tests.benchmark.scoring.summarize()` the scripted benchmark uses,
    so the two reports are directly comparable in shape. Computed once per test
    session (module-scoped) and reused by every test below plus the results-file side
    effect, so this genuinely non-trivial live run (real network calls, real
    statistical/ML tool executions) happens exactly once per `pytest` invocation of
    this file, not once per test."""
    cases = fixture_data["cases"]
    cases_by_id = {c["case_id"]: c for c in cases}
    results = []
    for case in cases:
        record = records[case.get("dataset", "primary")]
        results.append(run_real_case(case, record, retries=1))
    return summarize(results, cases_by_id)


def test_full_live_benchmark_run_covers_every_case(fixture_data, benchmark_report):
    assert benchmark_report["total_tasks"] == len(fixture_data["cases"])
    assert benchmark_report["total_tasks"] >= _MIN_TOTAL_CASES


def test_full_live_benchmark_run_report_has_the_expected_shape(benchmark_report):
    for key in ("total_tasks", "passed", "partial", "failed", "overall_score_pct", "category_scores_pct", "cases"):
        assert key in benchmark_report
    assert benchmark_report["passed"] + benchmark_report["partial"] + benchmark_report["failed"] == benchmark_report["total_tasks"]
    assert set(benchmark_report["category_scores_pct"]) == _REQUIRED_CATEGORIES
    assert len(benchmark_report["cases"]) == benchmark_report["total_tasks"]
    for entry in benchmark_report["cases"]:
        assert entry["verdict"] in ("PASS", "PARTIAL", "FAIL")


def test_full_live_benchmark_run_writes_the_real_measured_results_to_disk(benchmark_report):
    """Side effect: the real, measured summarize() output -- from real live Groq
    calls, not a mock -- is written to disk so the actual numbers are available
    afterward without re-running the (slow, real-network, rate-limited) live suite.

    Deliberately NOT asserting any minimum overall_score_pct here (unlike the scripted
    benchmark's test_full_benchmark_run_meets_a_reasonable_minimum_pass_rate): a real
    model disagreeing with an authored expectation is real, useful signal about the
    reasoning layer's actual behavior against a live LLM, not a regression to hide
    behind a lenient threshold. The honest numbers are whatever this prints and writes."""
    _RESULTS_PATH.write_text(json.dumps(benchmark_report, indent=2), encoding="utf-8")
    assert _RESULTS_PATH.exists()
    reloaded = json.loads(_RESULTS_PATH.read_text(encoding="utf-8"))
    assert reloaded["total_tasks"] == benchmark_report["total_tasks"]

    print("\n" + "=" * 88)
    print("REAL-LLM BENCHMARK RESULTS (live Groq run, not scripted)")
    print("=" * 88)
    print(f"total_tasks:       {benchmark_report['total_tasks']}")
    print(f"passed:            {benchmark_report['passed']}")
    print(f"partial:           {benchmark_report['partial']}")
    print(f"failed:            {benchmark_report['failed']}")
    print(f"overall_score_pct: {benchmark_report['overall_score_pct']}")
    print("-" * 88)
    print("category_scores_pct:")
    for cat, pct in sorted(benchmark_report["category_scores_pct"].items()):
        print(f"  {cat:28s} {pct:6.1f}%")
    print("-" * 88)
    print("per-case verdicts:")
    for entry in benchmark_report["cases"]:
        print(f"  {entry['case_id']:8s} {entry['verdict']:8s} {entry['explanation']}")
    print("=" * 88)
