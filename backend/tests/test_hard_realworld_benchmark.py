"""The hard real-world professional-analyst benchmark (user's explicit follow-up
mission, distinct from and much harder than `final_100_cases.json`).

Explicit, repeated instruction from the user for this benchmark: do NOT optimize for a
high pass rate, do NOT make cases artificially easy, do NOT modify a case merely
because the agent fails it, and a failure is valuable evidence -- the goal is to learn
what the system can and cannot do, not to post a number. Consistent with that mandate,
this file does NOT assert any minimum overall pass-rate threshold (unlike
`test_professional_benchmark.py` and `test_final_100_benchmark.py`, both of which score
a much easier suite and do assert a floor). The only hard assertions here are
structural: every case must actually execute and score (no crashes), the fixture must
be well-formed, and the paired adversarial (honest vs. overclaiming) cases must show
the honest answer scoring at least as well as its overclaiming twin -- because if an
overclaiming answer ever scored STRICTLY HIGHER than its honest counterpart, that would
be a real, load-bearing bug in the scoring/grounding machinery, not a benchmark-tuning
question.

See `.agent/hard_realworld_benchmark.md` for the full report: overall score, per-
category and per-dimension breakdowns, the most dangerous/common failure modes, and
root-cause classification for every non-trivial failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.datasets.storage import DatasetRecord
from app.reasoning.categories import ToolCategory
from tests.benchmark.hard_fixtures import HARD_FIXTURES
from tests.benchmark.hard_scoring import DIMENSIONS, run_hard_case, summarize

_BENCH_DIR = Path(__file__).resolve().parent / "benchmark"
_FIXTURE_PATH = _BENCH_DIR / "hard_realworld_cases.json"
_RESULTS_PATH = _BENCH_DIR / "hard_realworld_results.json"
_DEMO_XLSX_PATH = Path(__file__).resolve().parents[2] / "data" / "demo" / "sales_data.xlsx"

_VALID_DIFFICULTY_TIERS = {
    "difficult_solvable", "multi_step", "adversarial", "ambiguous", "insufficient_data", "scalability_data_quality",
}
_MIN_TOTAL_CASES = 100
_MIN_HIDDEN_TRAP_CASES = 30
_MIN_MULTI_STEP_CASES = 20  # cases whose script has >= 3 tool calls
_MIN_SCALABILITY_CASES = 10
_VERDICT_SCORE = {"PASS": 2, "PARTIAL": 1, "FAIL": 0, "UNMEASURED": 1}


# --- fixtures --------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_data() -> dict:
    with open(_FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def primary_df() -> pd.DataFrame:
    return pd.read_excel(_DEMO_XLSX_PATH)


@pytest.fixture(scope="module")
def records(primary_df: pd.DataFrame) -> dict[str, DatasetRecord]:
    recs: dict[str, DatasetRecord] = {
        "primary": DatasetRecord(
            id="hard-benchmark-primary", original_filename="sales_data.xlsx", extension=".xlsx",
            uploaded_at=pd.Timestamp.utcnow(), df=primary_df, stored_path="unused",
        ),
    }
    for name, builder in HARD_FIXTURES.items():
        recs[name] = builder()
    return recs


# --- schema-sanity tests (fast, no orchestrator run) ------------------------------


def test_fixture_file_is_valid_json_with_at_least_100_cases(fixture_data):
    assert isinstance(fixture_data["cases"], list)
    assert len(fixture_data["cases"]) >= _MIN_TOTAL_CASES


def test_case_ids_are_unique(fixture_data):
    ids = [c["case_id"] for c in fixture_data["cases"]]
    assert len(ids) == len(set(ids))


def test_every_case_references_a_known_fixture(fixture_data, records):
    known = set(records.keys())
    for case in fixture_data["cases"]:
        assert case["fixture"] in known, f"case {case['case_id']} references unknown fixture {case['fixture']!r}"


def test_every_case_has_a_valid_difficulty_tier(fixture_data):
    for case in fixture_data["cases"]:
        assert case["difficulty_tier"] in _VALID_DIFFICULTY_TIERS, f"case {case['case_id']} has unknown tier {case['difficulty_tier']!r}"


def test_expected_tool_categories_are_real_categories(fixture_data):
    valid = {c.value for c in ToolCategory}
    for case in fixture_data["cases"]:
        cats = case.get("expected", {}).get("tool_category")
        if cats:
            for cat in cats:
                assert cat in valid, f"case {case['case_id']} references unknown category {cat!r}"


def test_at_least_30_cases_have_hidden_traps(fixture_data):
    n = sum(1 for c in fixture_data["cases"] if c.get("hidden_traps"))
    assert n >= _MIN_HIDDEN_TRAP_CASES, f"only {n} cases have hidden_traps, need >= {_MIN_HIDDEN_TRAP_CASES}"


def test_at_least_20_cases_require_3_or_more_tool_calls(fixture_data):
    n = sum(1 for c in fixture_data["cases"] if c.get("script") and len(c["script"].get("tool_calls") or []) >= 3)
    assert n >= _MIN_MULTI_STEP_CASES, f"only {n} cases have >=3 scripted tool calls, need >= {_MIN_MULTI_STEP_CASES}"


def test_at_least_10_scalability_cases(fixture_data):
    n = sum(1 for c in fixture_data["cases"] if c["difficulty_tier"] == "scalability_data_quality")
    assert n >= _MIN_SCALABILITY_CASES, f"only {n} scalability_data_quality cases, need >= {_MIN_SCALABILITY_CASES}"


def test_adversarial_pairs_are_mutually_referenced(fixture_data):
    by_id = {c["case_id"]: c for c in fixture_data["cases"]}
    for case in fixture_data["cases"]:
        pair_id = case.get("adversarial_pair")
        if pair_id is None:
            continue
        assert pair_id in by_id, f"case {case['case_id']} references unknown pair {pair_id!r}"
        assert by_id[pair_id].get("adversarial_pair") == case["case_id"], f"pair {case['case_id']}<->{pair_id} is not mutual"


def test_no_pre_claimed_score_in_the_fixture_itself(fixture_data):
    assert "score" not in fixture_data
    assert "overall_score_pct" not in fixture_data
    for case in fixture_data["cases"]:
        assert "actual_result" not in case
        assert "verdict" not in case


# --- the real scoring run ----------------------------------------------------------


@pytest.fixture(scope="module")
def benchmark_report(fixture_data, records) -> dict:
    """Runs every case through the real, unmodified hard_scoring.run_hard_case() --
    the real ReasoningOrchestrator for scripted cases, the real premise_validator for
    deterministic-only cases. Computed once (module-scoped): 102 cases, several with
    real regression/clustering/forecasting/ARIMA fits, is genuinely slow work."""
    cases = fixture_data["cases"]
    cases_by_id = {c["case_id"]: c for c in cases}
    results = []
    for case in cases:
        record = records[case["fixture"]]
        results.append(run_hard_case(case, record))
    return summarize(results, cases_by_id), {r.case_id: r for r in results}


def test_full_benchmark_run_covers_every_case(fixture_data, benchmark_report):
    report, _ = benchmark_report
    assert report["total_cases"] == len(fixture_data["cases"])


def test_full_benchmark_run_report_has_the_expected_shape(benchmark_report):
    report, _ = benchmark_report
    for key in ("total_cases", "passed", "partial", "failed", "overall_score_pct", "category_scores_pct", "dimension_scores_pct", "cases"):
        assert key in report
    assert report["passed"] + report["partial"] + report["failed"] + report["unmeasured"] == report["total_cases"]
    assert set(report["dimension_scores_pct"]) == set(DIMENSIONS)


def test_full_benchmark_run_writes_the_real_measured_results_to_disk(benchmark_report):
    """Side effect: the real, measured summarize() output goes to disk exactly as
    produced -- no minimum-score assertion in this file, per the module docstring."""
    report, _ = benchmark_report
    _RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert _RESULTS_PATH.exists()


def test_honest_answers_never_score_strictly_worse_than_their_overclaiming_twin(fixture_data, benchmark_report):
    """The one load-bearing correctness assertion in this file: for every adversarial
    honest/overclaim pair, the honest case's verdict must never be STRICTLY WORSE than
    its overclaiming twin's. If this ever failed, it would mean the scoring/grounding
    machinery rewards confident fabrication over honest hedging -- a real bug, not a
    benchmark-tuning question, and exactly the property Part D of this project's
    original honesty-audit design goal (and this mission's section 3) requires."""
    _, results_by_id = benchmark_report
    failures = []
    # Every pair is authored with the overclaiming half's `category` ending in
    # "_overclaim" (see hard_realworld_cases.json) -- identify pairs from that real
    # metadata rather than guessing from case_id shape.
    for case in fixture_data["cases"]:
        if "overclaim" not in case["category"] and "overclaim" not in case["case_id"]:
            continue
        pair_id = case.get("adversarial_pair")
        if not pair_id:
            continue
        overclaim_result = results_by_id[case["case_id"]]
        honest_result = results_by_id[pair_id]
        if _VERDICT_SCORE[honest_result.verdict] < _VERDICT_SCORE[overclaim_result.verdict]:
            failures.append(
                f"{pair_id} (honest, verdict={honest_result.verdict}) scored WORSE than "
                f"{case['case_id']} (overclaiming, verdict={overclaim_result.verdict})"
            )
    assert not failures, "Honesty-pair inversion(s) found -- a real scoring/grounding bug:\n" + "\n".join(failures)
