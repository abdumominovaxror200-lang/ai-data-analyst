"""Self-test for the Phase 3C Part E structural scoring framework
(tests/benchmark/scoring.py) -- proves the scorer itself is trustworthy (catches a
wrong category, rewards a correct honest answer) before any benchmark case authored
against it can be trusted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.datasets.storage import DatasetRecord
from tests.benchmark.scoring import build_mock_provider_from_script, run_case, summarize


@pytest.fixture
def record() -> DatasetRecord:
    rng = np.random.default_rng(1)
    n = 100
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n),
            "revenue": rng.normal(500, 50, n),
            "region": rng.choice(["North", "South"], n),
        }
    )
    return DatasetRecord(
        id="x", original_filename="sales.csv", extension=".csv", uploaded_at=pd.Timestamp.utcnow(), df=df, stored_path="x"
    )


def _stats_case(**overrides) -> dict:
    case = {
        "case_id": "smoke1",
        "category": "statistics",
        "user_question": "Is revenue significantly different between regions?",
        "expected_tool_category": ["STATISTICS"],
        "required_constraints": [],
        "expected_classifications": ["STATISTICAL_RESULT"],
        "expected_limitations": [],
        "expected_causal_behavior": "not_applicable",
        "script": {
            "parsed_question": {
                "intent": "comparative",
                "requested_metrics": ["revenue"],
                "requested_dimensions": ["region"],
                "requested_time_range": None,
                "requested_population": None,
                "explicit_constraints": [],
                "required_confidence": None,
                "language": "en",
                "claims": [],
            },
            "plan": {
                "objective": "test",
                "capability_categories": ["STATISTICS"],
                "steps": [],
                "tools_required": ["t_test"],
                "expected_outputs": [],
                "validation_steps": [],
                "stopping_conditions": [],
                "hypotheses": [],
            },
            "tool_calls": [
                {"tool": "t_test", "arguments": {"column": "revenue", "group_column": "region", "group_a": "North", "group_b": "South"}}
            ],
            "final_answer_text": "No significant difference found.",
            "recommendation": None,
        },
    }
    case.update(overrides)
    return case


def test_correct_case_scores_pass(record):
    result = run_case(_stats_case(), record)
    assert result.verdict == "PASS"
    assert all(c.passed for c in result.checks if c.passed is not None)


def test_wrong_capability_category_scores_below_pass(record):
    bad = _stats_case()
    bad["script"]["plan"]["capability_categories"] = ["FORECASTING"]
    bad["script"]["plan"]["tools_required"] = ["forecast"]
    bad["script"]["tool_calls"] = [{"tool": "forecast", "arguments": {"date_column": "date", "value_column": "revenue", "periods": 3}}]
    result = run_case(bad, record)
    assert result.verdict in ("PARTIAL", "FAIL")
    failing = [c for c in result.checks if c.passed is False]
    assert any("capability" in c.name for c in failing)


def test_unhedged_causal_language_fails_the_causal_check(record):
    case = _stats_case(expected_causal_behavior="must_hedge_unless_causal_hypothesis_supported")
    case["script"]["final_answer_text"] = "Region caused the revenue drop."
    result = run_case(case, record)
    # the orchestrator's own causation guard hedges this before scoring sees it --
    # proving the guard, not just the scorer, actually fires end-to-end.
    causal_check = next(c for c in result.checks if c.name == "correct causal language")
    assert causal_check.passed is True
    assert "caused" not in result.result.final_answer_text.lower()


def test_missing_column_early_stop_case_scores_pass(record):
    case = {
        "case_id": "missing_col",
        "category": "missing_column",
        "user_question": "What is conversion_rate?",
        "expected_tool_category": [],
        "required_constraints": [],
        "expected_classifications": ["UNKNOWN"],
        "expected_limitations": [{"category": "missing_data", "severity": "blocks_conclusion"}],
        "expected_causal_behavior": "not_applicable",
        "script": {
            "parsed_question": {
                "intent": "descriptive",
                "requested_metrics": ["conversion_rate"],
                "requested_dimensions": [],
                "requested_time_range": None,
                "requested_population": None,
                "explicit_constraints": [],
                "required_confidence": None,
                "language": "en",
                "claims": [],
            },
            "final_answer_text": "The conversion_rate column does not exist.",
            "recommendation": None,
        },
    }
    result = run_case(case, record)
    assert result.verdict == "PASS"


def test_summarize_produces_category_breakdown(record):
    cases = [_stats_case(case_id="a"), _stats_case(case_id="b", category="statistics")]
    results = [run_case(c, record) for c in cases]
    cases_by_id = {c["case_id"]: c for c in cases}
    report = summarize(results, cases_by_id)
    assert report["total_tasks"] == 2
    assert report["passed"] == 2
    assert report["overall_score_pct"] == 100.0
    assert report["category_scores_pct"]["statistics"] == 100.0


def test_build_mock_provider_from_script_produces_expected_call_count():
    provider = build_mock_provider_from_script(_stats_case()["script"])
    assert len(provider._script) == 5  # parse, plan, 1 tool call, stop, synthesize
