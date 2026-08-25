"""Phase 3B.9: structural validation of the reasoning-benchmark fixture seed set.

Deliberately NOT a scoring test -- per the explicit instruction not to claim a
benchmark percentage yet, this only proves the fixture file is well-formed and
internally consistent (e.g. its category names are real, its stated dataset facts
match the actual demo file) so a future scoring pass has a trustworthy foundation to
run against.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.reasoning.categories import ToolCategory

_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "benchmark" / "reasoning_questions.json"
_REQUIRED_CASE_KEYS = {
    "id",
    "category",
    "question_en",
    "dataset",
    "expected_capability",
    "expected_tool_category",
    "expected_constraints",
    "expected_finding_classifications",
    "expected_limitations",
    "expected_causal_behavior",
}
_VALID_CLASSIFICATIONS = {"FACT", "CALCULATED_RESULT", "STATISTICAL_RESULT", "HYPOTHESIS", "ASSUMPTION", "UNKNOWN"}


@pytest.fixture(scope="module")
def fixture_data() -> dict:
    with open(_FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_fixture_file_is_valid_json_with_cases(fixture_data):
    assert isinstance(fixture_data["cases"], list)
    assert len(fixture_data["cases"]) >= 10


def test_every_case_has_all_required_fields(fixture_data):
    for case in fixture_data["cases"]:
        missing = _REQUIRED_CASE_KEYS - set(case.keys())
        assert not missing, f"case {case.get('id')} missing fields: {missing}"


def test_case_ids_are_unique(fixture_data):
    ids = [c["id"] for c in fixture_data["cases"]]
    assert len(ids) == len(set(ids))


def test_expected_tool_categories_are_real_categories(fixture_data):
    valid = {c.value for c in ToolCategory}
    for case in fixture_data["cases"]:
        for cat in case["expected_tool_category"]:
            assert cat in valid, f"case {case['id']} references unknown category {cat!r}"


def test_expected_finding_classifications_are_real(fixture_data):
    for case in fixture_data["cases"]:
        for classification in case["expected_finding_classifications"]:
            assert classification in _VALID_CLASSIFICATIONS


def test_at_least_one_case_covers_each_required_scenario_category(fixture_data):
    categories = {c["category"] for c in fixture_data["cases"]}
    required = {
        "descriptive",
        "missing_column",
        "coverage_mismatch",
        "false_scale_claim",
        "statistical_significance",
        "forecasting",
        "sql",
        "causal_trap",
        "unsupported_recommendation",
        "diagnostic_hypothesis",
        "tool_category_filtering",
        "unavailable_capability",
    }
    missing = required - categories
    assert not missing, f"benchmark fixture missing scenario coverage for: {missing}"


def test_primary_dataset_facts_match_the_real_demo_file(fixture_data):
    path = Path(__file__).resolve().parents[3] / fixture_data["primary_dataset"]
    df = pd.read_excel(path)
    facts = fixture_data["primary_dataset_facts"]
    assert len(df) == facts["rows"]
    assert list(df.columns) == facts["column_names"]


def test_no_benchmark_score_is_claimed_yet(fixture_data):
    """Guards the explicit instruction: this file is a seed set, not a scored result."""
    assert "score" not in fixture_data
    assert "pass_rate" not in fixture_data
    for case in fixture_data["cases"]:
        assert "actual_result" not in case
        assert "benchmark_result" not in case
