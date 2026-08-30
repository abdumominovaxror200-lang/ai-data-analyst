"""Independent tests for deterministic RCA tool-obligation classification."""
from __future__ import annotations

from app.reasoning.contracts import AnalysisPlan, AnalyticalQuestion, Evidence
from app.reasoning.coverage import assess_coverage


def _question(text="Explain the metric change", **updates):
    return AnalyticalQuestion(original_question=text, intent="diagnostic", **updates)


def _plan(*tools, objective="Explain the metric change", categories=None):
    return AnalysisPlan(objective=objective, capability_categories=categories or ["GENERAL_ANALYSIS"],
                        tools_required=list(tools))


def _evidence(tool, index=0, *, summary=None, evidence_type="CALCULATED_RESULT"):
    return Evidence(id=f"ev_{index}", source_tool=tool, evidence_type=evidence_type,
                    result_summary=summary or {"value": 1}, tool_call_ref=f"tool_call[{index}]")


def test_irrelevant_presentation_tools_never_block_or_recover():
    result = assess_coverage(
        _question(), _plan("generate_report", "generate_chart"), [], date_columns=[],
        executed_tools=[], recovery_finished=True)
    assert result.complete
    assert result.recovery_targets == [] and result.unresolved_tools == []
    assert {item.obligation for item in result.tools} == {"optional_supporting"}


def test_duplicate_analysis_is_optional_for_normal_analysis():
    result = assess_coverage(
        _question(), _plan("duplicate_analysis", categories=["DATA_QUALITY"]), [],
        date_columns=[], executed_tools=[], recovery_finished=True)
    assert result.complete
    assert result.tools[0].obligation == "conditional_data_quality"
    assert result.tools[0].unavailable is False


def test_duplicate_analysis_becomes_required_for_explicit_duplicate_risk():
    result = assess_coverage(
        _question("Check whether duplicate records inflated the comparison"),
        _plan(objective="Validate duplicate-row risk", categories=["DATA_QUALITY"]),
        [], date_columns=[], executed_tools=[], recovery_finished=False)
    assert result.complete is False
    assert result.recovery_targets == ["duplicate_analysis"]
    assert result.tools[0].obligation == "conditional_data_quality"


def test_duplicate_signal_in_evidence_activates_duplicate_obligation():
    profile = _evidence("profile_dataset", summary={"duplicate_rows": 4})
    result = assess_coverage(
        _question(), _plan(categories=["DATA_QUALITY"]), [profile],
        date_columns=[], executed_tools=["profile_dataset"], recovery_finished=False)
    assert result.recovery_targets == ["duplicate_analysis"]


def test_genuinely_missing_statistical_requirement_blocks():
    result = assess_coverage(
        _question("Is the difference statistically significant?"),
        _plan("t_test", objective="Test statistical significance", categories=["STATISTICS"]),
        [], date_columns=[], executed_tools=["t_test"], recovery_finished=True)
    assert not result.complete
    assert result.unresolved_tools == ["t_test"]
    assert "statistical" in result.unresolved_requirements


def test_genuinely_missing_segment_requirement_blocks():
    result = assess_coverage(
        _question(requested_dimensions=["channel"]), _plan("group_and_aggregate"), [],
        date_columns=[], executed_tools=[], recovery_finished=True)
    assert not result.complete
    assert "segment(channel)" in result.unresolved_requirements


def test_genuinely_missing_outlier_requirement_blocks():
    result = assess_coverage(
        _question("Test whether the result is robust to extreme observations"),
        _plan("outlier_analysis_multivariate", objective="Evaluate outlier robustness", categories=["REGRESSION"]),
        [], date_columns=[], executed_tools=["outlier_analysis_multivariate"], recovery_finished=True)
    assert not result.complete
    assert result.unresolved_tools == ["outlier_analysis_multivariate"]
    assert "outlier" in result.unresolved_requirements


def test_recovery_contains_only_required_analytical_gaps():
    result = assess_coverage(
        _question("Test statistical uncertainty"),
        _plan("t_test", "generate_report", "duplicate_analysis", objective="Test statistical uncertainty",
              categories=["STATISTICS", "GENERAL_ANALYSIS", "DATA_QUALITY"]),
        [], date_columns=[], executed_tools=[], recovery_finished=False)
    assert result.recovery_targets == ["t_test"]
