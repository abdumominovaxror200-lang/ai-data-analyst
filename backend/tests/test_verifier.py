"""Unit tests for app.reasoning.verifier -- the deterministic evidence-validation,
cross-check, and finding-classification module. No dedicated unit test file existed
for this module before (it was previously exercised only indirectly through the
end-to-end benchmark suites) -- added alongside the `_investigation_cross_check` fix
below so this module's real, non-obvious behavior has fast, precise, direct coverage
instead of only being visible through a 100+-case benchmark run.

`_investigation_cross_check` regression: a genuine architectural gap found via the
hard real-world benchmark (see .agent/hard_realworld_benchmark.md finding #1) --
`_cross_check` only corroborates two tools that report the exact same flat numeric
value for a metric, so a real multi-step root-cause investigation (compare a period,
break down by category, check for anomalies -- none of which produce a directly
comparable scalar) was never marked `cross_checked=True` despite being a genuine,
evidence-grounded verification sequence. Fixed by recognizing "an independent
verification tool examined the same metric and found nothing wrong" as its own,
additive corroboration signal.
"""

from __future__ import annotations

from app.reasoning.contracts import Evidence
from app.reasoning.verifier import build_findings


def _ev(id_, tool, metric, result_summary, evidence_type="CALCULATED_RESULT", sample_size=None):
    return Evidence(
        id=id_, source_tool=tool, evidence_type=evidence_type, metric=metric,
        result_summary=result_summary, sample_size=sample_size, tool_call_ref=f"tool_call[{id_}]",
    )


# --- classification ------------------------------------------------------------------


def test_evidence_type_maps_directly_to_finding_classification():
    evidence = [_ev("ev_0", "describe_data", "revenue", {"columns": {}}, evidence_type="FACT")]
    findings, _ = build_findings(evidence)
    assert findings[0].classification == "FACT"


def test_every_finding_traces_back_to_its_evidence_id():
    evidence = [_ev("ev_0", "group_and_aggregate", "revenue", {"groups": []})]
    findings, _ = build_findings(evidence)
    assert findings[0].supporting_evidence == ["ev_0"]


# --- sample-size limitation ------------------------------------------------------------


def test_low_sample_size_statistical_result_gets_a_limitation():
    evidence = [_ev("ev_0", "t_test", "revenue", {"p_value": 0.2}, evidence_type="STATISTICAL_RESULT", sample_size=4)]
    _, limitations = build_findings(evidence)
    assert any(l.category == "sample_size" for l in limitations)


def test_adequate_sample_size_gets_no_sample_size_limitation():
    evidence = [_ev("ev_0", "t_test", "revenue", {"p_value": 0.2}, evidence_type="STATISTICAL_RESULT", sample_size=500)]
    _, limitations = build_findings(evidence)
    assert not any(l.category == "sample_size" for l in limitations)


# --- _cross_check: literal scalar agreement / disagreement ---------------------------


def test_two_tools_reporting_the_same_metric_value_are_cross_checked():
    evidence = [
        _ev("ev_0", "describe_data", "revenue", {"mean": 100.0}),
        _ev("ev_1", "confidence_interval", "revenue", {"mean": 101.0}),
    ]
    findings, _ = build_findings(evidence)
    assert all(f.cross_checked for f in findings)


def test_two_tools_reporting_materially_different_values_are_flagged_not_corroborated():
    evidence = [
        _ev("ev_0", "describe_data", "revenue", {"mean": 100.0}),
        _ev("ev_1", "confidence_interval", "revenue", {"mean": 500.0}),
    ]
    findings, limitations = build_findings(evidence)
    assert not any(f.cross_checked for f in findings)
    assert any("disagree" in l.text.lower() for l in limitations)


def test_single_tool_alone_is_never_cross_checked():
    evidence = [_ev("ev_0", "describe_data", "revenue", {"mean": 100.0})]
    findings, _ = build_findings(evidence)
    assert not findings[0].cross_checked


# --- _investigation_cross_check: the new, broader corroboration rule -----------------


def test_verification_tool_finding_nothing_wrong_corroborates_a_diagnostic_investigation():
    """The real hard_prim_06 shape: compare_periods + group_and_aggregate +
    detect_anomalies, all on 'revenue', none producing a comparable flat scalar --
    should now be jointly cross_checked because detect_anomalies ran clean."""
    evidence = [
        _ev("ev_0", "compare_periods", "revenue", {"current_value": 900.0, "previous_value": 1000.0, "pct_change": -10.0}),
        _ev("ev_1", "group_and_aggregate", "revenue", {"groups": [{"group": "Electronics", "value": 400.0}], "group_count": 4}),
        _ev("ev_2", "detect_anomalies", "revenue", {"anomaly_count": 0, "anomaly_pct": 0.0, "anomalies": []}),
    ]
    findings, _ = build_findings(evidence)
    assert all(f.cross_checked for f in findings)


def test_verification_tool_finding_a_real_problem_does_not_corroborate():
    """A high anomaly rate is a real, disqualifying signal -- must NOT be treated as
    corroboration. (A single-digit rate, tested separately below, is ordinary noise on
    real skewed data and IS treated as clean -- see _ANOMALY_CLEAN_THRESHOLD_PCT.)"""
    evidence = [
        _ev("ev_0", "compare_periods", "revenue", {"current_value": 900.0, "previous_value": 1000.0}),
        _ev("ev_1", "detect_anomalies", "revenue", {"anomaly_count": 30, "anomaly_pct": 30.0, "anomalies": [{}] * 30}),
    ]
    findings, _ = build_findings(evidence)
    assert not any(f.cross_checked for f in findings)


def test_a_modest_baseline_anomaly_rate_on_skewed_data_still_counts_as_clean():
    """Real IQR-based outlier detection on any moderately right-skewed real-world
    column (revenue, deal size, etc.) routinely flags a nonzero single-digit
    percentage with no genuine data-quality problem behind it -- verified directly
    against this project's own demo dataset (6.81% on a region-filtered revenue
    column). A literal `anomaly_count == 0` bar would almost never be satisfied on
    real data; a percentage-based threshold is required for this signal to be useful
    in practice."""
    evidence = [
        _ev("ev_0", "compare_periods", "revenue", {"current_value": 900.0, "previous_value": 1000.0}),
        _ev("ev_1", "detect_anomalies", "revenue", {"anomaly_count": 7, "anomaly_pct": 6.81, "anomalies": [{}] * 7}),
    ]
    findings, _ = build_findings(evidence)
    assert all(f.cross_checked for f in findings)


def test_verification_tool_alone_with_no_other_analytical_tool_does_not_corroborate():
    """A clean detect_anomalies result with no OTHER tool examining the same metric
    is not an 'investigation' -- corroboration requires at least two distinct angles."""
    evidence = [
        _ev("ev_0", "detect_anomalies", "revenue", {"anomaly_count": 0, "anomaly_pct": 0.0, "anomalies": []}),
    ]
    findings, _ = build_findings(evidence)
    assert not findings[0].cross_checked


def test_duplicate_analysis_and_data_quality_report_also_count_as_verification_tools():
    evidence = [
        _ev("ev_0", "group_and_aggregate", "revenue", {"groups": [], "group_count": 2}),
        _ev("ev_1", "duplicate_analysis", "revenue", {"duplicate_row_count": 0, "duplicate_pct": 0.0}),
    ]
    findings, _ = build_findings(evidence)
    assert all(f.cross_checked for f in findings)

    evidence2 = [
        _ev("ev_0", "group_and_aggregate", "revenue", {"groups": [], "group_count": 2}),
        _ev("ev_1", "data_quality_report", "revenue", {"quality_score": 100, "quality_issues": []}),
    ]
    findings2, _ = build_findings(evidence2)
    assert all(f.cross_checked for f in findings2)


def test_dirty_data_quality_report_does_not_corroborate():
    evidence = [
        _ev("ev_0", "group_and_aggregate", "revenue", {"groups": [], "group_count": 2}),
        _ev("ev_1", "data_quality_report", "revenue", {"quality_score": 60, "quality_issues": ["missing values in revenue"]}),
    ]
    findings, _ = build_findings(evidence)
    assert not any(f.cross_checked for f in findings)


def test_investigation_cross_check_does_not_interfere_with_unrelated_metrics():
    """Evidence for a completely different metric must not be pulled into another
    metric's corroboration."""
    evidence = [
        _ev("ev_0", "compare_periods", "revenue", {"current_value": 900.0}),
        _ev("ev_1", "detect_anomalies", "revenue", {"anomaly_count": 0, "anomalies": []}),
        _ev("ev_2", "describe_data", "profit", {"columns": {}}),
    ]
    findings, _ = build_findings(evidence)
    by_id = {f.supporting_evidence[0]: f for f in findings}
    assert by_id["ev_0"].cross_checked and by_id["ev_1"].cross_checked
    assert not by_id["ev_2"].cross_checked
