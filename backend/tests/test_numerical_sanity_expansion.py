from __future__ import annotations

import pandas as pd
import pytest

from app.datasets.metric_registry import MetricDefinition, MetricRegistry
from app.reasoning.contracts import Evidence
from app.reasoning.numerical_sanity import check_numerical_sanity
from app.tools.derived_ratio import derived_ratio
from app.tools.errors import ToolExecutionError


def _ev(summary: dict, tool: str = "synthetic") -> Evidence:
    return Evidence(id="ev_0", source_tool=tool, evidence_type="CALCULATED_RESULT", metric="metric", result_summary=summary, tool_call_ref="tool_call[0]")


def test_negative_counts_and_durations_block_conclusion() -> None:
    limitations = check_numerical_sanity([_ev({"customer_count": -2, "duration_days": -1})])
    assert any(item.severity == "blocks_conclusion" and "negative" in item.text for item in limitations)


def test_nested_share_over_one_hundred_blocks_conclusion() -> None:
    limitations = check_numerical_sanity([_ev({"groups": [{"share_pct": 101.2}]})])
    assert any(item.severity == "blocks_conclusion" and "0-100" in item.text for item in limitations)


def test_signed_percentage_change_may_be_negative() -> None:
    limitations = check_numerical_sanity([_ev({"pct_change": -20.0})])
    assert not any("0-100" in item.text for item in limitations)


def test_group_breakdown_must_tie_to_reported_total() -> None:
    limitations = check_numerical_sanity([_ev({"total": 100.0, "groups": [{"value": 40.0}, {"value": 50.0}]}, "group_and_aggregate")])
    assert any(item.severity == "blocks_conclusion" and "tie out" in item.text for item in limitations)
    assert not any("tie out" in item.text for item in check_numerical_sanity([_ev({"total": 90.0, "groups": [{"value": 40.0}, {"value": 50.0}]})]))


def test_extreme_magnitude_against_known_total_blocks() -> None:
    limitations = check_numerical_sanity([_ev({"total": 10.0, "reported_value": 10000.0})])
    assert any("1000x" in item.text and item.severity == "blocks_conclusion" for item in limitations)


def test_derived_ratio_uses_explicit_registry_denominator() -> None:
    frame = pd.DataFrame({"converted": [1, 0, 1], "eligible": [1, 1, 1]})
    registry = MetricRegistry()
    registry.register(MetricDefinition(name="conversion_rate", kind="rate", numerator_column="converted", denominator_column="eligible", numerator_aggregation="sum", denominator_aggregation="sum", unit="proportion", reason="Reviewed definition."))
    result = derived_ratio(frame, registry, metric_name="conversion_rate")
    assert result["numerator"] == 2
    assert result["denominator"] == 3
    assert result["ratio_pct"] == pytest.approx(66.666667)


def test_derived_ratio_refuses_unresolved_or_zero_denominator() -> None:
    frame = pd.DataFrame({"conversion_rate": [0.1], "num": [1], "den": [0]})
    with pytest.raises(ToolExecutionError, match="explicit numerator and denominator"):
        derived_ratio(frame, MetricRegistry.from_dataframe(frame), metric_name="conversion_rate")
    with pytest.raises(ToolExecutionError, match="zero"):
        derived_ratio(frame, MetricRegistry(), numerator_column="num", denominator_column="den")


def test_percentage_ratio_over_one_hundred_is_flagged() -> None:
    evidence = _ev({"ratio_pct": 125.0, "as_percentage": True}, "derived_ratio")
    assert any(item.severity == "blocks_conclusion" for item in check_numerical_sanity([evidence]))
