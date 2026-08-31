from __future__ import annotations

import csv
import io

from app.reasoning.contracts import Evidence
from app.reasoning.fact_ledger import build_fact_ledger, enforce_fact_citations, export_fact_ledger_csv, export_fact_ledger_json


def _evidence() -> Evidence:
    return Evidence(
        id="ev_0", source_tool="compare_periods", evidence_type="CALCULATED_RESULT", metric="revenue",
        params={"value_column": "revenue", "filters": [{"column": "region", "op": "==", "value": "North"}]},
        result_summary={"current_period": {"value": 120.0, "n": 12}, "previous_period": {"value": 100.0, "n": 10}, "pct_change": 20.0},
        sample_size=22, tool_call_ref="tool_call[0]",
    )


def test_fact_ledger_records_every_numeric_leaf_with_stable_provenance() -> None:
    first = build_fact_ledger([_evidence()])
    second = build_fact_ledger([_evidence()])
    assert len(first.facts) == 5
    assert [fact.id for fact in first.facts] == [fact.id for fact in second.facts]
    assert all(len(fact.source_hash) == 64 for fact in first.facts)
    pct = next(fact for fact in first.facts if fact.result_path.endswith("pct_change"))
    assert pct.value == 20.0 and pct.unit == "percent"
    assert pct.filters[0]["column"] == "region"
    assert pct.row_count == 22


def test_fact_ledger_exports_json_and_csv() -> None:
    ledger = build_fact_ledger([_evidence()])
    assert '"source_hash"' in export_fact_ledger_json(ledger)
    rows = list(csv.DictReader(io.StringIO(export_fact_ledger_csv(ledger))))
    assert len(rows) == len(ledger.facts)
    assert rows[0]["id"].startswith("fact_")


def test_numeric_narrative_gets_fact_id_not_evidence_id() -> None:
    ledger = build_fact_ledger([_evidence()])
    rendered = enforce_fact_citations("Revenue increased 20.0 percent.", ledger)
    assert "[fact_" in rendered
    assert "ev_0" not in rendered and "tool_call" not in rendered
