from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Iterator

from app.reasoning.contracts import Evidence, Fact, FactLedger

_MAX_FACTS_PER_EVIDENCE = 500


def _numeric_leaves(value: Any, path: str = "result") -> Iterator[tuple[str, int | float]]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not isinstance(value, float) or math.isfinite(value):
            yield path, value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _numeric_leaves(value[key], f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _numeric_leaves(item, f"{path}[{index}]")


def _unit_for(path: str) -> str | None:
    lowered = path.lower()
    if any(token in lowered for token in ("_pct", ".pct", "percent", "percentage")):
        return "percent"
    if "rate" in lowered or "ratio" in lowered:
        return "proportion"
    if any(token in lowered for token in ("_count", ".count", ".n", "row_count")):
        return "count"
    return None


def build_fact_ledger(evidence: list[Evidence]) -> FactLedger:
    facts: list[Fact] = []
    timestamp = datetime.now(timezone.utc)
    for item in evidence:
        canonical = json.dumps({"tool": item.source_tool, "params": item.params, "result": item.result_summary}, sort_keys=True, default=str, separators=(",", ":"))
        source_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        ids: list[str] = []
        for path, value in list(_numeric_leaves(item.result_summary))[:_MAX_FACTS_PER_EVIDENCE]:
            digest = hashlib.sha256(f"{source_hash}:{path}".encode("utf-8")).hexdigest()[:16]
            fact_id = f"fact_{digest}"
            facts.append(Fact(
                id=fact_id, value=value, unit=_unit_for(path), tool=item.source_tool,
                params=item.params, filters=list(item.params.get("filters") or []),
                row_count=item.sample_size, source_hash=source_hash, ts=timestamp, result_path=path,
            ))
            ids.append(fact_id)
        item.fact_ids = ids
    return FactLedger(facts=facts)


def export_fact_ledger_json(ledger: FactLedger) -> str:
    return ledger.model_dump_json(indent=2)


def export_fact_ledger_csv(ledger: FactLedger) -> str:
    output = io.StringIO()
    fields = ["id", "value", "unit", "tool", "params", "filters", "row_count", "source_hash", "ts", "result_path"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for fact in ledger.facts:
        row = fact.model_dump(mode="json")
        row["params"] = json.dumps(row["params"], sort_keys=True)
        row["filters"] = json.dumps(row["filters"], sort_keys=True)
        writer.writerow(row)
    return output.getvalue()


_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def enforce_fact_citations(text: str, ledger: FactLedger) -> str:
    """Attach a stable Fact ID to every narrative number traceable to the ledger."""
    by_value: dict[float, str] = {}
    for fact in ledger.facts:
        by_value.setdefault(float(fact.value), fact.id)
    additions: list[tuple[int, str]] = []
    for match in _NUMBER.finditer(text):
        value = float(match.group())
        fact_id = next((identifier for number, identifier in by_value.items() if math.isclose(number, value, rel_tol=1e-9, abs_tol=1e-9)), None)
        if fact_id and f"[{fact_id}]" not in text[max(0, match.start() - 80):match.end() + 80]:
            additions.append((match.end(), f" [{fact_id}]"))
    for position, citation in reversed(additions):
        text = text[:position] + citation + text[position:]
    return text
