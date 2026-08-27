"""One-off, manually-invoked real-LLM spot-check of the v2 reliability mission's
new capabilities (Confound Engine 2.0, Contradiction Engine 2.0). A single case,
per the standing quota-conservation rule -- the scripted suites already prove the
mechanisms fire correctly against a MockProvider; this checks only whether a real
model's own (unscripted) tool calls still exercise them the same way.

Run directly (never via pytest -- this makes a real network call):

    cd backend
    venv/Scripts/python.exe -m tests.benchmark.real_llm.run_v2_spotcheck
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.benchmark.hard_fixtures import HARD_FIXTURES
from tests.benchmark.real_llm.hard_runner import run_real_hard_case

_BENCH_DIR = Path(__file__).resolve().parents[1]
_FIXTURE_PATH = _BENCH_DIR / "hard_realworld_cases.json"
_REPORT_PATH = Path(__file__).resolve().parents[4] / ".agent" / "v2_real_llm_spotcheck.md"

_CASE_ID = "hard_confound_01a"  # honest confound case -- exercises Confound Engine 2.0 for real


def main() -> None:
    cases_by_id = {c["case_id"]: c for c in json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]}
    case = cases_by_id[_CASE_ID]
    record = HARD_FIXTURES[case["fixture"]]()

    print(f"Running {_CASE_ID}...", flush=True)
    result = run_real_hard_case(case, record)

    lines = ["# v2 mission real-LLM spot-check", ""]
    if result.provider_failure:
        lines.append(f"## {_CASE_ID}: PROVIDER_ERROR")
        lines.append(f"- {result.explanation}")
        print(f"  -> PROVIDER_ERROR: {result.explanation}")
    else:
        lines.append(f"## {_CASE_ID}: {result.verdict}")
        lines.append(f"- explanation: {result.explanation}")
        if result.result is not None:
            lines.append(f"- final_answer_text: {result.result.final_answer_text!r}")
            lines.append(f"- limitations: {[(l.category, l.severity, l.text[:150]) for l in result.result.limitations]}")
            if result.result.analytical_audit is not None:
                lines.append(f"- conclusion_status: {result.result.analytical_audit.conclusion_status}")
                lines.append(f"- confounds: {result.result.analytical_audit.confounds}")
        print(f"  -> {result.verdict}: {result.explanation}")

    _REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {_REPORT_PATH}")


if __name__ == "__main__":
    main()
