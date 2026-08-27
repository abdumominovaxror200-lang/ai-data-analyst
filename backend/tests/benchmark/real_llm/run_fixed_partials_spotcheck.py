"""One-off, manually-invoked real-LLM spot-check of the 3 hard-benchmark cases that
were PARTIAL before this session's fixes (final stress-test mission, Phase 8).

The scripted MockProvider suite proves the new deterministic checks
(group_size_imbalance, unusual_baseline_window, conclusion_guard's caveat) fire
correctly against a SCRIPTED adversarial answer. This checks the thing a scripted
script cannot: does a REAL model's own (unscripted) tool calls and phrasing still get
caught by the same deterministic backstops? That is exactly the highest-value,
deterministic-tests-cannot-validate case Phase 8 asks to prioritize.

Run directly (never via pytest -- this makes real network calls):

    cd backend
    venv/Scripts/python.exe -m tests.benchmark.real_llm.run_fixed_partials_spotcheck

Mirrors run_hard_spotcheck.py's exact discipline: sequential, one call at a time,
stops immediately on anything that looks like quota exhaustion, and separates
PROVIDER_ERROR from a genuine model FAIL.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.datasets.storage import DatasetRecord
from tests.benchmark.hard_fixtures import HARD_FIXTURES
from tests.benchmark.real_llm.hard_runner import run_real_hard_case

_BENCH_DIR = Path(__file__).resolve().parents[1]
_FIXTURE_PATH = _BENCH_DIR / "hard_realworld_cases.json"
_REPORT_PATH = Path(__file__).resolve().parents[4] / ".agent" / "hard_realworld_fixed_partials_spotcheck.md"

_SAMPLE_CASE_IDS = ["hard_confound_01b", "hard_ab_01b", "hard_price_01b"]

_QUOTA_ERROR_MARKERS = ("rate limit", "quota", "429", "tokens per day", "tpd")


def _looks_like_quota_exhaustion(err: str) -> bool:
    lowered = err.lower()
    return any(marker in lowered for marker in _QUOTA_ERROR_MARKERS)


def main() -> None:
    cases_by_id = {c["case_id"]: c for c in json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]}

    lines = [
        "# Real-LLM spot-check: the 3 fixed PARTIAL cases",
        "",
        "Checks whether a REAL (unscripted) model's own tool calls and phrasing still "
        "get caught by this session's new deterministic checks "
        "(group_size_imbalance, unusual_baseline_window, conclusion_guard).",
        "",
    ]

    for case_id in _SAMPLE_CASE_IDS:
        case = cases_by_id[case_id]
        record: DatasetRecord = HARD_FIXTURES[case["fixture"]]()
        print(f"Running {case_id}...", flush=True)
        result = run_real_hard_case(case, record)

        if result.provider_failure:
            lines.append(f"## {case_id}: PROVIDER_ERROR")
            lines.append(f"- {result.explanation}")
            lines.append("")
            print(f"  -> PROVIDER_ERROR: {result.explanation}")
            if _looks_like_quota_exhaustion(result.explanation):
                lines.append("**Stopping: this looks like quota exhaustion, not a transient blip.**")
                print("Quota exhaustion detected -- stopping sequential run.")
                break
            continue

        lines.append(f"## {case_id}: {result.verdict}")
        lines.append(f"- explanation: {result.explanation}")
        if result.result is not None:
            lines.append(f"- final_answer_text: {result.result.final_answer_text!r}")
            lines.append(f"- limitations: {[(l.category, l.severity, l.text[:120]) for l in result.result.limitations]}")
            if result.result.recommendation is not None:
                lines.append(f"- recommendation.confidence: {result.result.recommendation.confidence!r}")
        lines.append("")
        print(f"  -> {result.verdict}: {result.explanation}")

    _REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {_REPORT_PATH}")


if __name__ == "__main__":
    main()
