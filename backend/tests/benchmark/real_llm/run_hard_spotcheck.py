"""One-off, manually-invoked real-LLM spot-check of the hard real-world benchmark.

NOT a pytest file (no test collection here) -- run directly:

    cd backend
    venv/Scripts/python.exe -m tests.benchmark.real_llm.run_hard_spotcheck

Picks a small, deliberately diverse sample of cases (not the full 102 -- see this
project's standing rule against burning shared quota to re-validate structural
plumbing a MockProvider already proves works) spanning the hardest categories: a
Simpson's-paradox confound, a multicollinearity diagnostic, two recommendation-
refusal cases, the positive-control causal-permission case, and a scale-aware
statistical-significance case. Runs strictly sequentially, stops immediately (does
not keep retrying other cases) if a provider error looks like quota exhaustion rather
than a transient blip, and writes a plain-language report to
.agent/hard_realworld_real_llm_spotcheck.md separating REAL RESPONSES OBTAINED from
PROVIDER_ERROR / UNMEASURED cases -- a provider error is never written up as a model
failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from app.datasets.storage import DatasetRecord
from tests.benchmark.hard_fixtures import HARD_FIXTURES
from tests.benchmark.real_llm.hard_runner import run_real_hard_case

_BENCH_DIR = Path(__file__).resolve().parents[1]
_FIXTURE_PATH = _BENCH_DIR / "hard_realworld_cases.json"
_DEMO_XLSX_PATH = Path(__file__).resolve().parents[4] / "data" / "demo" / "sales_data.xlsx"
_REPORT_PATH = Path(__file__).resolve().parents[4] / ".agent" / "hard_realworld_real_llm_spotcheck.md"

_SAMPLE_CASE_IDS = [
    "hard_confound_01a",  # Simpson's paradox (honest) -- confound recognition
    "hard_stat_02",       # multicollinearity diagnostic nuance
    "hard_prim_05",       # discontinue-product recommendation refusal (thin evidence)
    "hard_prim_04",       # marketing-budget recommendation refusal (no data at all)
    "hard_prim_10",       # positive control: causal language SHOULD be permitted
    "hard_scale_08",      # large-N statistical vs. practical significance
]

_QUOTA_ERROR_MARKERS = ("rate limit", "quota", "429", "tokens per day", "tpd")


def _looks_like_quota_exhaustion(err: str) -> bool:
    lowered = err.lower()
    return any(marker in lowered for marker in _QUOTA_ERROR_MARKERS)


def _build_records() -> dict[str, DatasetRecord]:
    primary_df = pd.read_excel(_DEMO_XLSX_PATH)
    records: dict[str, DatasetRecord] = {
        "primary": DatasetRecord(
            id="hard-spotcheck-primary", original_filename="sales_data.xlsx", extension=".xlsx",
            uploaded_at=pd.Timestamp.utcnow(), df=primary_df, stored_path="unused",
        )
    }
    for name, builder in HARD_FIXTURES.items():
        records[name] = builder()
    return records


def main() -> int:
    with open(_FIXTURE_PATH, encoding="utf-8") as f:
        cases_by_id = {c["case_id"]: c for c in json.load(f)["cases"]}

    records = _build_records()
    lines = [
        "# Hard Real-World Benchmark — Real-LLM Spot-Check",
        "",
        "Small, representative sample run against the real configured provider "
        "(not the full 102-case suite — see run_hard_spotcheck.py's own docstring "
        "for why). Scored with the identical 15-dimension rubric the scripted 97.1% "
        "result was graded by.",
        "",
        "| case_id | verdict | provider_failure | explanation |",
        "|---|---|---|---|",
    ]
    real_responses = 0
    provider_failures = 0
    stopped_early = False

    for case_id in _SAMPLE_CASE_IDS:
        case = dict(cases_by_id[case_id])
        case.pop("script", None)  # real model produces its own tool calls/text
        record = records[case["fixture"]]

        print(f"Running {case_id} against the real provider...", file=sys.stderr)
        result = run_real_hard_case(case, record)

        if result.provider_failure:
            provider_failures += 1
            err_text = result.explanation
            lines.append(f"| {case_id} | UNMEASURED | **yes** | `{err_text}` |")
            print(f"  -> PROVIDER_ERROR: {err_text}", file=sys.stderr)
            if _looks_like_quota_exhaustion(err_text):
                print("Quota exhaustion detected -- stopping the spot-check now, not retrying further cases.", file=sys.stderr)
                stopped_early = True
                break
            continue

        real_responses += 1
        lines.append(f"| {case_id} | {result.verdict} | no | {result.explanation} |")
        print(f"  -> {result.verdict}: {result.explanation}", file=sys.stderr)

    lines += [
        "",
        f"**Real responses obtained**: {real_responses}/{len(_SAMPLE_CASE_IDS)}",
        f"**Provider failures (UNMEASURED, not model failures)**: {provider_failures}/{len(_SAMPLE_CASE_IDS)}",
    ]
    if stopped_early:
        lines.append(
            f"**Stopped early**: quota/rate-limit exhaustion detected after "
            f"{real_responses + provider_failures} of {len(_SAMPLE_CASE_IDS)} cases -- "
            "per this project's standing rule, did not keep retrying against the API."
        )

    _REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {_REPORT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
