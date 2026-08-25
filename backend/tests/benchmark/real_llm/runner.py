"""Phase 4 P1: real-LLM benchmark harness foundation.

Built once, centrally, before the parallel real-LLM case-authoring wave
(REAL-LLM-BENCHMARK-ENGINEER, ADVERSARIAL-LLM-QA-ENGINEER) starts -- same
"foundation lands first" pattern as `tests/benchmark/scoring.py` in Phase 3C, so
both agents author cases against one shared, stable contract without seeing each
other's worktree.

**This module makes REAL network calls to the configured LLM provider (Groq) when
invoked.** It must NEVER run as part of an ordinary `pytest -q` -- see
`real_llm_enabled()`/`skip_unless_real_llm` below. Every test file under
`tests/benchmark/real_llm/` MUST apply `pytestmark = skip_unless_real_llm` (or the
per-test decorator) at the top of the file, so the whole live suite is skipped by
default and only runs when a human deliberately opts in.

Reuses `tests/benchmark/scoring.py`'s structural checks and case-schema conventions
exactly (`_structural_checks`, `_verdict`, `_explain`, `CheckResult`, `CaseResult`) --
the SAME 10 structural properties that grade the scripted benchmarks also grade real
runs, so results are directly comparable. Real-LLM cases use the identical case dict
shape as `professional_benchmark.json`/`adversarial_cases.json` EXCEPT they must NOT
include a `"script"` key (there is no MockProvider here -- the real model generates
its own tool calls and text).
"""

from __future__ import annotations

import os
import time

import pytest

from app.agent.providers import LLMProviderError, build_provider_from_settings
from app.datasets.storage import DatasetRecord
from app.reasoning.orchestrator import ReasoningOrchestrator
from tests.benchmark.scoring import CaseResult, CheckResult, _explain, _structural_checks, _verdict

REAL_LLM_ENV_VAR = "RUN_REAL_LLM_BENCHMARK"


def real_llm_enabled() -> bool:
    """Explicit opt-in required -- the mere presence of a configured LLM_API_KEY is
    NOT enough on its own to run live tests; a human must also set this env var."""
    return os.environ.get(REAL_LLM_ENV_VAR) == "1"


skip_unless_real_llm = pytest.mark.skipif(
    not real_llm_enabled(),
    reason=f"Set {REAL_LLM_ENV_VAR}=1 (and a real LLM_API_KEY) to run live-LLM benchmark cases. "
    "Costs real API calls/tokens and takes real wall-clock time -- never runs in an ordinary pytest pass.",
)


def build_real_provider():
    """Reuses the project's existing provider-construction path verbatim -- the same
    function `routes_chat.py`/`routes_reasoning.py` call in production. Raises
    ValueError if no key is configured (caught by callers, surfaced as a clear skip
    reason rather than an opaque failure)."""
    return build_provider_from_settings()


def run_real_case(case: dict, record: DatasetRecord, *, retries: int = 1) -> CaseResult:
    """Drives the REAL `ReasoningOrchestrator` (real Groq calls, real tool
    execution) for one case, then scores the result with the identical structural
    checks the scripted benchmarks use. `case` must NOT contain a `"script"` key.

    On a transient provider error (rate limit, timeout), retries once after a short
    backoff before giving up -- live APIs are flakier than a MockProvider, and one
    real network hiccup should not fail an entire case outright. A persistent
    failure is reported as a FAIL verdict with the error in the explanation, never
    raised uncaught (a live-benchmark run must finish and report on every case, not
    crash partway through)."""
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            provider = build_real_provider()
            orchestrator = ReasoningOrchestrator(provider)
            result = orchestrator.analyze(record, case["user_question"])
            checks = _structural_checks(case, result)
            return CaseResult(
                case_id=case["case_id"],
                verdict=_verdict(checks),
                checks=checks,
                explanation=_explain(checks),
                result=result,
            )
        except (LLMProviderError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))
                continue

    check = CheckResult(name="live LLM call succeeded", passed=False, detail=str(last_error))
    return CaseResult(case_id=case["case_id"], verdict="FAIL", checks=[check], explanation=f"Provider error: {last_error}", result=None)


def validate_real_case_schema(case: dict) -> None:
    """Same required keys as the scripted-benchmark schema, minus `script` (which
    must be ABSENT, not just optional, for a real-LLM case)."""
    required = {
        "case_id",
        "category",
        "user_question",
        "expected_tool_category",
        "required_constraints",
        "expected_classifications",
        "expected_limitations",
        "expected_causal_behavior",
    }
    missing = required - set(case.keys())
    if missing:
        raise ValueError(f"case {case.get('case_id')} missing required fields: {missing}")
    if "script" in case:
        raise ValueError(f"case {case['case_id']} must not define 'script' -- this is a real-LLM case, there is no MockProvider.")
