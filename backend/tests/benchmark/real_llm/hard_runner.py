"""Real-LLM spot-check harness for the hard real-world benchmark.

Mirrors `real_llm/runner.py`'s discipline exactly (never runs in an ordinary pytest
pass, explicit `RUN_REAL_LLM_BENCHMARK=1` opt-in, one real network call at a time, no
concurrency) but scores against `hard_scoring.py`'s 15-dimension checks instead of
`scoring.py`'s 10 structural ones, so a real run is graded by the identical rubric the
scripted 97.1% result was graded by -- directly comparable, not a different bar.

Per the mission's explicit instruction: a small, REPRESENTATIVE sample only, never the
full 102-case suite against a live provider (quota is shared and limited; a scripted
suite this large exists specifically so real-LLM quota doesn't need to be spent
validating structural plumbing that a MockProvider already proves works). This module
also explicitly separates PROVIDER_ERROR from a genuine model FAIL -- a rate limit or
timeout is never reported as the model having failed the case.
"""

from __future__ import annotations

import time

from app.agent.providers import LLMProviderError
from app.datasets.storage import DatasetRecord
from tests.benchmark.hard_scoring import DimensionCheck, HardCaseResult, _run_with_provider
from tests.benchmark.real_llm.runner import build_real_provider


def run_real_hard_case(case: dict, record: DatasetRecord, *, retries: int = 1) -> HardCaseResult:
    """Drives the REAL `ReasoningOrchestrator` (real provider calls, real tool
    execution) for one hard-benchmark case, scored with the identical
    `hard_scoring.py` dimension checks the scripted suite uses. `case` may contain a
    `"script"` key (reused from the scripted fixture) -- it is simply ignored; the
    real model produces its own tool calls and text instead.

    On a transient provider error (rate limit, timeout), retries once after a short
    backoff. A persistent failure is reported with `provider_failure=True` and verdict
    "UNMEASURED" -- never as a FAIL, since that would misrepresent a quota/network
    problem as the model having gotten the case wrong."""
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            provider = build_real_provider()
            return _run_with_provider(case, record, provider)
        except (LLMProviderError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))
                continue

    return HardCaseResult(
        case_id=case["case_id"],
        category=case["category"],
        verdict="UNMEASURED",
        dimension_checks=[DimensionCheck("provider_call", None, str(last_error))],
        explanation=f"PROVIDER_ERROR: {last_error}",
        result=None,
        provider_failure=True,
    )
