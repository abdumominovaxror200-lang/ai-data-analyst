"""Causation guard (Phase 3B.6).

Deterministic, standalone, and independently testable on purpose — the synthesizer's
system prompt is the *first* line of defense (instructing the LLM not to overclaim),
but per this project's established pattern (see the "sandwich" mitigation in
`backend/docs/security/prompt-injection-trust-boundary.md`), a single prompt-level
instruction is never treated as a hard guarantee on its own. This module is the
second, code-level layer: it scans the model's own output for unhedged causal
language and, if found without a supporting causal hypothesis, deterministically
rewrites it to hedged language rather than trusting the model to have complied.

Allowed without a causal hypothesis: description ("revenue decreased by 5.2%"),
ranking ("Region A had the largest decline"), and association ("Region A and the
decline are associated").

Requires an explicit, evidence-backed causal `Hypothesis` (is_causal=True and
status in {"supported", "weakly_supported"}) before the *specific* causal relationship
it names may appear in unhedged form.
"""

from __future__ import annotations

import re

from app.reasoning.contracts import Hypothesis

# Ordered so longer/more specific phrases are checked before their substrings would be
# (not load-bearing here since re.sub already handles overlap correctly per phrase,
# but kept for readability).
_CAUSAL_PHRASE_HEDGES: dict[str, str] = {
    r"\bcaused by\b": "is associated with",
    r"\bcaused\b": "is associated with",
    r"\bcauses?\b": "is associated with",
    r"\bcausing\b": "contributing to",
    r"\bdue to\b": "possibly related to",
    r"\bbecause of\b": "possibly related to",
    r"\bled to\b": "may have contributed to",
    r"\bleads? to\b": "may contribute to",
    r"\bresulted? in\b": "is consistent with",
    r"\bresulting in\b": "consistent with",
    r"\bis the reason (for|behind)\b": "is a possible explanation for",
    r"\bis why\b": "may help explain why",
    r"\bdrove\b": "is associated with",
    r"\bdriving\b": "associated with",
}

_CAUSAL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _CAUSAL_PHRASE_HEDGES]


def find_causal_phrases(text: str) -> list[str]:
    """Returns every distinct causal phrase literally matched in `text` (for tests
    and for logging), case-insensitive."""
    found = []
    for pattern in _CAUSAL_PATTERNS:
        for match in pattern.finditer(text):
            found.append(match.group(0))
    return found


def _has_justifying_causal_hypothesis(hypotheses: list[Hypothesis]) -> bool:
    return any(h.is_causal and h.status in ("supported", "weakly_supported") for h in hypotheses)


def enforce_causation_guard(text: str, hypotheses: list[Hypothesis]) -> tuple[str, bool, list[str]]:
    """Returns (possibly-rewritten text, was_rewritten, matched_phrases).

    If `text` contains causal language and no hypothesis in `hypotheses` justifies a
    causal claim (is_causal=True with supporting status), every matched causal phrase
    is deterministically replaced with its hedged equivalent. If a justifying
    hypothesis exists, the text is returned unchanged — this guard restricts
    *unsupported* causal language, it does not ban causal language outright when the
    system has actually done the work to support it.
    """
    matched = find_causal_phrases(text)
    if not matched:
        return text, False, []
    if _has_justifying_causal_hypothesis(hypotheses):
        return text, False, matched

    rewritten = text
    for pattern, hedge in _CAUSAL_PHRASE_HEDGES.items():
        rewritten = re.sub(pattern, hedge, rewritten, flags=re.IGNORECASE)
    return rewritten, True, matched
