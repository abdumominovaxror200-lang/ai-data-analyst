"""Deterministic final-text safety net for `blocks_conclusion`-severity limitations
(final professional-analyst stress-test mission, Phase 4/6).

`recommendation_grounding.py`'s `blocks_conclusion` override already prevents a
structured `Recommendation.confidence` from surviving when a limitation this severe
is present (see that module's docstring). But `final_answer_text` -- the free-text
prose the user actually reads -- is a SEPARATE field the LLM writes independently, and
nothing previously touched it: a scripted/real model could correctly get its
`recommendation` field capped to `None` while `final_answer_text` still read as a
flatly confident, unhedged conclusion ("Yes, North is clearly a better-performing
region..."). Found as a real gap while auditing the hard real-world benchmark's
honesty-vs-overclaiming adversarial pairs: an overclaiming script's answer text
carried no caveat at all despite a severe, correctly-detected confound.

Mirrors `causation_guard.enforce_causation_guard`'s existing "prompt instruction +
code-level check" pattern (see that module and `synthesizer.py`'s own docstring) --
this is the second half of that same defense-in-depth idea, generalized from
"unhedged causal language" to "any conclusion the evidence structurally cannot
support at all". Deliberately a prepended caveat sentence, not a rewrite of the
model's own wording (unlike `causation_guard`, which substitutes specific matched
phrases in place) -- there is no reliable way to know which specific clause of an
arbitrary free-text answer needs softening for an unbounded set of possible
blocking reasons, so this makes the limitation impossible to miss instead of trying
to edit around it.
"""

from __future__ import annotations

import re

from app.reasoning.contracts import Finding, Hypothesis, Limitation

_CAVEAT_MARKER = "Important caveat:"

# Narrow, high-confidence patterns: these target conclusion/recommendation language,
# not ordinary factual comparisons. The prompt is the broad first line of defence;
# this list is the deterministic safety boundary when the model ignores it.
_DEFINITIVE_DRIVER_PATTERNS = (
    r"\b(?:primary|dominant|main|key|root) (?:driver|cause)\b",
    r"\broot cause\b",
    r"\b(?:was|is) driven by\b",
    r"\bcaused by\b",
    r"\b(?:explains?|accounts? for|responsible for) (?:most|the majority|the decline|the change)\b",
)
_RECOMMENDATION_PATTERNS = (
    r"\b(?:recommend|recommendation|recommended)\b",
    r"\bshould\b",
    r"\bconsider\b",
    r"\bprioriti[sz]e\b",
    r"\b(?:take|implement|pursue) (?:action|steps?|measures?)\b",
    r"\bnext action\b",
)


def blocked_narrative_violations(text: str) -> list[str]:
    """Return deterministic violation classes found in a blocked narrative."""
    violations = []
    if any(re.search(pattern, text or "", re.IGNORECASE) for pattern in _DEFINITIVE_DRIVER_PATTERNS):
        violations.append("definitive_driver")
    if any(re.search(pattern, text or "", re.IGNORECASE) for pattern in _RECOMMENDATION_PATTERNS):
        violations.append("recommendation_language")
    return violations


def enforce_conclusion_guard(
    text: str,
    limitations: list[Limitation],
    findings: list[Finding] | None = None,
) -> tuple[str, bool]:
    """Returns (possibly-caveated text, whether a caveat was added).

    Safe to call on any input: never raises, and idempotent against being applied
    twice (checks for its own marker before prepending again -- relevant since
    `synthesize()`'s early-stop paths and the main path both funnel through here).
    """
    blocking = [l for l in (limitations or []) if l.severity == "blocks_conclusion"]
    if not blocking or not text:
        return text, False

    if blocked_narrative_violations(text):
        return _safe_blocked_response(findings or [], blocking), True

    if _CAVEAT_MARKER in text:
        return text, False

    reasons = "; ".join(l.text for l in blocking)
    caveat = (
        f"{_CAVEAT_MARKER} at least one issue here is serious enough that a confident "
        f"conclusion is not justified from the available evidence alone ({reasons}). "
        "Treat the analysis below with that in mind.\n\n"
    )
    return caveat + text, True


def _safe_blocked_response(findings: list[Finding], limitations: list[Limitation]) -> str:
    safe_findings = [
        finding.statement for finding in findings
        if finding.classification in ("FACT", "CALCULATED_RESULT", "STATISTICAL_RESULT")
        and not blocked_narrative_violations(finding.statement)
    ]
    lines = [
        "A definitive conclusion cannot be made from the available evidence.",
        "",
        "Verified findings:",
    ]
    lines.extend(f"- {statement}" for statement in safe_findings)
    if not safe_findings:
        lines.append("- No verified finding is sufficient to support a conclusion.")
    lines.extend(["", "Limitations:"])
    for limitation in limitations:
        if blocked_narrative_violations(limitation.text):
            lines.append(f"- A {limitation.category} limitation prevents a conclusion.")
        else:
            lines.append(f"- {limitation.text}")
    lines.extend(["", "Additional evidence is required before identifying a driver or taking action."])
    return "\n".join(lines)


def sanitize_blocked_hypotheses(
    hypotheses: list[Hypothesis], limitations: list[Limitation]
) -> list[Hypothesis]:
    """Remove causal/driver claims and support labels from every visible hypothesis."""
    if not any(item.severity == "blocks_conclusion" for item in limitations):
        return hypotheses
    return [
        item.model_copy(update={
            "description": "This hypothesis remains unresolved; additional evidence is required.",
            "is_causal": False,
            "status": "inconclusive",
        })
        for item in hypotheses
    ]
