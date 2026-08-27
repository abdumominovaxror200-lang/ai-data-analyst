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

from app.reasoning.contracts import Limitation

_CAVEAT_MARKER = "Important caveat:"


def enforce_conclusion_guard(text: str, limitations: list[Limitation]) -> tuple[str, bool]:
    """Returns (possibly-caveated text, whether a caveat was added).

    Safe to call on any input: never raises, and idempotent against being applied
    twice (checks for its own marker before prepending again -- relevant since
    `synthesize()`'s early-stop paths and the main path both funnel through here).
    """
    blocking = [l for l in (limitations or []) if l.severity == "blocks_conclusion"]
    if not blocking or not text or _CAVEAT_MARKER in text:
        return text, False

    reasons = "; ".join(l.text for l in blocking)
    caveat = (
        f"{_CAVEAT_MARKER} at least one issue here is serious enough that a confident "
        f"conclusion is not justified from the available evidence alone ({reasons}). "
        "Treat the analysis below with that in mind.\n\n"
    )
    return caveat + text, True
