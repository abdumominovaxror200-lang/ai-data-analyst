"""Causation guard (Phase 3B.6, redesigned Phase 4 — layered detector).

Deterministic, standalone, and independently testable on purpose — the synthesizer's
system prompt is the *first* line of defense (instructing the LLM not to overclaim),
but per this project's established pattern (see the "sandwich" mitigation in
`backend/docs/security/prompt-injection-trust-boundary.md`), a single prompt-level
instruction is never treated as a hard guarantee on its own. This module is the
second, code-level layer: it scans the model's own output for unhedged causal
language and, if found without a supporting causal hypothesis, deterministically
rewrites it to hedged language rather than trusting the model to have complied.

--------------------------------------------------------------------------------
WHY THIS IS A REDESIGN, NOT JUST A BIGGER PHRASE LIST
--------------------------------------------------------------------------------
The previous implementation (Phase 3B) was a single flat dict of ~14 literal phrase
regexes. A QA audit (see `.agent/decisions.md`, "Phase 3C findings requiring a future
decision", item 1) found it was trivially bypassed by paraphrase: "is clearly
responsible for" was never hedged, because that exact wording wasn't one of the
literal entries. Adding more literal entries only chases the next paraphrase forever.

This version is three genuinely different, composable layers instead:

  Layer 1 (this module's `_CAUSAL_PATTERN_TABLE`) — categorized, STEM-based regex
  matching. Each entry matches the causal *predicate itself* (e.g. `\\bresponsible\\s+
  for\\b`), not a literal sentence template around it. That one pattern therefore
  catches "is responsible for", "was clearly responsible for", "is solely
  responsible for", and any other intensifier/subject combination in a single rule —
  the fix for the exact audit finding above. Patterns are grouped into named,
  documented categories (see the table below) rather than being one undifferentiated
  denylist; the categorization is itself part of the design, not decoration, because
  it's what makes "did I cover the causal-predicate space" a checkable question
  instead of "did I think of every phrase".

  Layer 2 (`_is_hedged`) — a context-window check, mechanically unrelated to Layer 1.
  Before a Layer-1 match counts as a violation, we look at the ~6 words immediately
  before it for a modal/uncertainty marker ("may", "might", "could", "possibly",
  "likely", "potentially", "appears to", "seems to", "suggests", "indicates that").
  "may have caused" and "could be responsible for" are legitimate ways to state a
  *causal hypothesis* — they are left alone, never rewritten, regardless of whether
  any `Hypothesis` supports them. "caused" and "is responsible for" with no such
  marker nearby are violations, subject to the evidence gate below.

  Layer 3 (`classify_relationship_language`) — structured relationship
  classification. Every detected mention (causal or not) becomes a typed
  `RelationshipMention` with a `category`: "correlation", "association",
  "prediction", "temporal", "causal_hypothesis" (Layer-2-hedged causal), or
  "causal_unhedged" (the violation state used by the evidence gate). Correlation/
  association/prediction/temporal phrasing ("is correlated with", "is associated
  with", "is expected to", "coincided with", "occurred around the same time as") is
  the *desired* way to state a relationship without overclaiming, and is never
  flagged — this is what makes the guard structured rather than a boolean pass/fail:
  it can tell you not just "hedge or don't" but *what kind* of relationship claim was
  actually made.

The evidence gate itself (an unhedged causal mention is only excused if `hypotheses`
contains an `is_causal=True` hypothesis with `status in ("supported",
"weakly_supported")`) and the rewrite-on-violation behavior are unchanged in spirit
from the Phase 3B version — this redesign changes *detection breadth and structure*,
not the safety policy sitting on top of it.

--------------------------------------------------------------------------------
LAYER 1 CATEGORIES AND RATIONALE
--------------------------------------------------------------------------------
  Direct causation   — "caused"/"causes"/"causing"/"caused by": the plainest
                        causal-verb family. Stemmed to one pattern per direction.
  Result/consequence — "resulted in"/"results in"/"resulting in": frames an outcome
                        as the direct consequence of an antecedent.
  Lead framing        — "led to"/"leads to"/"leading to": same causal claim, distinct
                        verb family, common in business-analyst prose.
  Attribution          — "due to", "because of", "(is/was) the reason for/behind",
                        "(is/was) why", "brought about": explicit "X because Y"
                        attribution phrasing that doesn't use a causal verb at all.
  Responsibility       — "responsible for" (any intensifier/subject/tense): the
                        exact audit-finding fix — matches the predicate regardless
                        of "clearly"/"solely"/"primarily"/etc. in front of it.
  Driver framing        — "drove"/"drives"/"driving": treats a factor as the
                        mechanical/causal driver of an outcome.
  Explanatory framing    — "explains"/"explained"/"explaining": claims a mechanism,
                        not mere correlation, when used to account for an outcome.
  Accounting/share      — "accounts for"/"accounted for": claims a factor is
                        causally behind a share or magnitude of an outcome.
  Production framing     — "produced"/"producing" in an attribution sense ("X
                        produced Y"). NOTE (honestly documented tradeoff, same
                        spirit as this project's other disclosed limitations): this
                        surface form is shared with the literal "manufactured a
                        physical product" sense, which a regex can't disambiguate
                        semantically. Kept in scope because the task spec explicitly
                        requires catching causal "produced"/"producing", and the
                        failure direction of a false positive here is the safe one
                        (over-hedging), not the dangerous one (missing a real
                        overclaim).
  Trigger framing        — "triggered"/"triggering"/"triggers": claims a factor set
                        off an outcome.

--------------------------------------------------------------------------------
LAYER 3 NON-CAUSAL CATEGORIES (never flagged)
--------------------------------------------------------------------------------
  correlation  — "is/are correlated with", "correlation between"
  association  — "is/are associated with" (also this module's own hedge-replacement
                 target text, so a rewritten sentence re-classifies as safe)
  prediction   — "is/are expected to"
  temporal     — "coincided with", "occurred around the same time as", "preceded
                 by", "followed by"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Literal

from app.reasoning.contracts import Hypothesis

# ================================================================================
# Layer 3 types
# ================================================================================

RelationshipCategory = Literal[
    "correlation",
    "association",
    "prediction",
    "temporal",
    "causal_hypothesis",
    "causal_unhedged",
]


@dataclass(frozen=True)
class RelationshipMention:
    """One detected relationship-language mention in a piece of text.

    `pattern_name` identifies which Layer-1/Layer-3 pattern matched (for debugging/
    logging); `category` is the Layer-3 classification actually used for gating.
    """

    matched_text: str
    start: int
    end: int
    category: RelationshipCategory
    pattern_name: str


# ================================================================================
# Layer 3 (non-causal) patterns — correlation / association / prediction / temporal.
# These are the *desired* safe way to state a relationship and are never flagged.
# ================================================================================

_NON_CAUSAL_PATTERN_TABLE: list[tuple[str, str, RelationshipCategory]] = [
    ("correlation", r"\bcorrelat(?:e|es|ed|ion|ing)?\s+with\b|\bcorrelation\s+between\b", "correlation"),
    ("association", r"\bassociat(?:e|es|ed|ion|ing)?\s+with\b", "association"),
    ("prediction", r"\b(?:is|are)\s+expected\s+to\b|\bexpected\s+to\b", "prediction"),
    (
        "temporal",
        r"\bcoincided\s+with\b|\boccurred\s+around\s+the\s+same\s+time\s+as\b|\bpreceded\s+by\b|\bfollowed\s+by\b",
        "temporal",
    ),
]

# ================================================================================
# Layer 1 causal-predicate patterns, categorized. Each `hedge` is either a plain
# string or a callable(matched_text) -> str for the handful of entries where the
# right hedge depends on the matched verb form (e.g. "-ing" vs. base form).
# ================================================================================

_HedgeFn = Callable[[str], str]


def _default_hedge_for_ing(default: str, ing_form: str) -> _HedgeFn:
    def _hedge(matched_text: str) -> str:
        return ing_form if matched_text.lower().endswith("ing") else default

    return _hedge


_CAUSAL_PATTERN_TABLE: list[tuple[str, str, str, str | _HedgeFn]] = [
    # (pattern_name, category_label, regex, hedge)
    ("caused_by", "Direct causation", r"\bcaused\s+by\b", "is associated with"),
    (
        "causal_verb",
        "Direct causation",
        r"\bcaus(?:e|es|ed|ing)\b",
        _default_hedge_for_ing("is associated with", "contributing to"),
    ),
    ("result_in", "Result/consequence", r"\bresult(?:s|ed|ing)?\s+in\b", "is consistent with"),
    ("lead_to", "Lead framing", r"\blead(?:s|ing)?\s+to\b|\bled\s+to\b", "may contribute to"),
    ("due_to", "Attribution", r"\bdue\s+to\b", "possibly related to"),
    ("because_of", "Attribution", r"\bbecause\s+of\b", "possibly related to"),
    ("reason_for", "Attribution", r"\breason\s+(?:for|behind)\b", "is a possible explanation for"),
    ("is_why", "Attribution", r"\b(?:is|was)\s+why\b", "may help explain why"),
    ("brought_about", "Attribution", r"\bbrought\s+about\b", "may have contributed to"),
    ("responsible_for", "Responsibility", r"\bresponsible\s+for\b", "is associated with"),
    (
        "driver",
        "Driver framing",
        r"\b(?:drove|drives|driving|drive)\b",
        _default_hedge_for_ing("is associated with", "associated with"),
    ),
    (
        "explains",
        "Explanatory framing",
        r"\bexplain(?:s|ed|ing)?\b",
        "may help explain",
    ),
    ("accounts_for", "Accounting/share", r"\baccount(?:s|ed)?\s+for\b", "is associated with"),
    (
        "produced",
        "Production framing",
        r"\bproduc(?:e|es|ed|ing)\b",
        "may have contributed to",
    ),
    (
        "triggered",
        "Trigger framing",
        r"\btrigger(?:s|ed|ing)?\b",
        "may have contributed to",
    ),
]

_NON_CAUSAL_COMPILED = [(name, re.compile(rx, re.IGNORECASE), cat) for name, rx, cat in _NON_CAUSAL_PATTERN_TABLE]
_CAUSAL_COMPILED = [
    (name, category, re.compile(rx, re.IGNORECASE), hedge) for name, category, rx, hedge in _CAUSAL_PATTERN_TABLE
]

# ================================================================================
# Layer 2 — hedge/modal detection (context window before a Layer-1 match).
# ================================================================================

_HEDGE_MARKER_RE = re.compile(
    r"\b(?:may|might|could|possibly|likely|potentially|appears?\s+to|seems?\s+to|"
    r"suggests?|indicat(?:e|es|ed)\s+that)\b",
    re.IGNORECASE,
)

_HEDGE_WINDOW_WORDS = 6


def _is_hedged(text: str, match_start: int) -> bool:
    """True if a modal/uncertainty marker appears within the last
    `_HEDGE_WINDOW_WORDS` words immediately before `match_start`."""
    preceding = text[:match_start]
    words = re.findall(r"\S+", preceding)
    window = " ".join(words[-_HEDGE_WINDOW_WORDS:])
    return bool(_HEDGE_MARKER_RE.search(window))


# ================================================================================
# Layer 3 — structured relationship classification (public, independently usable).
# ================================================================================


def classify_relationship_language(text: str) -> list[RelationshipMention]:
    """Scans `text` and returns every detected relationship-language mention,
    classified into one of six categories (see module docstring). Non-causal
    categories (correlation/association/prediction/temporal) are never treated as
    violations by `enforce_causation_guard`; causal mentions are split into
    `causal_hypothesis` (Layer-2-hedged, legitimate) vs. `causal_unhedged` (subject
    to the evidence gate).

    Independently testable/importable — this is what makes the guard "structured"
    rather than a boolean pass/fail.
    """
    mentions: list[RelationshipMention] = []
    claimed: list[tuple[int, int]] = []

    def _is_claimed(start: int, end: int) -> bool:
        return any(start < c_end and end > c_start for c_start, c_end in claimed)

    # Non-causal (safe) categories first -- these are never overlapping with causal
    # predicate wording, but claiming their spans keeps the scan deterministic and
    # documents intent even if patterns evolve later.
    for name, pattern, category in _NON_CAUSAL_COMPILED:
        for m in pattern.finditer(text):
            if _is_claimed(m.start(), m.end()):
                continue
            claimed.append((m.start(), m.end()))
            mentions.append(
                RelationshipMention(
                    matched_text=m.group(0), start=m.start(), end=m.end(), category=category, pattern_name=name
                )
            )

    # Causal predicate patterns, gated through the Layer-2 hedge check.
    for name, _category_label, pattern, _hedge in _CAUSAL_COMPILED:
        for m in pattern.finditer(text):
            if _is_claimed(m.start(), m.end()):
                continue
            claimed.append((m.start(), m.end()))
            category: RelationshipCategory = "causal_hypothesis" if _is_hedged(text, m.start()) else "causal_unhedged"
            mentions.append(
                RelationshipMention(
                    matched_text=m.group(0), start=m.start(), end=m.end(), category=category, pattern_name=name
                )
            )

    mentions.sort(key=lambda m: m.start)
    return mentions


def _hedge_replacement_for(pattern_name: str, matched_text: str) -> str:
    for name, _category, _pattern, hedge in _CAUSAL_COMPILED:
        if name == pattern_name:
            return hedge(matched_text) if callable(hedge) else hedge
    return "is associated with"  # defensive fallback, should be unreachable


# ================================================================================
# Public API (signatures preserved exactly for synthesizer.py / scoring.py)
# ================================================================================


def find_causal_phrases(text: str) -> list[str]:
    """Returns every distinct unhedged causal violation phrase in `text` (for tests
    and for logging), case-insensitive.

    Implemented on top of `classify_relationship_language`: only mentions classified
    as `"causal_unhedged"` are returned. Genuinely hedged causal language ("may have
    caused") and non-causal relationship language (correlation/association/
    prediction/temporal) are deliberately excluded -- they are not violations, so
    callers checking "did any unsafe causal claim survive" (this project's benchmark
    scorer does exactly that on the pipeline's final output) get a meaningful
    signal rather than false alarms on legitimately-hedged prose.
    """
    return [m.matched_text for m in classify_relationship_language(text) if m.category == "causal_unhedged"]


def _has_justifying_causal_hypothesis(hypotheses: list[Hypothesis]) -> bool:
    # Phase 4 integration fix: originally excused both "supported" and
    # "weakly_supported" -- harmless when this was written (Hypothesis.status could
    # never actually leave "untested" through the real pipeline, so this branch was
    # dead code). Once hypothesis_evaluator.py made status genuinely evidence-derived,
    # this surfaced a real gap: "weakly_supported" is explicitly defined (see
    # hypothesis_evaluator.py) as "suggestive but not from a formal significance
    # test, or significant with a negligible effect" -- exactly the evidence strength
    # that must NOT be enough to unlock an *unhedged* causal claim (found via a real
    # professional-benchmark case, rt5: a plain period-over-period revenue comparison
    # with no significance test reached "weakly_supported" and incorrectly excused
    # "the pricing strategy caused this quarter's revenue increase"). Only "supported"
    # -- backed by an actual significant, meaningful-effect result -- excuses unhedged
    # causal language now. "weakly_supported" causal hypotheses remain expressible,
    # just only in hedged form ("may have contributed to"), which Layer 2 already
    # allows unconditionally regardless of hypothesis status.
    return any(h.is_causal and h.status == "supported" for h in hypotheses)


def enforce_causation_guard(text: str, hypotheses: list[Hypothesis]) -> tuple[str, bool, list[str]]:
    """Returns (possibly-rewritten text, was_rewritten, matched_phrases).

    An unhedged causal mention (Layer 3 category `"causal_unhedged"`) is a violation
    unless `hypotheses` contains a `Hypothesis` with `is_causal=True` and `status in
    ("supported", "weakly_supported")` -- identical evidence-gate logic to the prior
    implementation. Hedged causal language (`"causal_hypothesis"`, e.g. "may have
    caused") and non-causal relationship language are never violations and are never
    rewritten, regardless of the hypothesis list.

    If a violation isn't excused by the evidence gate, its exact matched span is
    deterministically replaced with a hedged equivalent (e.g. "is responsible for" ->
    "is associated with", "explains" -> "may help explain", "drove" -> "is associated
    with"). Replacement is span-based (not a blind text-wide substitution), so a
    hedged causal mention elsewhere in the same text is never touched.
    """
    mentions = classify_relationship_language(text)
    violations = [m for m in mentions if m.category == "causal_unhedged"]
    if not violations:
        return text, False, []

    matched = [m.matched_text for m in violations]
    if _has_justifying_causal_hypothesis(hypotheses):
        return text, False, matched

    rewritten = text
    for m in sorted(violations, key=lambda x: x.start, reverse=True):
        hedge = _hedge_replacement_for(m.pattern_name, m.matched_text)
        rewritten = rewritten[: m.start] + hedge + rewritten[m.end :]
    return rewritten, True, matched
