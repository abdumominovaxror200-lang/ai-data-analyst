"""Evidence-driven hypothesis status (Phase 4 P1).

Closes the gap logged in `.agent/decisions.md` ("Phase 4 findings", item 2) and
`.agent/completed_tasks.md`'s Phase 3C section: `Hypothesis.status` was set to
`"untested"` at creation (`planner.py`) and never updated by anything in the real
pipeline, which made `causation_guard.py`'s "a supported causal hypothesis may use
unhedged language" branch permanently dead code.

This module is the missing update step. It is entirely deterministic -- no LLM call,
no new tool call -- following the same spirit as `verifier.py`'s
`_extract_uncertainty`/`_cross_check`: read the real fields tool results already
produced (`Evidence.result_summary`), never invent new ones, never let the LLM simply
declare a hypothesis "supported."

--- Design: linking (which evidence is "about" which hypothesis) -----------------

A deliberately simple bag-of-words overlap heuristic, in the same honest-about-its-
limits spirit as this project's other documented heuristics (e.g. `eda.py`'s
cardinality thresholds). This is NOT real NLP / semantic matching -- it is a token-set
intersection test:

  - `hypothesis.description` is tokenized (lowercased, alphanumeric words only,
    English stopwords and words of length <= 2 dropped).
  - Each `Evidence` item is tokenized the same way from: its `metric` field, its
    `source_tool` name (underscores treated as word breaks), and a SMALL, fixed set
    of string-valued fields inside `result_summary` that name what was analyzed
    (`test`, `column`, `group_column`, `value_column`, `target_column`, and the
    `label` of `group_a`/`group_b` when those are the nested-dict shape
    `hypothesis.py`/`comparison.py` tools produce). This is intentionally shallow --
    it does not recursively walk arbitrary result payloads.
  - Evidence is "linked" to a hypothesis when the two token sets share at least one
    word. A hypothesis with zero linked evidence stays `"untested"`.

Known failure modes of this heuristic (documented, not hidden): synonyms are not
matched ("churn" vs. "customer loss"), a shared generic word can create a spurious
link, and a real semantic relationship with zero shared vocabulary is missed
entirely. This mirrors the same trade-off `causation_guard.py`'s literal phrase list
already makes elsewhere in this codebase -- correctness on the common case, honestly
scoped, not a claim of true language understanding.

--- Design: scoring (given linked evidence, what status) -------------------------

Each linked `Evidence` item is classified individually into one bucket:

  - `significant_opposing` -- has a real `significant` field (a formal test:
    t_test/chi_square_test/anova_test/... -- see `_evidence_type_for_tool` in
    `executor.py`), `significant is True`, AND a direction can be computed for BOTH
    the hypothesis and the evidence, and they disagree.
  - `significant_matching`  -- has `significant is True` and direction either
    agrees or could not be computed on one/both sides (ambiguity never manufactures
    a contradiction -- see "Direction matching" below).
  - `non_significant`       -- has a `significant` field and it is `False`.
  - `suggestive_pattern`    -- no `significant` field, but a directional numeric
    result (`delta`/`pct_change`, the `compare_periods` shape) whose sign agrees
    with the hypothesis's claimed direction (or the hypothesis claims no direction),
    and the magnitude is not negligible.
  - `confident_null`        -- no `significant` field, but a directional numeric
    result whose magnitude is negligible (see `_NULL_PCT_CHANGE_THRESHOLD`) -- a
    real "nothing happened" reading, not just "we didn't formally test it."
  - `ambiguous`             -- anything linked that doesn't cleanly fit above (e.g.
    an `effect_size` result with no `significant`/directional field of its own).

Separately, ANY linked evidence item carrying an `effect_size`-tool-shaped
`magnitude` field of `"small"`/`"negligible"` is tracked as a small-effect signal,
independent of which bucket it landed in above -- this is what lets a `t_test` item
(no magnitude field of its own) be downgraded when a *separately linked*
`effect_size` result on the same relationship says the difference is small. This
cross-referencing (not a per-item-only check) is deliberate: the two tools produce
independent evidence items about the same relationship, and this module's job is to
combine what was actually gathered, not to grade each tool call in isolation.

Per-hypothesis aggregation (priority order, evaluated over ALL linked evidence):

  1. Any `significant_opposing`                              -> "contradicted"
  2. Any `significant_matching` AND no small-effect signal    -> "supported"
  3. Any `significant_matching` WITH a small-effect signal,
     or any `suggestive_pattern` (regardless of significance) -> "weakly_supported"
  4. Any `confident_null` (and nothing above matched)         -> "unsupported"
  5. Otherwise (only `non_significant`/`ambiguous` linked)    -> "inconclusive"
  0. No linked evidence at all                                -> "untested"

The three subtlest statuses, explicitly distinguished:
  - `"unsupported"` = linked evidence exists and POSITIVELY shows no effect/
    relationship (a confident null: e.g. `compare_periods` finding ~0% change).
    Confidence here comes from the evidence's own magnitude, not from absence of
    testing.
  - `"inconclusive"` = linked evidence exists but is weak or ambiguous -- a
    non-significant test with no directional/effect-size read on it, or a
    directional field that doesn't cleanly indicate "no effect" either. We
    genuinely don't know.
  - `"contradicted"` = linked evidence is not just unhelpful, it actively points
    the OPPOSITE way from what the hypothesis claims, and does so with statistical
    significance and a resolvable direction behind it.

--- Direction matching (documented limitation) ------------------------------------

Both the hypothesis's claimed direction and each evidence item's actual direction are
determined by small, literal heuristics and are frequently `None` (unknown):

  - Hypothesis direction: substring keyword match on the description text
    ("declin"/"drop"/"lower"/... => "decrease"; "increas"/"ris"/"higher"/... =>
    "increase"). No match => direction is unclaimed/unknown.
  - Evidence direction: only computed for shapes where "which side is the outcome"
    is unambiguous -- a one-sample t-test's `mean` vs. `popmean`, or a `delta`/
    `pct_change` field (the `compare_periods` shape: current vs. previous is
    well-defined). A two-sample t-test/ANOVA's `group_a` vs. `group_b` does NOT
    produce a computed direction here -- which group represents "before"/"after"
    (or "the effect" vs. "the baseline") is not recoverable from the result shape
    alone, so per the task spec's own escape hatch, those items fall back to
    "significance only, no direction check" rather than a guessed and
    possibly-wrong direction.

Per the task spec: when EITHER side's direction is unknown, direction-matching is
skipped and significance alone determines `significant_matching` vs.
`significant_opposing` -- ambiguity never manufactures a false contradiction.

--- Non-mutation guarantee ---------------------------------------------------------

`update_hypothesis_status` returns a NEW list of NEW `Hypothesis` objects
(`.model_copy(update={...})`, pydantic's documented non-mutating copy constructor).
The input `hypotheses` list and its objects, and the `evidence`/`findings` lists, are
never mutated -- callers (the orchestrator, once this is wired in) can safely keep
using their original `plan.hypotheses` reference alongside the returned list.
"""

from __future__ import annotations

import re

from app.reasoning.contracts import Evidence, Finding, Hypothesis, HypothesisStatus

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "to", "of",
    "for", "and", "or", "by", "with", "that", "this", "did", "does", "do", "has",
    "have", "had", "be", "been", "it", "its", "as", "than", "then", "why", "what",
    "which", "not", "no", "there", "their", "from", "into", "over", "about",
}

_DECREASE_WORDS = (
    "declin", "decreas", "drop", "fell", "fall", "lower", "down", "reduc",
    "shrink", "shrank", "worse", "loss", "negative", "slump", "plunge", "slowdown",
)
_INCREASE_WORDS = (
    "increas", "ris", "rose", "higher", "up", "grew", "grow", "improv", "gain",
    "positive", "surge", "spike", "boost", "jump",
)

# CALCULATED_RESULT directional field (compare_periods-shaped `pct_change`) whose
# magnitude is small enough to call "no real change" rather than merely "not
# statistically tested". Deliberately a loose, documented threshold in the same
# spirit as this project's other fixed heuristic thresholds -- not derived from any
# formal test.
_NULL_PCT_CHANGE_THRESHOLD = 1.0  # percentage points

_SMALL_EFFECT_MAGNITUDES = ("small", "negligible")

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Fields inside Evidence.result_summary that name what was analyzed -- deliberately a
# small, fixed allowlist (not a recursive walk of the whole payload), matching the
# "shallow, documented heuristic" standard set elsewhere in this module.
_METRIC_NAME_FIELDS = ("test", "column", "group_column", "value_column", "target_column")
_GROUP_LABEL_FIELDS = ("group_a", "group_b")


def _tokenize(text: str) -> set[str]:
    words = _TOKEN_RE.findall(text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _hypothesis_tokens(hypothesis: Hypothesis) -> set[str]:
    return _tokenize(hypothesis.description)


def _evidence_tokens(evidence: Evidence) -> set[str]:
    tokens: set[str] = set()
    if evidence.metric:
        tokens |= _tokenize(evidence.metric)
    tokens |= _tokenize(evidence.source_tool.replace("_", " "))

    r = evidence.result_summary or {}
    for key in _METRIC_NAME_FIELDS:
        value = r.get(key)
        if isinstance(value, str):
            tokens |= _tokenize(value)
    for key in _GROUP_LABEL_FIELDS:
        value = r.get(key)
        if isinstance(value, dict):
            label = value.get("label")
            if isinstance(label, str):
                tokens |= _tokenize(label)
    return tokens


def _is_linked(hyp_tokens: set[str], ev_tokens: set[str]) -> bool:
    return len(hyp_tokens & ev_tokens) > 0


def _claimed_direction(description: str) -> str | None:
    text = description.lower()
    if any(w in text for w in _DECREASE_WORDS):
        return "decrease"
    if any(w in text for w in _INCREASE_WORDS):
        return "increase"
    return None


def _evidence_direction(evidence: Evidence) -> str | None:
    """Only computed for result shapes where "which side is the outcome" is
    unambiguous. See the module docstring's "Direction matching" section for why a
    two-sample t-test/ANOVA's group_a/group_b is deliberately excluded here."""
    r = evidence.result_summary or {}

    mean = r.get("mean")
    popmean = r.get("popmean")
    if isinstance(mean, (int, float)) and isinstance(popmean, (int, float)):
        if mean < popmean:
            return "decrease"
        if mean > popmean:
            return "increase"
        return None

    delta = r.get("delta")
    if isinstance(delta, (int, float)):
        if delta < 0:
            return "decrease"
        if delta > 0:
            return "increase"
        return None

    pct_change = r.get("pct_change")
    if isinstance(pct_change, (int, float)):
        if pct_change < 0:
            return "decrease"
        if pct_change > 0:
            return "increase"
        return None

    return None


def _has_small_effect_signal(evidence: Evidence) -> bool:
    """True when this evidence item itself carries an `effect_size`-tool-shaped
    `magnitude` field of "small"/"negligible". Checked across ALL linked evidence
    (not just the significant-test item) so a separately linked `effect_size` call
    can downgrade a plain significant test -- see the module docstring."""
    magnitude = (evidence.result_summary or {}).get("magnitude")
    return magnitude in _SMALL_EFFECT_MAGNITUDES


def _is_confident_null(evidence: Evidence) -> bool:
    """A directional CALCULATED_RESULT field (`pct_change`) that is essentially
    zero -- a real "nothing happened" reading, distinct from "we never formally
    tested it" (that case is `ambiguous`, not `confident_null`)."""
    pct_change = (evidence.result_summary or {}).get("pct_change")
    if isinstance(pct_change, (int, float)):
        return abs(pct_change) < _NULL_PCT_CHANGE_THRESHOLD
    return False


_EvidenceBucket = str  # one of the classifications documented in the module docstring


def _classify_linked_evidence(evidence: Evidence, hyp_direction: str | None) -> _EvidenceBucket:
    r = evidence.result_summary or {}

    if "significant" in r:
        significant = bool(r.get("significant"))
        if not significant:
            return "non_significant"

        ev_direction = _evidence_direction(evidence)
        if hyp_direction is not None and ev_direction is not None and hyp_direction != ev_direction:
            return "significant_opposing"
        return "significant_matching"

    # No formal significance field on this item -- an aggregate/descriptive result
    # (e.g. compare_periods), or an effect_size-only result (handled via the
    # separate small-effect-signal check, not this bucket).
    ev_direction = _evidence_direction(evidence)
    if ev_direction is not None:
        if _is_confident_null(evidence):
            return "confident_null"
        if hyp_direction is None or hyp_direction == ev_direction:
            return "suggestive_pattern"
        # A CALCULATED_RESULT pointing the opposite way from a claimed direction is
        # not treated as "contradicted" (that status is reserved for statistically
        # significant opposition, per spec) -- it is simply not supportive.
        return "ambiguous"

    return "ambiguous"


def _score_hypothesis(
    hypothesis: Hypothesis, linked: list[Evidence]
) -> tuple[HypothesisStatus, list[str], list[str]]:
    if not linked:
        return "untested", [], []

    hyp_direction = _claimed_direction(hypothesis.description)
    buckets: dict[str, list[Evidence]] = {}
    for ev in linked:
        buckets.setdefault(_classify_linked_evidence(ev, hyp_direction), []).append(ev)
    small_effect_evidence = [ev for ev in linked if _has_small_effect_signal(ev)]

    if buckets.get("significant_opposing"):
        against = [e.id for e in buckets["significant_opposing"]]
        supporting = buckets.get("significant_matching", []) + buckets.get("suggestive_pattern", [])
        return "contradicted", [e.id for e in supporting], against

    significant_matching = buckets.get("significant_matching", [])
    suggestive_pattern = buckets.get("suggestive_pattern", [])

    if significant_matching and not small_effect_evidence:
        return "supported", [e.id for e in significant_matching], []

    if (significant_matching and small_effect_evidence) or suggestive_pattern:
        supporting_ids = {e.id for e in significant_matching + suggestive_pattern + small_effect_evidence}
        return "weakly_supported", sorted(supporting_ids), []

    if buckets.get("confident_null"):
        return "unsupported", [], []

    # Only non_significant/ambiguous items linked -- a real signal was looked for
    # but what came back doesn't clearly say yes or no.
    return "inconclusive", [], []


def update_hypothesis_status(
    hypotheses: list[Hypothesis], evidence: list[Evidence], findings: list[Finding]
) -> list[Hypothesis]:
    """Deterministically derives each hypothesis's `status` (and, where the linking
    heuristic can genuinely attribute it, `evidence_for`/`evidence_against`) from the
    evidence actually gathered for this analysis. See the module docstring for the
    full linking + scoring design. Never mutates its inputs; always returns a new list
    of new `Hypothesis` objects (`.model_copy(update=...)`).

    `findings` is accepted for interface symmetry with `verifier.build_findings`'s
    output and to leave room for a future richer scoring pass (e.g. weighing
    `Finding.uncertainty`/`cross_checked`) -- the current scoring rule reads
    `Evidence.result_summary` directly, per the task's "deterministic function of
    tool results" requirement, and does not need `findings` today.
    """
    del findings  # not needed by the current deterministic rule -- see docstring

    updated: list[Hypothesis] = []
    for hypothesis in hypotheses:
        hyp_tokens = _hypothesis_tokens(hypothesis)
        linked = [
            ev for ev in evidence
            if _is_linked(hyp_tokens, _evidence_tokens(ev))
            and (not hypothesis.is_causal or ev.causal_eligible)
        ]
        status, evidence_for, evidence_against = _score_hypothesis(hypothesis, linked)
        updated.append(
            hypothesis.model_copy(
                update={
                    "status": status,
                    "evidence_for": evidence_for,
                    "evidence_against": evidence_against,
                }
            )
        )
    return updated
