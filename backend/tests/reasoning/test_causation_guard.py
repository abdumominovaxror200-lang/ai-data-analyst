from __future__ import annotations

import pytest

from app.reasoning.causation_guard import (
    classify_relationship_language,
    enforce_causation_guard,
    find_causal_phrases,
)
from app.reasoning.contracts import Hypothesis

# =============================================================================
# Original Phase 3B suite (behavior preserved; the redesign must still pass
# these exactly as written -- they document still-valid, still-required
# behavior: descriptive/comparative/associative language is never flagged, an
# unhedged causal claim is detected/rewritten absent supporting evidence, and a
# supported causal hypothesis permits causal language through unchanged).
# =============================================================================


def test_descriptive_statements_are_never_flagged():
    for text in [
        "Revenue decreased by 5.2%.",
        "Region A had the largest decline.",
        "Region A and revenue decline are associated.",
        "Region A's decline is consistent with a seasonal pattern.",
    ]:
        assert find_causal_phrases(text) == []
        rewritten, hedged, matched = enforce_causation_guard(text, [])
        assert hedged is False
        assert rewritten == text


def test_unhedged_causal_language_is_detected_without_a_hypothesis():
    text = "Region A caused the revenue decline."
    assert find_causal_phrases(text) != []


def test_unhedged_causal_language_is_rewritten_when_no_causal_hypothesis_exists():
    text = "Region A caused the revenue decline."
    rewritten, hedged, matched = enforce_causation_guard(text, [])
    assert hedged is True
    assert "caused" not in rewritten.lower()
    assert matched  # something was actually detected


def test_unhedged_causal_language_is_rewritten_even_with_an_unsupported_hypothesis():
    hyps = [Hypothesis(id="h1", description="Region A caused decline", is_causal=True, status="unsupported")]
    text = "Region A caused the revenue decline."
    rewritten, hedged, matched = enforce_causation_guard(text, hyps)
    assert hedged is True
    assert "caused" not in rewritten.lower()


def test_causal_language_is_permitted_when_a_supported_causal_hypothesis_exists():
    hyps = [Hypothesis(id="h1", description="Region A caused decline", is_causal=True, status="supported")]
    text = "Region A caused the revenue decline."
    rewritten, hedged, matched = enforce_causation_guard(text, hyps)
    assert hedged is False
    assert rewritten == text


def test_weakly_supported_hypothesis_does_not_excuse_unhedged_causal_language():
    """Tightened during Phase 4 integration: 'weakly_supported' is explicitly weak/
    untested-significance evidence (see hypothesis_evaluator.py) -- discovered via a
    real professional-benchmark case (rt5) where a plain period-over-period
    comparison, with no significance test, reached 'weakly_supported' and incorrectly
    excused an unhedged causal claim. Only 'supported' excuses unhedged language now."""
    hyps = [Hypothesis(id="h1", description="Region A caused decline", is_causal=True, status="weakly_supported")]
    rewritten, hedged, _ = enforce_causation_guard("This was due to Region A.", hyps)
    assert hedged is True
    assert "due to" not in rewritten.lower()


def test_non_causal_hypothesis_does_not_excuse_causal_language():
    hyps = [Hypothesis(id="h1", description="Region A is associated with decline", is_causal=False, status="supported")]
    rewritten, hedged, _ = enforce_causation_guard("Region A caused the decline.", hyps)
    assert hedged is True


def test_various_causal_phrasings_are_all_caught():
    phrases = [
        "The decline was due to Region A.",
        "Region A led to the decline.",
        "This resulted in a large drop.",
        "Marketing spend is why revenue fell.",
        "Region A drove the decline.",
    ]
    for text in phrases:
        assert find_causal_phrases(text), f"expected a causal phrase to be found in: {text!r}"


# =============================================================================
# NEW -- the core deliverable. This is the exact, documented failure QA found:
# "is clearly responsible for" bypassed the old fixed literal-phrase list
# entirely. The new stem-based "responsible for" predicate pattern must catch
# it (and any other intensifier wrapped around it).
# =============================================================================


def test_previously_missed_paraphrase_is_now_caught():
    text = "Region A is clearly responsible for the revenue decline."
    matched = find_causal_phrases(text)
    assert matched, "the previously-missed paraphrase must now be detected"
    assert any("responsible for" in m.lower() for m in matched)

    rewritten, hedged, matched2 = enforce_causation_guard(text, [])
    assert hedged is True
    assert "responsible for" not in rewritten.lower()
    assert matched2

    mentions = classify_relationship_language(text)
    causal = [m for m in mentions if m.category == "causal_unhedged"]
    assert causal, "must be classified as an unhedged causal violation, not silently ignored"


# =============================================================================
# NEW -- every specifically-named phrase from the task spec, proven caught,
# each with at least one intensifier/tense variant to prove stem-based
# generalization (not a re-enumeration of literal phrases).
# =============================================================================


@pytest.mark.parametrize(
    "text",
    [
        "Region A is clearly responsible for the decline.",
        "Region A was the reason for the decline.",
        "The marketing change resulted in a revenue drop.",
        "Region A drove the decline.",
        "Region A was driving the decline all quarter.",
        "This chart explains the revenue drop.",
        "The marketing change explained the decline.",
        "Region A is responsible for the decline.",
        "Region A accounts for most of the decline.",
        "Region A accounted for the majority of the drop.",
        "Poor weather brought about the decline.",
        "The price change produced a sharp drop in volume.",
        "The price change is producing a sharp drop in volume.",
        "A supply shock triggered the decline.",
        "A supply shock is triggering further declines.",
        "Region A caused the decline.",
        "Region A causes the decline.",
        "Region A is causing the decline.",
        "The decline was due to Region A.",
        "The decline occurred because of Region A.",
        "Region A led to the decline.",
        "Region A's pricing leads to lower volume.",
    ],
)
def test_all_named_paraphrases_are_caught(text):
    assert find_causal_phrases(text), f"expected a causal violation to be found in: {text!r}"
    _, hedged, _ = enforce_causation_guard(text, [])
    assert hedged is True


@pytest.mark.parametrize(
    "text,phrase_name",
    [
        ("Region A is clearly responsible for the decline.", "responsible for"),
        ("Region A was the reason for the decline.", "reason for"),
        ("This resulted in a large drop.", "resulted in"),
        ("Region A drove the decline.", "drove"),
        ("Region A was driving the decline.", "driving"),
        ("This chart explains the drop.", "explains"),
        ("The report explained the drop.", "explained"),
        ("Region A is responsible for the decline.", "responsible for"),
        ("Region A accounts for the decline.", "accounts for"),
        ("Region A accounted for the decline.", "accounted for"),
        ("Poor weather brought about the decline.", "brought about"),
        ("The change produced a sharp drop.", "produced"),
        ("The change is producing a sharp drop.", "producing"),
        ("A supply shock triggered the decline.", "triggered"),
        ("A supply shock is triggering declines.", "triggering"),
        ("Region A caused the decline.", "caused"),
        ("Region A causes the decline.", "causes"),
        ("Region A is causing the decline.", "causing"),
        ("The decline was due to Region A.", "due to"),
        ("The decline happened because of Region A.", "because of"),
        ("Region A led to the decline.", "led to"),
        ("Region A leads to lower volume.", "leads to"),
    ],
)
def test_matched_text_reflects_the_causal_predicate(text, phrase_name):
    matched = find_causal_phrases(text)
    assert matched
    assert any(phrase_name in m.lower() or m.lower() in phrase_name for m in matched)


# =============================================================================
# NEW -- Layer 2: hedged causal language is a legitimate causal hypothesis, not
# a violation. Must be classified "causal_hypothesis" and left untouched by
# enforce_causation_guard, even with zero supporting hypotheses.
# =============================================================================


@pytest.mark.parametrize(
    "text",
    [
        "Region A may have caused the decline.",
        "Region A might have caused the decline.",
        "The pricing change could be responsible for the decline.",
        "This possibly caused the drop.",
        "The change likely drove the decline.",
        "The outage potentially triggered the drop.",
        "The data suggests Region A caused the decline.",
        "It appears to have caused the drop.",
        "It seems to have caused the drop.",
    ],
)
def test_hedged_causal_language_is_classified_as_causal_hypothesis(text):
    mentions = classify_relationship_language(text)
    causal_mentions = [m for m in mentions if m.category in ("causal_hypothesis", "causal_unhedged")]
    assert causal_mentions, f"expected some causal mention to be detected in: {text!r}"
    assert all(m.category == "causal_hypothesis" for m in causal_mentions), (
        f"expected hedged causal language to classify as causal_hypothesis, got: {causal_mentions!r}"
    )


@pytest.mark.parametrize(
    "text",
    [
        "Region A may have caused the decline.",
        "The pricing change could be responsible for the decline.",
        "The change likely drove the decline.",
    ],
)
def test_hedged_causal_language_is_never_rewritten_even_without_a_hypothesis(text):
    rewritten, hedged, matched = enforce_causation_guard(text, [])
    assert hedged is False
    assert rewritten == text
    assert matched == []


def test_hedged_causal_language_still_not_rewritten_with_hypotheses_present():
    # A hedged claim needs no evidence-gate excuse at all -- it was never a
    # violation in the first place, regardless of what hypotheses exist.
    hyps = [Hypothesis(id="h1", description="unrelated", is_causal=True, status="unsupported")]
    rewritten, hedged, matched = enforce_causation_guard("Region A may have caused the decline.", hyps)
    assert hedged is False
    assert rewritten == "Region A may have caused the decline."


# =============================================================================
# NEW -- Layer 3: correlation / association / prediction / temporal language is
# the desired, safe way to state a relationship and must never be flagged.
# =============================================================================


@pytest.mark.parametrize(
    "text,expected_category",
    [
        ("Region A is correlated with the revenue decline.", "correlation"),
        ("There is a correlation between Region A and the decline.", "correlation"),
        ("Region A is associated with the revenue decline.", "association"),
        ("Revenue is expected to decline next quarter.", "prediction"),
        ("The launch coincided with the revenue decline.", "temporal"),
        ("The decline occurred around the same time as the launch.", "temporal"),
    ],
)
def test_safe_relationship_language_is_classified_correctly_and_never_flagged(text, expected_category):
    mentions = classify_relationship_language(text)
    assert mentions, f"expected a relationship mention to be detected in: {text!r}"
    assert all(m.category == expected_category for m in mentions)
    assert find_causal_phrases(text) == []
    rewritten, hedged, matched = enforce_causation_guard(text, [])
    assert hedged is False
    assert rewritten == text
    assert matched == []


# =============================================================================
# NEW -- classify_relationship_language is independently importable/testable
# and returns properly-typed, span-accurate mentions.
# =============================================================================


def test_classify_relationship_language_returns_accurate_spans():
    text = "Region A is clearly responsible for the decline."
    mentions = classify_relationship_language(text)
    assert len(mentions) == 1
    m = mentions[0]
    assert text[m.start : m.end] == m.matched_text
    assert m.matched_text.lower() == "responsible for"
    assert m.category == "causal_unhedged"


def test_classify_relationship_language_handles_multiple_mentions_in_one_text():
    text = "Region A is associated with the decline, but marketing spend caused the drop."
    mentions = classify_relationship_language(text)
    categories = {m.category for m in mentions}
    assert "association" in categories
    assert "causal_unhedged" in categories
    # Only the causal_unhedged mention should be a violation.
    assert find_causal_phrases(text) == ["caused"]


def test_classify_relationship_language_empty_for_plain_text():
    assert classify_relationship_language("Revenue decreased by 5.2% this quarter.") == []


# =============================================================================
# NEW -- rewrite only touches the actual violation span, leaving hedged/safe
# language elsewhere in the same sentence untouched.
# =============================================================================


def test_rewrite_only_touches_the_unhedged_violation_not_hedged_language_nearby():
    text = "Region A is associated with seasonality, and marketing spend caused the decline."
    rewritten, hedged, matched = enforce_causation_guard(text, [])
    assert hedged is True
    assert "is associated with seasonality" in rewritten  # untouched
    assert "caused" not in rewritten.lower()


def test_rewrite_leaves_a_hedged_causal_clause_alone_while_fixing_an_unhedged_one():
    text = "Region A may have caused the early dip, but marketing spend is responsible for the decline."
    rewritten, hedged, matched = enforce_causation_guard(text, [])
    assert hedged is True
    assert "may have caused" in rewritten  # hedged clause left untouched
    assert "responsible for" not in rewritten.lower()  # unhedged clause rewritten
    assert matched == ["responsible for"]


# =============================================================================
# NEW -- the evidence gate still governs newly-added patterns, not just the
# original "caused"/"due to"/"led to" set.
# =============================================================================


def test_evidence_gate_also_covers_the_new_responsible_for_pattern():
    hyps = [Hypothesis(id="h1", description="Region A responsible for decline", is_causal=True, status="supported")]
    text = "Region A is clearly responsible for the decline."
    rewritten, hedged, matched = enforce_causation_guard(text, hyps)
    assert hedged is False
    assert rewritten == text


def test_evidence_gate_also_covers_the_new_triggered_pattern():
    # Uses "supported" (not "weakly_supported", tightened during Phase 4 integration
    # -- see test_weakly_supported_hypothesis_does_not_excuse_unhedged_causal_language)
    # to keep testing what this test is actually about: the new "triggered" pattern
    # is covered by the evidence gate at all, same as every other Layer 1 pattern.
    hyps = [Hypothesis(id="h1", description="Outage triggered decline", is_causal=True, status="supported")]
    text = "The outage triggered the revenue decline."
    rewritten, hedged, matched = enforce_causation_guard(text, hyps)
    assert hedged is False
    assert rewritten == text
