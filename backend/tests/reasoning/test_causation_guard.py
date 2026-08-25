from __future__ import annotations

from app.reasoning.causation_guard import enforce_causation_guard, find_causal_phrases
from app.reasoning.contracts import Hypothesis


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


def test_causal_language_is_permitted_for_weakly_supported_hypothesis_too():
    hyps = [Hypothesis(id="h1", description="Region A caused decline", is_causal=True, status="weakly_supported")]
    rewritten, hedged, _ = enforce_causation_guard("This was due to Region A.", hyps)
    assert hedged is False


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
