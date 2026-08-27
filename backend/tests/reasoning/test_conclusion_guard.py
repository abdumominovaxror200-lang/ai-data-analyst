"""Unit tests for app.reasoning.conclusion_guard -- the deterministic caveat-prepend
safety net for `blocks_conclusion`-severity limitations (final professional-analyst
stress-test mission, Phase 4/6).

Built after auditing the hard real-world benchmark's honesty-vs-overclaiming
adversarial pairs found a real gap: `recommendation_grounding.py`'s blocks_conclusion
override already prevents a structured `Recommendation.confidence` from surviving,
but the free-text `final_answer_text` a user actually reads was never touched --
an overclaiming answer could read as flatly confident even when a severe,
correctly-detected problem existed. See tests/test_reasoning_integration.py and
tests/test_blocks_conclusion_enforcement.py for the end-to-end (real orchestrator)
proof this reaches the final answer in practice; these are the isolated unit tests
for the guard function itself.
"""

from __future__ import annotations

from app.reasoning.conclusion_guard import enforce_conclusion_guard
from app.reasoning.contracts import Limitation


def _limitation(severity: str, text: str = "something is wrong") -> Limitation:
    return Limitation(category="methodological", text=text, severity=severity)


def test_no_limitations_leaves_text_unchanged():
    text = "North outperforms South by $55."
    result, added = enforce_conclusion_guard(text, [])
    assert result == text
    assert added is False


def test_only_reduces_confidence_limitations_leave_text_unchanged():
    text = "North outperforms South by $55."
    result, added = enforce_conclusion_guard(text, [_limitation("reduces_confidence")])
    assert result == text
    assert added is False


def test_only_minor_limitations_leave_text_unchanged():
    text = "North outperforms South by $55."
    result, added = enforce_conclusion_guard(text, [_limitation("minor")])
    assert result == text
    assert added is False


def test_a_blocks_conclusion_limitation_prepends_a_caveat():
    text = "North outperforms South by $55."
    result, added = enforce_conclusion_guard(text, [_limitation("blocks_conclusion", "severe confound")])
    assert added is True
    assert "Important caveat:" in result
    assert "severe confound" in result
    # the model's own text is preserved verbatim, not rewritten/truncated
    assert result.endswith(text)


def test_a_mix_of_severities_only_reports_the_blocking_ones_in_the_caveat():
    limitations = [
        _limitation("reduces_confidence", "a minor sample-size note"),
        _limitation("blocks_conclusion", "the fatal problem"),
    ]
    result, added = enforce_conclusion_guard("Some answer.", limitations)
    assert added is True
    assert "the fatal problem" in result
    assert "a minor sample-size note" not in result


def test_multiple_blocking_limitations_are_all_included():
    limitations = [
        _limitation("blocks_conclusion", "first fatal issue"),
        _limitation("blocks_conclusion", "second fatal issue"),
    ]
    result, added = enforce_conclusion_guard("Some answer.", limitations)
    assert "first fatal issue" in result
    assert "second fatal issue" in result


def test_is_idempotent_against_being_applied_twice():
    """Relevant because synthesize()'s early-stop paths and its main path both
    funnel through the same guard -- applying it twice to already-caveated text
    must not double the caveat."""
    once, _ = enforce_conclusion_guard("Some answer.", [_limitation("blocks_conclusion")])
    twice, added_again = enforce_conclusion_guard(once, [_limitation("blocks_conclusion")])
    assert twice == once
    assert added_again is False


def test_empty_text_is_returned_unchanged_even_with_blocking_limitations():
    result, added = enforce_conclusion_guard("", [_limitation("blocks_conclusion")])
    assert result == ""
    assert added is False


def test_none_limitations_argument_is_safe():
    text = "North outperforms South by $55."
    result, added = enforce_conclusion_guard(text, None)
    assert result == text
    assert added is False
