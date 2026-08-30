"""Independent tests for blocked-conclusion synthesis safety."""
from __future__ import annotations

import json

from app.agent.providers import MockProvider, ProviderResponse
from app.reasoning.conclusion_guard import blocked_narrative_violations, enforce_conclusion_guard
from app.reasoning.contracts import AnalyticalQuestion, Finding, Limitation
from app.reasoning.synthesizer import synthesize


def _blocker() -> Limitation:
    return Limitation(category="insufficient_coverage", text="A required comparison is unavailable.",
                      severity="blocks_conclusion")


def _finding() -> Finding:
    return Finding(id="finding_1", statement="The measured total is lower in the later period.",
                   classification="CALCULATED_RESULT", supporting_evidence=["ev_1"])


def test_validator_detects_driver_and_recommendation_language():
    text = "The primary driver is channel mix, so leaders should prioritize retention."
    assert blocked_narrative_violations(text) == ["definitive_driver", "recommendation_language"]


def test_blocked_invalid_narrative_is_replaced_not_merely_caveated():
    unsafe = "The dominant cause is pricing. Consider changing the offer immediately."
    result, changed = enforce_conclusion_guard(unsafe, [_blocker()], [_finding()])
    assert changed is True
    assert unsafe not in result
    assert "The measured total is lower" in result
    assert "A required comparison is unavailable" in result
    assert "Additional evidence is required" in result
    assert blocked_narrative_violations(result) == []


def test_blocked_synthesis_prompt_gates_output_and_discards_recommendation():
    provider = MockProvider([ProviderResponse(content=json.dumps({
        "final_answer_text": "The root cause is pricing and the team should change it.",
        "recommendation": {"recommendation": "Change pricing", "confidence": "high"},
    }))])
    text, recommendation, _hedged, _matched = synthesize(
        provider,
        AnalyticalQuestion(original_question="Explain the change", intent="diagnostic"),
        [], None, [], [_finding()], [], [_blocker()],
    )
    system_text = "\n".join(message["content"] for message in provider.calls[0] if message["role"] == "system")
    assert "CONCLUSION STATUS: BLOCKED" in system_text
    assert recommendation is None
    assert "root cause" not in text.lower()
    assert "should" not in text.lower()
    assert "Additional evidence is required" in text


def test_safe_next_analysis_language_is_allowed_when_blocked():
    text = "The measured total declined. Additional evidence is required before identifying a driver."
    result, changed = enforce_conclusion_guard(text, [_blocker()], [_finding()])
    assert result.endswith(text)
    assert changed is True  # the blocker caveat was added, but the safe text survived


def test_supported_flow_remains_byte_for_byte_unchanged():
    text = "The primary driver is supported, and the team should prioritize the response."
    result, changed = enforce_conclusion_guard(text, [], [_finding()])
    assert result == text
    assert changed is False


def test_supported_synthesis_keeps_normal_conclusion_and_recommendation():
    provider = MockProvider([ProviderResponse(content=json.dumps({
        "final_answer_text": "The measured difference is supported by complete evidence.",
        "recommendation": {"recommendation": "Run the validated intervention", "confidence": "medium"},
    }))])
    text, recommendation, _hedged, _matched = synthesize(
        provider,
        AnalyticalQuestion(original_question="Compare periods", intent="comparative"),
        [], None, [], [_finding()], [], [],
    )
    assert text == "The measured difference is supported by complete evidence."
    assert recommendation is not None
    assert recommendation.recommendation == "Run the validated intervention"
    assert recommendation.confidence == "medium"
    assert "CONCLUSION STATUS: BLOCKED" not in "\n".join(
        message["content"] for message in provider.calls[0] if message["role"] == "system"
    )
