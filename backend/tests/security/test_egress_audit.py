from __future__ import annotations

import json
import logging
import re

import pytest

from app.agent.providers import LLMProviderError
from app.security.privacy import (
    DisabledProvider,
    EgressMode,
    PrivacyEnforcingProvider,
    classify_dataset,
)
from tests.security.test_privacy_data_plane import RecordingProvider, _sensitive_frame


LOGGER_NAME = "app.security.llm_egress"
TOOLS = [{"type": "function", "function": {"name": "describe_data"}}]


def _events(caplog) -> list[dict]:
    return [json.loads(record.getMessage()) for record in caplog.records if record.name == LOGGER_NAME]


def _assert_safe_event(event: dict) -> None:
    serialized = json.dumps(event)
    assert event["event"] == "llm_egress"
    assert event["tool_names"] == ["describe_data"]
    assert re.fullmatch(r"[0-9a-f]{64}", event["payload_sha256"])
    assert "alice@example.test" not in serialized
    assert "+1 (202) 555-0198" not in serialized
    assert "customer_email" not in serialized


def test_external_redacted_emits_one_safe_structured_audit_record(caplog) -> None:
    frame = _sensitive_frame()
    provider = PrivacyEnforcingProvider(
        RecordingProvider(),
        EgressMode.EXTERNAL_REDACTED,
        classify_dataset(frame),
        model="external-model",
    )
    messages = [
        {"role": "user", "content": "Analyze customer_email for alice@example.test"},
        {
            "role": "tool",
            "name": "filter_data",
            "content": "UNTRUSTED_TOOL_OUTPUT\n{\"customer_email\":\"alice@example.test\"}",
        },
    ]
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        provider.complete(messages, TOOLS)
    events = _events(caplog)
    assert len(events) == 1
    event = events[0]
    _assert_safe_event(event)
    assert event["mode"] == "external_redacted"
    assert event["model"] == "external-model"
    assert event["message_count"] == 2
    assert event["redactions"]["aliased_columns"] >= 1
    assert event["redactions"]["quarantined_hits"] >= 1
    assert event["redactions"]["withheld_tool_payloads"] == 1
    assert event["call_made"] is True


def test_local_only_emits_one_hash_only_audit_record(caplog) -> None:
    frame = _sensitive_frame()
    provider = PrivacyEnforcingProvider(
        RecordingProvider(), EgressMode.LOCAL_ONLY, classify_dataset(frame), model="local-model"
    )
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        provider.complete([{"role": "user", "content": "alice@example.test"}], TOOLS)
    events = _events(caplog)
    assert len(events) == 1
    _assert_safe_event(events[0])
    assert events[0]["mode"] == "local_only"
    assert events[0]["call_made"] is True


def test_llm_disabled_emits_one_no_call_audit_record(caplog) -> None:
    provider = DisabledProvider(model="disabled-model")
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        with pytest.raises(LLMProviderError, match="disabled"):
            provider.complete([{"role": "user", "content": "alice@example.test"}], TOOLS)
    events = _events(caplog)
    assert len(events) == 1
    _assert_safe_event(events[0])
    assert events[0]["mode"] == "llm_disabled"
    assert events[0]["call_made"] is False
