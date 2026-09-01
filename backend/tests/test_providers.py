from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pandas as pd
import pytest

from app.agent.providers import (
    LLMProviderError,
    OpenAICompatibleProvider,
    _friendly_error_message,
    _parse_wait_seconds,
    build_provider_from_settings,
)
from app.config import get_settings


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2.347s", pytest.approx(2.347)),
        ("15m50.4s", pytest.approx(15 * 60 + 50.4)),
        ("5", pytest.approx(5.0)),
        (None, None),
        ("not-a-duration", None),
    ],
)
def test_parse_wait_seconds(value, expected):
    assert _parse_wait_seconds(value) == expected


def test_parse_retry_after_http_date_and_clamps_past_dates():
    now = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    assert _parse_wait_seconds("Tue, 01 Sep 2026 12:00:07 GMT", now=now) == 7
    assert _parse_wait_seconds("Tue, 01 Sep 2026 11:59:00 GMT", now=now) == 0


def _fake_http_error(status_code: int, error_message: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.example.com/chat/completions")
    response = httpx.Response(status_code, json={"error": {"message": error_message}}, request=request)
    return httpx.HTTPStatusError(f"{status_code} error", request=request, response=response)


# The real Groq 413 body that leaked to a user before this fix — used as the exact
# reproduction case for the sanitization requirement.
_REAL_LEAKED_DETAIL = (
    "Request too large for model `openai/gpt-oss-120b` in organization "
    "`org_01m08cbcj0e9t9h2mc38mdwm4c` service tier `on_demand` on tokens per minute "
    "(TPM): Limit 8000, Requested 21186, please reduce your message size and try "
    "again. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing"
)


@pytest.mark.parametrize(
    "status_code,leaked_substrings",
    [
        (413, ["org_01m08cbcj0e9t9h2mc38mdwm4c", "TPM", "billing", "21186", "console.groq.com"]),
        (429, ["org_", "TPM"]),
        (401, ["org_"]),
        (500, ["org_"]),
        (418, ["org_"]),
    ],
)
def test_friendly_error_message_never_leaks_provider_internals(status_code, leaked_substrings):
    exc = _fake_http_error(status_code, _REAL_LEAKED_DETAIL)
    message = _friendly_error_message(exc)
    for substring in leaked_substrings:
        assert substring not in message
    # Must still be a real, non-empty, user-readable sentence — not blank.
    assert len(message) > 10


def test_friendly_error_message_413_is_actionable():
    exc = _fake_http_error(413, _REAL_LEAKED_DETAIL)
    message = _friendly_error_message(exc)
    assert "specific" in message.lower() or "narrow" in message.lower()


def test_network_error_message_does_not_leak_connection_details(monkeypatch):
    provider = OpenAICompatibleProvider(api_key="fake-key", base_url="https://api.example.com", model="test-model")

    def _raise_connect_error(*args, **kwargs):
        raise httpx.ConnectError("Connection refused to internal-host-10.0.0.5:8443")

    monkeypatch.setattr(provider._client, "post", _raise_connect_error)

    with pytest.raises(LLMProviderError) as exc_info:
        provider.complete([{"role": "user", "content": "hi"}], [])

    assert "10.0.0.5" not in str(exc_info.value)
    assert "internal-host" not in str(exc_info.value)


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_status_honors_retry_after_then_succeeds(monkeypatch, status):
    provider = OpenAICompatibleProvider(
        api_key="fake-key", base_url="https://api.example.com", model="test-model"
    )
    request = httpx.Request("POST", "https://api.example.com/chat/completions")
    responses = iter(
        [
            httpx.Response(status, headers={"Retry-After": "0"}, request=request),
            httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}, request=request),
        ]
    )
    waits: list[float] = []
    monkeypatch.setattr(provider._client, "post", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr("app.agent.providers.time.sleep", waits.append)

    result = provider.complete([{"role": "user", "content": "hello"}], [])

    assert result.content == "ok"
    assert waits == [0.0]


def test_retry_after_is_bounded(monkeypatch):
    provider = OpenAICompatibleProvider(
        api_key="fake-key", base_url="https://api.example.com", model="test-model"
    )
    request = httpx.Request("POST", "https://api.example.com/chat/completions")
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "3600"}, request=request),
            httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}, request=request),
        ]
    )
    waits: list[float] = []
    monkeypatch.setattr(provider._client, "post", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr("app.agent.providers.time.sleep", waits.append)

    provider.complete([{"role": "user", "content": "hello"}], [])

    assert waits == [20.0]


def test_non_transient_http_error_is_not_retried(monkeypatch):
    provider = OpenAICompatibleProvider(
        api_key="fake-key", base_url="https://api.example.com", model="test-model"
    )
    request = httpx.Request("POST", "https://api.example.com/chat/completions")
    response = httpx.Response(413, json={"error": {"message": "too large"}}, request=request)
    calls = 0

    def post_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        return response

    monkeypatch.setattr(provider._client, "post", post_once)
    with pytest.raises(LLMProviderError):
        provider.complete([{"role": "user", "content": "hello"}], [])
    assert calls == 1


def test_optional_service_tier_is_forwarded_only_when_configured(monkeypatch):
    captured: list[dict] = []

    def successful_post(*args, **kwargs):
        captured.append(kwargs["json"])
        request = httpx.Request("POST", "https://api.example.com/chat/completions")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}, request=request)

    default_provider = OpenAICompatibleProvider(
        api_key="fake-key", base_url="https://api.example.com", model="test-model"
    )
    paid_provider = OpenAICompatibleProvider(
        api_key="fake-key",
        base_url="https://api.example.com",
        model="test-model",
        service_tier="priority",
    )
    monkeypatch.setattr(default_provider._client, "post", successful_post)
    monkeypatch.setattr(paid_provider._client, "post", successful_post)

    default_provider.complete([{"role": "user", "content": "hello"}], [])
    paid_provider.complete([{"role": "user", "content": "hello"}], [])

    assert "service_tier" not in captured[0]
    assert captured[1]["service_tier"] == "priority"


def test_settings_timeout_reaches_the_http_client(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "fake-key")
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT", "17.5")
    monkeypatch.setenv("LLM_EGRESS_MODE", "external_redacted")
    get_settings.cache_clear()
    try:
        provider = build_provider_from_settings(pd.DataFrame({"metric": [1, 2]}))
        assert provider._delegate._client.timeout.connect == 17.5
        assert provider._delegate._client.timeout.read == 17.5
    finally:
        get_settings.cache_clear()
