from __future__ import annotations

import httpx
import pytest

from app.agent.providers import LLMProviderError, OpenAICompatibleProvider, _friendly_error_message, _parse_wait_seconds


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
