from __future__ import annotations

"""Independent adversarial verification of this session's error-sanitization fixes.

`test_providers.py` already covers: the exact real leaked-string reproduction case,
429/413/401/500/418 status mapping, and one network-error (ConnectError) case. This
file tries to break the SAME code paths (`app/agent/providers.py::_friendly_error_message`
and the global handler in `app/main.py`) with adversarial inputs that file does not
exercise: malformed/unexpected error-body shapes, other RequestError subtypes, and a
full HTTP-response-level (not just exception-string-level) check that nothing leaks
through the FastAPI route + global exception handler.

Every test here was actually run against the real code, not just read and trusted.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

from app.agent.providers import (
    LLMProviderError,
    MockProvider,
    OpenAICompatibleProvider,
    ProviderResponse,
    _friendly_error_message,
)

_SENSITIVE = [
    "org_01m08cbcj0e9t9h2mc38mdwm4c",
    "internal-host-10.0.0.5",
    "TPM",
    "console.groq.com",
    "/etc/shadow",
    "sk-secret-key-12345",
]


def _fake_http_error(status_code: int, body) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.example.com/chat/completions")
    if isinstance(body, (dict, list)):
        response = httpx.Response(status_code, json=body, request=request)
    else:
        response = httpx.Response(status_code, content=body, request=request)
    return httpx.HTTPStatusError(f"{status_code} error", request=request, response=response)


# ---------------------------------------------------------------------------
# Malformed / unexpected error-body shapes not covered by test_providers.py
# ---------------------------------------------------------------------------


def test_error_field_is_a_string_not_a_dict_does_not_crash_or_leak():
    """A misbehaving/compromised upstream could send {"error": "raw string"} instead
    of the expected {"error": {"message": ...}}. `.get("error", {}).get("message")`
    would raise AttributeError on a plain string -- confirm the broad except still
    catches it and still returns a clean, generic message."""
    exc = _fake_http_error(500, {"error": "org_secret_leaked_directly_as_string sk-secret-key-12345"})
    message = _friendly_error_message(exc)
    for s in _SENSITIVE:
        assert s not in message
    assert len(message) > 10


def test_error_key_entirely_missing_falls_back_gracefully():
    exc = _fake_http_error(413, {"message": "org_hidden_elsewhere sk-secret-key-12345", "code": "too_big"})
    message = _friendly_error_message(exc)
    for s in _SENSITIVE:
        assert s not in message


def test_error_body_is_a_json_array_not_an_object():
    exc = _fake_http_error(500, [{"error": "sk-secret-key-12345"}])
    message = _friendly_error_message(exc)
    for s in _SENSITIVE:
        assert s not in message


def test_non_json_body_with_sensitive_content_never_returned_to_client():
    """Some failure modes (proxy error pages, WAF blocks) return HTML/plain text, not
    JSON. The fallback path reads response.text for logging only -- verify it never
    ends up in the user-facing message."""
    html_body = b"<html><body>Internal error: org_01m08cbcj0e9t9h2mc38mdwm4c leaked host internal-host-10.0.0.5</body></html>"
    exc = _fake_http_error(500, html_body)
    message = _friendly_error_message(exc)
    for s in _SENSITIVE:
        assert s not in message


def test_empty_response_body_does_not_crash():
    exc = _fake_http_error(503, b"")
    message = _friendly_error_message(exc)
    assert len(message) > 10


@pytest.mark.parametrize("status_code", [402, 407, 422, 451, 499])
def test_unmapped_status_codes_fall_back_to_generic_default(status_code):
    """Only 429/413/401/403/>=500 are explicitly mapped. Anything else must still be
    generic and non-empty, never the raw upstream detail."""
    exc = _fake_http_error(status_code, {"error": {"message": "org_01m08cbcj0e9t9h2mc38mdwm4c sk-secret-key-12345"}})
    message = _friendly_error_message(exc)
    for s in _SENSITIVE:
        assert s not in message
    assert len(message) > 10


# ---------------------------------------------------------------------------
# Network-error subtypes beyond the single ConnectError case already tested
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: httpx.ConnectTimeout("Connect timeout to internal-host-10.0.0.5:8443"),
        lambda: httpx.ReadTimeout("Read timeout on internal-host-10.0.0.5"),
        lambda: httpx.WriteTimeout("Write timeout org_01m08cbcj0e9t9h2mc38mdwm4c"),
        lambda: httpx.PoolTimeout("Pool exhausted for internal-host-10.0.0.5"),
        lambda: httpx.ProtocolError("Bad protocol frame from internal-host-10.0.0.5"),
        lambda: httpx.RemoteProtocolError("Server disconnected: internal-host-10.0.0.5"),
    ],
)
def test_all_httpx_request_error_subtypes_are_sanitized(monkeypatch, exc_factory):
    """`OpenAICompatibleProvider.complete` catches the base `httpx.RequestError`, but
    verify every concrete subtype we could realistically hit (timeouts, protocol
    errors -- not just ConnectError) is actually caught by that `except` clause and
    produces the sanitized network message, not the raw exception text."""
    provider = OpenAICompatibleProvider(api_key="fake-key", base_url="https://api.example.com", model="test-model")

    def _raise(*args, **kwargs):
        raise exc_factory()

    monkeypatch.setattr(provider._client, "post", _raise)

    with pytest.raises(LLMProviderError) as exc_info:
        provider.complete([{"role": "user", "content": "hi"}], [])

    message = str(exc_info.value)
    assert "10.0.0.5" not in message
    assert "org_" not in message
    assert "8443" not in message


def test_malformed_response_body_missing_choices_never_leaks_body_content():
    """response.json() succeeds (200 OK) but the shape is unexpected (missing
    'choices') -- this hits the KeyError/IndexError path, which must log the raw body
    server-side but return only the generic default message."""
    provider = OpenAICompatibleProvider(api_key="fake-key", base_url="https://api.example.com", model="test-model")

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"unexpected": "shape", "leaked_detail": "org_01m08cbcj0e9t9h2mc38mdwm4c sk-secret-key-12345"}

        text = '{"unexpected": "shape", "leaked_detail": "org_01m08cbcj0e9t9h2mc38mdwm4c sk-secret-key-12345"}'

    monkeypatch_target = provider._client
    monkeypatch_target.post = lambda *a, **k: _FakeResponse()

    with pytest.raises(LLMProviderError) as exc_info:
        provider.complete([{"role": "user", "content": "hi"}], [])

    message = str(exc_info.value)
    for s in _SENSITIVE:
        assert s not in message


# ---------------------------------------------------------------------------
# Full-stack (HTTP response body) checks -- not just the exception string
# ---------------------------------------------------------------------------


def test_chat_endpoint_http_response_body_never_leaks_provider_detail(client: TestClient, monkeypatch):
    """End-to-end: a real 413 from the upstream provider must not leak through
    agent.ask -> routes_chat.chat -> HTTPException -> FastAPI's JSON response body.
    Exercises the full stack, not just the provider unit in isolation."""
    from app.datasets.storage import DatasetRecord
    import pandas as pd

    import app.api.routes_chat as routes_chat_module

    df = pd.DataFrame({"a": [1, 2, 3]})
    record = DatasetRecord(
        id="did-1",
        original_filename="x.csv",
        extension=".csv",
        uploaded_at=pd.Timestamp.utcnow(),
        df=df,
        stored_path="unused",
    )

    class _FakeStore:
        def get(self, dataset_id):
            return record

    monkeypatch.setattr(routes_chat_module, "get_dataset_store", lambda: _FakeStore())

    real_leaked_detail = (
        "Request too large for model `openai/gpt-oss-120b` in organization "
        "`org_01m08cbcj0e9t9h2mc38mdwm4c` on tokens per minute (TPM): Limit 8000."
    )

    def _return_413(*args, **kwargs):
        # A real httpx client never raises HTTPStatusError from `.post()` itself --
        # that only happens when the *caller* invokes `response.raise_for_status()`,
        # which is exactly what `OpenAICompatibleProvider.complete` does. Returning the
        # Response object here (status 413, not yet raised) matches real httpx
        # semantics instead of short-circuiting past the code path under test.
        request = httpx.Request("POST", "https://api.example.com/chat/completions")
        return httpx.Response(413, json={"error": {"message": real_leaked_detail}}, request=request)

    real_provider = OpenAICompatibleProvider(api_key="fake-key", base_url="https://api.example.com", model="m")
    monkeypatch.setattr(real_provider._client, "post", _return_413)
    monkeypatch.setattr(routes_chat_module, "build_provider_from_settings", lambda: real_provider)

    response = client.post("/api/chat", json={"dataset_id": "did-1", "message": "hi", "history": []})

    assert response.status_code == 503
    body_text = response.text
    assert "org_01m08cbcj0e9t9h2mc38mdwm4c" not in body_text
    assert "TPM" not in body_text
    assert "8000" not in body_text


def test_global_exception_handler_hides_unexpected_error_detail(monkeypatch):
    """An unanticipated bug (e.g. a RuntimeError deep in the dataset store, not one of
    the handled exception types) must still produce a clean generic 500 body -- proving
    the global handler in app/main.py, not just the provider-specific paths, holds.

    NOTE on `raise_server_exceptions=False`: Starlette's `ServerErrorMiddleware` sends
    the registered handler's response to the client and THEN re-raises the original
    exception (by design, so `uvicorn`/servers can still log it -- see
    `starlette/middleware/errors.py::ServerErrorMiddleware.__call__`, the trailing
    `raise exc` after `await response(scope, receive, send)`). `TestClient`'s default
    `raise_server_exceptions=True` surfaces that re-raise as a Python exception in the
    test instead of the response a real HTTP client would receive. That is a TestClient
    debugging aid, not the client-visible behavior we're verifying here, so this test
    disables it to see exactly what a real client gets on the wire."""
    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)
    import app.api.routes_datasets as routes_datasets_module

    class _ExplodingStore:
        def get(self, dataset_id):
            raise RuntimeError("db connection string: postgres://admin:sk-secret-key-12345@internal-host-10.0.0.5/prod")

    monkeypatch.setattr(routes_datasets_module, "get_dataset_store", lambda: _ExplodingStore())

    response = client.get("/api/datasets/some-id")

    assert response.status_code == 500
    body = response.json()
    assert body == {"detail": "An unexpected error occurred. Please try again."}
    for s in _SENSITIVE:
        assert s not in response.text


def test_global_exception_handler_hides_detail_on_upload_path(monkeypatch):
    """Same check on the upload path, which handles user-supplied file bytes -- the
    highest-risk entry point for triggering an unanticipated parser exception. See the
    `raise_server_exceptions=False` note on the sibling test above."""
    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)
    import app.api.routes_datasets as routes_datasets_module

    class _ExplodingStore:
        def save(self, filename, content):
            raise RuntimeError("unexpected parser crash near /etc/shadow with key sk-secret-key-12345")

    monkeypatch.setattr(routes_datasets_module, "get_dataset_store", lambda: _ExplodingStore())

    files = {"file": ("data.csv", b"a,b\n1,2\n", "text/csv")}
    response = client.post("/api/datasets/upload", files=files)

    assert response.status_code == 500
    for s in _SENSITIVE:
        assert s not in response.text
