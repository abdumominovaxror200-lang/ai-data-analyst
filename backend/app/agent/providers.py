from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx
import pandas as pd

from app.config import get_settings

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2
_MAX_BACKOFF_SECONDS = 20.0
_DURATION_RE = re.compile(r"(?:(?P<minutes>\d+)m)?(?:(?P<seconds>\d+(?:\.\d+)?)s)?")


class LLMProviderError(Exception):
    """Raised when the upstream LLM API cannot fulfill a request (rate limit, outage,
    network failure, etc.) — distinct from ToolExecutionError, which is our own code."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ProviderResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ProviderResponse: ...


def _parse_wait_seconds(value: str | None) -> float | None:
    """Parses a Retry-After-style duration. Handles plain seconds ('2.3') and the
    '15m50.4s' / '2.3s' style used by Groq's x-ratelimit-reset-* headers."""
    if not value:
        return None
    match = _DURATION_RE.fullmatch(value.strip())
    if not match or not (match.group("minutes") or match.group("seconds")):
        try:
            return float(value)
        except ValueError:
            return None
    total = 0.0
    if match.group("minutes"):
        total += int(match.group("minutes")) * 60
    if match.group("seconds"):
        total += float(match.group("seconds"))
    return total


class OpenAICompatibleProvider(LLMProvider):
    """Talks to any OpenAI-compatible /chat/completions endpoint.

    This is what makes the LLM provider swappable: OpenAI, OpenRouter, Groq, DeepSeek
    (and most other hosted LLM APIs) all expose an OpenAI-compatible chat-completions
    endpoint, so switching providers is a base_url/model/api_key change in .env — no
    code change required.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        reasoning_effort: str = "",
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("LLM_API_KEY is not configured.")
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ProviderResponse:
        payload: dict[str, Any] = {"model": self._model, "messages": messages}
        if tools:
            # Omit "tools"/"tool_choice" entirely when empty rather than sending an
            # empty array — used to force a final natural-language answer (see the
            # agent's stagnant-loop stop condition) without the model reaching for
            # another tool call.
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort

        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self._client.post("/chat/completions", json=payload)
            except httpx.RequestError as exc:
                # Log the real network detail server-side only — a raw exception string
                # can include hostnames/ports that shouldn't reach the end user.
                logger.warning("LLM provider network error (details redacted)")
                raise LLMProviderError(_FRIENDLY_MESSAGES["network"]) from exc

            if response.status_code == 429 and attempt < _MAX_RETRIES:
                # Free-tier rate limits (e.g. Groq's per-minute token budget) are often
                # short-lived — back off using the provider's own reset hint and retry
                # rather than failing the whole multi-tool-call conversation outright.
                wait = (
                    _parse_wait_seconds(response.headers.get("retry-after"))
                    or _parse_wait_seconds(response.headers.get("x-ratelimit-reset-tokens"))
                    or 5.0
                )
                wait = min(wait, _MAX_BACKOFF_SECONDS)
                logger.warning("LLM provider rate limited (attempt %d/%d), retrying in %.1fs", attempt + 1, _MAX_RETRIES, wait)
                time.sleep(wait)
                continue

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise LLMProviderError(_friendly_error_message(exc)) from exc

            try:
                message = response.json()["choices"][0]["message"]
            except (ValueError, KeyError, IndexError) as exc:
                # A malformed/unexpected response body must never crash the request —
                # log the raw body server-side, tell the user something generic.
                logger.error("LLM provider returned an unparseable response (body redacted)")
                raise LLMProviderError(_FRIENDLY_MESSAGES["default"]) from exc

            tool_calls = []
            for call in message.get("tool_calls") or []:
                try:
                    arguments = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append(ToolCall(id=call["id"], name=call["function"]["name"], arguments=arguments))
            return ProviderResponse(content=message.get("content"), tool_calls=tool_calls)

        raise LLMProviderError(_FRIENDLY_MESSAGES["rate_limit"])


# User-facing messages only — deliberately generic. The real provider detail (org ids,
# token-budget numbers, billing links, raw exception text) is always logged server-side
# via `logger.warning`/`logger.error` instead, never returned in an API response.
_FRIENDLY_MESSAGES = {
    "rate_limit": "The AI is receiving a lot of requests right now. Please try again in a moment.",
    "too_large": "That question needed more data than the AI can process in one go. Try asking something more specific or narrower.",
    "auth": "The AI provider rejected the request. Please contact the site administrator.",
    "unavailable": "The AI provider is temporarily unavailable. Please try again shortly.",
    "network": "Could not reach the AI provider. Please try again shortly.",
    "default": "The AI couldn't complete this request. Please try again or rephrase your question.",
}


def _friendly_error_message(exc: httpx.HTTPStatusError) -> str:
    status = exc.response.status_code
    logger.warning("LLM provider HTTP error %s (detail redacted)", status)

    if status == 429:
        return _FRIENDLY_MESSAGES["rate_limit"]
    if status == 413:
        return _FRIENDLY_MESSAGES["too_large"]
    if status in (401, 403):
        return _FRIENDLY_MESSAGES["auth"]
    if status >= 500:
        return _FRIENDLY_MESSAGES["unavailable"]
    return _FRIENDLY_MESSAGES["default"]


class MockProvider(LLMProvider):
    """Deterministic scripted provider used in tests — no network calls, no API key."""

    def __init__(self, script: list[ProviderResponse]) -> None:
        self._script = list(script)
        self.calls: list[list[dict[str, Any]]] = []
        self.tools_per_call: list[list[dict[str, Any]]] = []

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ProviderResponse:
        self.calls.append(messages)
        self.tools_per_call.append(tools)
        if not self._script:
            return ProviderResponse(content="No more scripted responses.")
        return self._script.pop(0)


def build_provider_from_settings(dataset: pd.DataFrame | None = None) -> LLMProvider:
    settings = get_settings()
    # Lazy import avoids a providers/privacy import cycle while keeping this the
    # single construction point for every production provider call.
    from app.security.privacy import (
        DisabledProvider,
        EgressMode,
        PrivacyEnforcingProvider,
        classify_dataset,
        validate_local_endpoint,
    )

    try:
        mode = EgressMode(settings.llm_egress_mode)
    except ValueError as exc:
        raise ValueError("LLM_EGRESS_MODE must be local_only, external_redacted, or llm_disabled.") from exc
    if mode == EgressMode.LLM_DISABLED:
        return DisabledProvider()
    if mode == EgressMode.LOCAL_ONLY:
        validate_local_endpoint(settings.llm_base_url)
    if mode == EgressMode.EXTERNAL_REDACTED and dataset is None:
        raise ValueError("external_redacted requires dataset context for fail-closed sanitization.")
    delegate = OpenAICompatibleProvider(
        api_key=settings.llm_api_key or ("local" if mode == EgressMode.LOCAL_ONLY else ""),
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        reasoning_effort=settings.llm_reasoning_effort,
    )
    profile = classify_dataset(dataset) if dataset is not None else None
    return PrivacyEnforcingProvider(delegate, mode, profile)
