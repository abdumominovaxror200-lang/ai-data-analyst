from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

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
        payload = {"model": self._model, "messages": messages, "tools": tools, "tool_choice": "auto"}
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort

        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self._client.post("/chat/completions", json=payload)
            except httpx.RequestError as exc:
                raise LLMProviderError(f"Could not reach the LLM provider: {exc}") from exc

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
                raise LLMProviderError(_describe_http_error(exc)) from exc

            message = response.json()["choices"][0]["message"]
            tool_calls = []
            for call in message.get("tool_calls") or []:
                try:
                    arguments = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append(ToolCall(id=call["id"], name=call["function"]["name"], arguments=arguments))
            return ProviderResponse(content=message.get("content"), tool_calls=tool_calls)

        raise LLMProviderError("LLM provider is rate-limiting requests. Please try again shortly.")


def _describe_http_error(exc: httpx.HTTPStatusError) -> str:
    status = exc.response.status_code
    if status == 429:
        return "LLM provider is rate-limiting requests. Please try again shortly."
    try:
        detail = exc.response.json().get("error", {}).get("message")
    except Exception:  # noqa: BLE001 - response body may not be JSON
        detail = None
    return f"LLM provider request failed ({status}){': ' + detail if detail else '.'}"


class MockProvider(LLMProvider):
    """Deterministic scripted provider used in tests — no network calls, no API key."""

    def __init__(self, script: list[ProviderResponse]) -> None:
        self._script = list(script)
        self.calls: list[list[dict[str, Any]]] = []

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ProviderResponse:
        self.calls.append(messages)
        if not self._script:
            return ProviderResponse(content="No more scripted responses.")
        return self._script.pop(0)


def build_provider_from_settings() -> LLMProvider:
    settings = get_settings()
    return OpenAICompatibleProvider(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        reasoning_effort=settings.llm_reasoning_effort,
    )
