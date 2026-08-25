"""Shared helper for the reasoning layer's three bounded, structured-output LLM calls
(question_parser, planner, synthesizer). Not a public module -- internal to
app.reasoning.

Every structured call goes through `complete_json`: request JSON-only output
(`tools=[]`, so this is never a tool-calling completion), parse it, and on failure
retry exactly once with an error-correction nudge before giving up. This bounds each
of the 3 reasoning stages to at most 2 provider calls (1 + 1 retry) -- resilience, not
an open-ended loop -- and callers must treat a `None` return as "degrade gracefully,"
never as license to retry further themselves.
"""

from __future__ import annotations

import json

from app.agent.providers import LLMProvider


def complete_json(provider: LLMProvider, messages: list[dict]) -> dict | None:
    response = provider.complete(messages, [])
    parsed = _try_parse(response.content)
    if parsed is not None:
        return parsed
    retry_messages = messages + [
        {
            "role": "system",
            "content": "Your previous response was not valid JSON in the required shape. Reply with JSON only -- no prose, no markdown fences.",
        }
    ]
    response = provider.complete(retry_messages, [])
    return _try_parse(response.content)


def _try_parse(content: str | None) -> dict | None:
    if not content:
        return None
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        result = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return result if isinstance(result, dict) else None
