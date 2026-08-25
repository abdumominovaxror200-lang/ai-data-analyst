"""LLM reasoning call 1 of 3: question parsing + claim extraction (Phase 3B.2).

Structured-output only (no tool-calling here — `tools=[]`), so the model's response
space is a fixed JSON shape, not free-form chat. On a malformed response, retries once
with an error-correction nudge, then degrades gracefully to a conservative default
rather than raising — the pipeline must never crash because one structured-output call
came back malformed (see the reasoning-layer failure-mode plan in
`.agent/reasoning-layer-design.md` §7, still the governing rationale here even though
the concrete contracts were refined in this phase).
"""

from __future__ import annotations

import logging

from app.agent.providers import LLMProvider
from app.reasoning._structured_call import complete_json
from app.reasoning.contracts import AnalyticalQuestion, Claim

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are the question-understanding stage of an analytical reasoning pipeline for "
    "a dataset that has already been uploaded. Read the user's question and the dataset "
    "summary, then reply with JSON ONLY (no prose, no markdown code fences) matching "
    "exactly this shape:\n"
    '{"intent": "descriptive|diagnostic|comparative|predictive|prescriptive", '
    '"requested_metrics": [string], "requested_dimensions": [string], '
    '"requested_time_range": string or null, "requested_population": string or null, '
    '"explicit_constraints": [string], "required_confidence": string or null, '
    '"language": string, '
    '"claims": [{"text": string, "source": "user_asserted or system_inferred"}]}\n\n'
    "intent guide: 'descriptive' = what/how much happened; 'diagnostic' = why did "
    "something happen; 'comparative' = is X different from / significantly different "
    "than Y; 'predictive' = forecast / what will happen; 'prescriptive' = what should "
    "we do about it.\n"
    "requested_metrics/requested_dimensions: exact column-name-like tokens when the "
    "user names them, not generic phrases. explicit_constraints: any explicit "
    "scale/scope claim the user makes (e.g. '10 million rows', 'last 12 months') as "
    "separate strings, verbatim enough to be checked against the data later. claims: "
    "any factual assertion the question makes or implies that should be verified "
    "against the data before being trusted, tagged user_asserted (the user stated it) "
    "or system_inferred (you inferred it from phrasing)."
)


def parse_question(
    provider: LLMProvider, question: str, dataset_summary: str
) -> tuple[AnalyticalQuestion, list[Claim]]:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "system", "content": f"Dataset summary: {dataset_summary}"},
        {"role": "user", "content": question},
    ]
    raw = complete_json(provider, messages)
    if raw is None:
        logger.warning("question_parser: structured output unparseable after retry; using conservative fallback")
        return AnalyticalQuestion(original_question=question, intent="descriptive"), []

    try:
        analytical_question = AnalyticalQuestion(
            original_question=question,
            intent=raw.get("intent") or "descriptive",
            requested_metrics=list(raw.get("requested_metrics") or []),
            requested_dimensions=list(raw.get("requested_dimensions") or []),
            requested_time_range=raw.get("requested_time_range"),
            requested_population=raw.get("requested_population"),
            explicit_constraints=list(raw.get("explicit_constraints") or []),
            required_confidence=raw.get("required_confidence"),
            language=raw.get("language") or "en",
        )
    except Exception:
        logger.warning("question_parser: structured output had an invalid shape; using conservative fallback")
        return AnalyticalQuestion(original_question=question, intent="descriptive"), []

    claims: list[Claim] = []
    for entry in raw.get("claims") or []:
        try:
            claims.append(Claim(text=entry["text"], source=entry.get("source", "system_inferred")))
        except Exception:
            continue  # one malformed claim entry does not invalidate the whole call
    return analytical_question, claims
