"""Constraint validation (Phase 3B.4).

Deterministic, no LLM call, no extra tool-router call — reuses `profile_dataset`
directly (the same function `agent.py` already calls to build `dataset_context`), so
this costs nothing beyond what the system already computes per request.

Turns the free-text scope fields on `AnalyticalQuestion` (requested metrics/
dimensions/time range/population, plus any explicit scale claim in
`explicit_constraints`) into `Claim`s with a verified/falsified status, and raises a
`Limitation` for every hard mismatch. This is the deterministic, testable replacement
for what used to be purely a prose instruction in `agent.py`'s `SYSTEM_PROMPT` — e.g.
the project's original "10-million-row question answered silently against a 4,000-row
dataset" finding is now a specific, regression-tested code path (`_check_scale_claims`
below), not something relying on the model remembering to check.

This module never silently substitutes a smaller/different scope for what was asked —
per Phase 3B's explicit rule, a mismatch always produces a `verified_false` Claim plus
a `blocks_conclusion` Limitation, for the orchestrator to surface, never to quietly
work around.
"""

from __future__ import annotations

import re
from datetime import datetime

from app.reasoning.contracts import AnalyticalQuestion, Claim, Limitation
from app.tools.profiler import profile_dataset

_LAST_N_RE = re.compile(r"last\s+(\d+)\s+(day|week|month|year)s?", re.IGNORECASE)
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}

_SCALE_RE = re.compile(
    r"([\d,]+(?:\.\d+)?)\s*(million|thousand|k|m)?\s*(rows|records|transactions|entries|customers)",
    re.IGNORECASE,
)
_SCALE_MULTIPLIER = {"million": 1_000_000, "m": 1_000_000, "thousand": 1_000, "k": 1_000}
_SCALE_MISMATCH_RATIO = 10.0  # order-of-magnitude difference before flagging


def _known_columns(profile: dict) -> set[str]:
    return (
        set(profile["numeric_columns"])
        | set(profile["categorical_columns"])
        | set(profile["date_columns"])
        | set(profile["boolean_columns"])
        | set(profile.get("text_columns", []))
    )


def _check_columns_exist(names: list[str], known: set[str], role: str) -> tuple[list[Claim], list[Limitation]]:
    claims: list[Claim] = []
    limitations: list[Limitation] = []
    for name in names:
        exists = name in known
        claims.append(
            Claim(
                text=f"Column '{name}' exists in the dataset (requested as a {role}).",
                source="system_inferred",
                status="verified_true" if exists else "verified_false",
            )
        )
        if not exists:
            limitations.append(
                Limitation(
                    category="missing_data",
                    text=f"Requested {role} '{name}' is not a column in this dataset.",
                    severity="blocks_conclusion",
                )
            )
    return claims, limitations


def _check_time_range(requested: str, date_ranges: dict) -> tuple[Claim, Limitation | None]:
    if not date_ranges:
        claim = Claim(
            text=f"Dataset has date coverage supporting: {requested!r}",
            source="user_asserted",
            status="unverifiable",
            note="Dataset has no date columns at all.",
        )
        return claim, Limitation(
            category="insufficient_coverage",
            text="A time-range request was made but this dataset has no date columns.",
            severity="blocks_conclusion",
        )

    match = _LAST_N_RE.search(requested)
    if not match:
        return (
            Claim(
                text=f"Dataset has date coverage supporting: {requested!r}",
                source="user_asserted",
                status="unverifiable",
                note="Time range was not in a recognized 'last N <day/week/month/year>' form; not automatically checked.",
            ),
            None,
        )

    n, unit = int(match.group(1)), match.group(2).lower()
    requested_days = n * _UNIT_DAYS[unit]

    widest_col, max_span_days = None, 0
    for col, rng in date_ranges.items():
        try:
            span = (datetime.fromisoformat(rng["max"]) - datetime.fromisoformat(rng["min"])).days
        except (KeyError, ValueError):
            continue
        if span > max_span_days:
            widest_col, max_span_days = col, span

    if widest_col is None:
        return (
            Claim(
                text=f"Dataset has date coverage supporting: {requested!r}",
                source="user_asserted",
                status="unverifiable",
                note="Could not parse this dataset's date coverage.",
            ),
            None,
        )

    available_units = round(max_span_days / _UNIT_DAYS[unit], 1)
    if max_span_days + 1 < requested_days * 0.9:
        note = (
            f"Requested {n} {unit}(s) (~{requested_days} days), but column '{widest_col}' only "
            f"spans about {available_units} {unit}(s) ({max_span_days} days) of data."
        )
        return (
            Claim(text=f"Dataset covers the requested {requested!r}", source="user_asserted", status="verified_false", note=note),
            Limitation(category="insufficient_coverage", text=note, severity="blocks_conclusion"),
        )

    note = f"Column '{widest_col}' spans {available_units} {unit}(s), covering the requested range."
    return (
        Claim(text=f"Dataset covers the requested {requested!r}", source="user_asserted", status="verified_true", note=note),
        None,
    )


def _check_scale_claims(constraints: list[str], actual_rows: int) -> tuple[list[Claim], list[Limitation]]:
    claims: list[Claim] = []
    limitations: list[Limitation] = []
    for text in constraints:
        match = _SCALE_RE.search(text)
        if not match:
            continue
        num_str, unit = match.group(1), match.group(2)
        try:
            claimed = float(num_str.replace(",", "")) * _SCALE_MULTIPLIER.get((unit or "").lower(), 1)
        except ValueError:
            continue
        if claimed <= 0:
            continue
        ratio = max(claimed / max(actual_rows, 1), actual_rows / claimed)
        if ratio >= _SCALE_MISMATCH_RATIO:
            note = f"Dataset actually has {actual_rows:,} rows, not ~{int(claimed):,} as the question implies."
            claims.append(Claim(text=text, source="user_asserted", status="verified_false", note=note))
            limitations.append(
                Limitation(
                    category="insufficient_coverage",
                    text=f"Requested scale (~{int(claimed):,} rows) does not match the dataset's actual {actual_rows:,} rows.",
                    severity="blocks_conclusion",
                )
            )
        else:
            claims.append(Claim(text=text, source="user_asserted", status="verified_true"))
    return claims, limitations


def validate_question(question: AnalyticalQuestion, df) -> tuple[list[Claim], list[Limitation], dict]:
    """Returns (claims, limitations, profile). `profile` is exposed so the caller
    (orchestrator) doesn't have to call `profile_dataset` a second time."""
    profile = profile_dataset(df)
    known = _known_columns(profile)

    claims: list[Claim] = []
    limitations: list[Limitation] = []

    c, l = _check_columns_exist(question.requested_metrics, known, "metric")
    claims += c
    limitations += l

    c, l = _check_columns_exist(question.requested_dimensions, known, "dimension")
    claims += c
    limitations += l

    if question.requested_time_range:
        claim, limitation = _check_time_range(question.requested_time_range, profile.get("date_ranges", {}))
        claims.append(claim)
        if limitation:
            limitations.append(limitation)

    if question.requested_population:
        # Light-touch: population is free text (e.g. "customers in Region A"), not a
        # single column name, so it is not structurally validated here beyond noting
        # that it was not checked -- documented as a known limitation of this minimum
        # viable pass, not silently treated as verified.
        claims.append(
            Claim(
                text=f"Requested population: {question.requested_population!r}",
                source="user_asserted",
                status="unverifiable",
                note="Population scope is free text and is not structurally validated against the dataset in this phase.",
            )
        )

    c, l = _check_scale_claims(question.explicit_constraints, profile["rows"])
    claims += c
    limitations += l

    return claims, limitations, profile
