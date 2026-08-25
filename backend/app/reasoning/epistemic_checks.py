"""Machine-checkable epistemic-principle constraints (Phase 4 P2).

This module deliberately does NOT bolt on a generic "philosophy knowledge base."
Each function below implements exactly ONE concrete epistemic principle as a pure,
deterministic check over the pipeline's *already-computed* contract objects
(`contracts.py`) -- it inspects what `question_parser`/`premise_validator`/
`planner`/`executor`/`verifier`/`synthesizer` already produced, it never re-derives
a value one of those modules already computes (e.g. it never re-classifies a
`Finding` or re-hedges causal text; it calls into `causation_guard.find_causal_phrases`
for check #2 rather than reimplementing phrase detection).

Every function returns a `list[str]` of human-readable violation descriptions --
empty means "no violation of this principle was detected", never "not checked".
Every function is safe on `None`/empty input: no argument is ever assumed non-None,
and no check ever raises.

`check_all(...)` aggregates all 10 checks and is the function
`orchestrator.py`/`AnalysisResult.principle_violations` is expected to call once
this module is wired in (a future wave's job, not this one's) -- its signature
intentionally takes the same plain values `orchestrator.py` already threads through
`verifier.build_findings`/`synthesizer.synthesize` (see that file), not a full
`AnalysisResult`, since it is meant to be callable mid-pipeline before the final
object is assembled.

Numbered checks below correspond to the 9 named epistemic principles from the task
brief (principle 2, "distinguish correlation from causation", and the redundant
"don't manufacture certainty" check are both represented; see each function's
docstring for exactly which principle it encodes).
"""

from __future__ import annotations

import re

from app.reasoning.causation_guard import find_causal_phrases
from app.reasoning.contracts import (
    AnalyticalQuestion,
    Claim,
    Evidence,
    Finding,
    Hypothesis,
    Limitation,
    Recommendation,
)

_OBSERVATION_CLASSIFICATIONS = ("FACT", "CALCULATED_RESULT", "STATISTICAL_RESULT")

# Numbers embedded in free text, e.g. "12,345", "3.5", "40%" (the trailing % is not
# part of the match group -- callers scanning result_summary values pick up the bare
# number too, so "40" in "40%" still matches "40" in a result_summary of 40.0).
_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")


# --- 1. Observation vs. interpretation -----------------------------------------


def check_observation_vs_interpretation(findings: list[Finding] | None) -> list[str]:
    """Principle: distinguish observation from interpretation.

    Every `Finding` classified as an "observation" (FACT / CALCULATED_RESULT /
    STATISTICAL_RESULT) must be backed by at least one piece of `Evidence`
    (`supporting_evidence` non-empty). A "solid" classification with zero backing
    evidence is an unlabeled interpretation wearing an observation's badge.
    """
    violations: list[str] = []
    for f in findings or []:
        if f.classification in _OBSERVATION_CLASSIFICATIONS and not f.supporting_evidence:
            violations.append(
                f"Finding '{f.id}' is classified {f.classification} (an observation) "
                "but has no supporting_evidence -- an unsupported claim should not "
                "carry an observation-level classification."
            )
    return violations


# --- 2. Correlation vs. causation (belt-and-suspenders second opinion) ---------


def check_correlation_vs_causation(
    final_answer_text: str | None, hypotheses: list[Hypothesis] | None
) -> list[str]:
    """Principle: distinguish correlation from causation.

    Reuses `causation_guard.find_causal_phrases` (the module's own public detector)
    to scan the final answer text for unhedged causal phrasing. If unhedged causal
    language remains AND no hypothesis justifies it, this is flagged.

    This check is deliberately redundant with `causation_guard.enforce_causation_guard`,
    which should already have rewritten any unjustified causal phrasing in
    `final_answer_text` before this check ever runs. In normal operation this
    function should always return an empty list -- it exists purely as a regression
    trap in case `enforce_causation_guard` is ever skipped, bypassed, or its output
    discarded somewhere upstream of the final result.

    The "is a hypothesis justifying this causal language" gate below is a minimal,
    explicitly-commented reimplementation of `causation_guard`'s private gate
    condition (`is_causal=True` and `status in {"supported", "weakly_supported"}`)
    -- duplicated here on purpose because this check must not depend on
    `causation_guard`'s internal (non-public) implementation, only its two stable
    public functions (`find_causal_phrases`/`enforce_causation_guard`).
    """
    if not final_answer_text:
        return []
    matched = find_causal_phrases(final_answer_text)
    if not matched:
        return []

    # --- minimal reimplementation of causation_guard's private justification gate ---
    justified = any(
        h.is_causal and h.status in ("supported", "weakly_supported") for h in (hypotheses or [])
    )
    # --- end reimplementation ---
    if justified:
        return []

    return [
        "final_answer_text contains unhedged causal phrasing "
        f"{sorted(set(matched))} with no supporting causal hypothesis -- "
        "causation_guard should have hedged this; treat as a regression."
    ]


# --- 3. Evidence vs. assumption --------------------------------------------------


def _is_labeled_assumption(finding: Finding, assumptions: list[str]) -> bool:
    # Heuristic: `Recommendation.assumptions` is free text, not keyed by Finding.id,
    # so "explicitly listed" is checked via a loose substring match against the
    # finding's own statement. Documented as approximate, not exact, matching.
    statement = (finding.statement or "").strip().lower()
    if not statement:
        return False
    for a in assumptions:
        a_norm = (a or "").strip().lower()
        if a_norm and (a_norm in statement or statement in a_norm):
            return True
    return False


def check_evidence_vs_assumption(
    findings: list[Finding] | None, recommendation: Recommendation | None
) -> list[str]:
    """Principle: distinguish evidence from assumption.

    A `Recommendation`'s `supporting_findings` must never cite a `Finding`
    classified `ASSUMPTION` as if it were solid support unless that assumption is
    also explicitly disclosed in `recommendation.assumptions` -- i.e. the reader
    must be told "this recommendation rests on an assumption", not left to assume
    every supporting finding is equally solid.
    """
    if recommendation is None:
        return []
    findings_by_id = {f.id: f for f in (findings or [])}
    violations: list[str] = []
    for fid in recommendation.supporting_findings:
        f = findings_by_id.get(fid)
        if f is None:
            continue
        if f.classification == "ASSUMPTION" and not _is_labeled_assumption(
            f, recommendation.assumptions
        ):
            violations.append(
                f"Recommendation cites finding '{fid}' (classified ASSUMPTION) in "
                "supporting_findings without disclosing it in recommendation.assumptions."
            )
    return violations


# --- 4. Uncertainty acknowledged -------------------------------------------------


def check_uncertainty_acknowledged(findings: list[Finding] | None) -> list[str]:
    """Principle: acknowledge uncertainty.

    Every `Finding` classified `STATISTICAL_RESULT` must carry a non-null
    `uncertainty` field. A statistical result presented with zero uncertainty
    information is manufacturing false precision.
    """
    violations: list[str] = []
    for f in findings or []:
        if f.classification == "STATISTICAL_RESULT" and f.uncertainty is None:
            violations.append(
                f"Finding '{f.id}' is classified STATISTICAL_RESULT but has no "
                "uncertainty information attached."
            )
    return violations


# --- 5. No inference beyond the data ---------------------------------------------


def _numbers_in_text(text: str) -> list[float]:
    numbers = []
    for m in _NUMBER_RE.finditer(text):
        raw = m.group(0).replace(",", "")
        try:
            numbers.append(float(raw))
        except ValueError:
            continue
    return numbers


def _collect_numbers(value, acc: set[float]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        acc.add(round(float(value), 2))
    elif isinstance(value, str):
        for n in _numbers_in_text(value):
            acc.add(round(n, 2))
    elif isinstance(value, dict):
        for v in value.values():
            _collect_numbers(v, acc)
    elif isinstance(value, list):
        for v in value:
            _collect_numbers(v, acc)


def check_no_inference_beyond_data(
    recommendation: Recommendation | None, evidence: list[Evidence] | None
) -> list[str]:
    """Principle: don't infer beyond the data.

    Light heuristic, not a perfect check: if `recommendation.expected_business_effect`
    states a specific quantitative claim (a number or percentage), that number
    should appear somewhere in the gathered `Evidence`'s `result_summary` values.
    A number in the stated business effect that appears nowhere in any evidence is
    a strong signal of an invented figure not grounded in what was actually
    computed.

    This is a string/number-scanning heuristic: it will miss values that were
    algebraically derived from evidence numbers (e.g. "20% growth" computed from
    two evidence values of 100 and 120, neither of which is literally "20"), and it
    may under-flag if the same digits appear coincidentally elsewhere in evidence.
    It compares numbers rounded to 2 decimal places, so it will also miss cases
    where the recommendation text rounds/approximates a number that appears more
    precisely in evidence (e.g. "~40%" vs. an evidence value of 39.87) -- callers
    should treat this as a red flag worth a human look, not a proof of fabrication.
    """
    if recommendation is None or not recommendation.expected_business_effect:
        return []
    text_numbers = _numbers_in_text(recommendation.expected_business_effect)
    if not text_numbers:
        return []

    evidence_numbers: set[float] = set()
    for e in evidence or []:
        _collect_numbers(e.result_summary, evidence_numbers)

    violations: list[str] = []
    for n in text_numbers:
        rn = round(n, 2)
        if not any(abs(rn - en) < 0.01 for en in evidence_numbers):
            violations.append(
                f"recommendation.expected_business_effect states a figure ({n}) that "
                "does not appear in any gathered Evidence.result_summary -- possible "
                "invented number not grounded in the analysis (heuristic check)."
            )
    return violations


# --- 6. Alternative explanations considered --------------------------------------


def check_alternative_explanations_considered(
    question_intent: str | None, hypotheses: list[Hypothesis] | None
) -> list[str]:
    """Principle: consider alternative explanations.

    For a "diagnostic" question (a "why did X happen" question), fewer than 2
    competing hypotheses is flagged as informational -- this is NOT necessarily
    wrong (some diagnostic questions genuinely have one clear, well-evidenced
    explanation), so treat this as a prompt to double-check reasoning depth, not a
    hard failure. Always returned in the list when it fires; callers that want a
    "blocking" vs. "informational" split should filter on this docstring's
    semantics, since `principle_violations` itself is an undifferentiated
    `list[str]`.
    """
    if question_intent != "diagnostic":
        return []
    count = len(hypotheses or [])
    if count < 2:
        return [
            f"Diagnostic question has only {count} hypothesis(es) considered -- "
            "fewer than 2 can be legitimate, but is worth double-checking that "
            "alternative explanations were actually considered (informational)."
        ]
    return []


# --- 7. Falsifiability / internal consistency ------------------------------------


def check_falsifiability(hypotheses: list[Hypothesis] | None) -> list[str]:
    """Principle: prefer falsifiable hypotheses (checked here as internal
    self-consistency of the hypothesis-evaluator's own output).

    A `Hypothesis` marked `status == "supported"` while it also carries non-empty
    `evidence_against` is an internally inconsistent record: you cannot be
    "supported" while your own recorded contradicting evidence is non-trivial. This
    check does not re-derive `status` from the evidence itself -- it only checks
    that whatever process set `status` did not contradict the same object's own
    `evidence_against` list.
    """
    violations: list[str] = []
    for h in hypotheses or []:
        if h.status == "supported" and h.evidence_against:
            violations.append(
                f"Hypothesis '{h.id}' has status='supported' but also lists "
                f"evidence_against={h.evidence_against} -- internally inconsistent."
            )
    return violations


# --- 8. Missing evidence identified ----------------------------------------------


def check_missing_evidence_identified(
    question: AnalyticalQuestion | None,
    limitations: list[Limitation] | None,
    claims: list[Claim] | None,
) -> list[str]:
    """Principle: identify missing evidence.

    Every `Claim` with `status in ("verified_false", "unverifiable")` represents a
    detected problem with the question's premise (a nonexistent column, an
    unmet scale/time-range claim, an unverifiable population scope, ...). That
    problem must actually surface to the reader as a `Limitation` somewhere in the
    result -- flagged if such a claim exists but the result carries zero
    `Limitation`s at all (a detected problem silently dropped between premise
    validation and the final result).

    `question` is accepted (matching the calling convention `orchestrator.py`
    threads through the pipeline) but is not itself inspected -- the check operates
    purely on `claims`/`limitations`, since `Limitation` objects are not linked back
    to the specific `Claim` that produced them (`Limitation.affected_findings`
    links to `Finding.id`, not `Claim`), so a per-claim correspondence check is not
    structurally possible with today's contracts; this checks "at least one
    Limitation exists at all" as the practical, checkable version of that principle.
    """
    del question  # accepted for calling-convention parity; not inspected -- see docstring
    problematic = [c for c in (claims or []) if c.status in ("verified_false", "unverifiable")]
    if problematic and not (limitations or []):
        return [
            f"{len(problematic)} claim(s) marked verified_false/unverifiable exist "
            "but the result carries no Limitation at all -- a detected premise "
            "problem appears to have been silently dropped."
        ]
    return []


# --- 9. No overconfidence ---------------------------------------------------------


def check_no_overconfidence(recommendation: Recommendation | None) -> list[str]:
    """Principle: avoid overconfidence.

    Conservative heuristic floor: a `Recommendation` at `confidence == "high"`
    backed by fewer than 2 `supporting_findings` is flagged -- a single finding,
    however solid on its own, is thin support for a "high" confidence business
    recommendation. This is deliberately a blunt floor, distinct from and
    complementary to a separate, more rigorous evidence-strength/grounding module
    (owned elsewhere this wave) -- this function does not import or depend on that
    module.
    """
    if recommendation is None:
        return []
    if recommendation.confidence == "high" and len(recommendation.supporting_findings) < 2:
        return [
            "Recommendation has confidence='high' but only "
            f"{len(recommendation.supporting_findings)} supporting_finding(s) -- "
            "thin support for a high-confidence claim (conservative heuristic floor)."
        ]
    return []


# --- 10. No manufactured certainty ------------------------------------------------


def check_no_manufactured_certainty(findings: list[Finding] | None) -> list[str]:
    """Principle: never manufacture certainty.

    A `Finding.uncertainty.level == "known"` claims the value is definitively
    established -- but if `point_estimate` is `None`, the record never actually
    states what the "known" value is. Claiming certainty about an unstated value is
    a contradiction in the record itself.
    """
    violations: list[str] = []
    for f in findings or []:
        u = f.uncertainty
        if u is not None and u.level == "known" and u.point_estimate is None:
            violations.append(
                f"Finding '{f.id}' has uncertainty.level='known' but "
                "uncertainty.point_estimate is None -- claiming certainty without "
                "stating the known value."
            )
    return violations


# --- Aggregator --------------------------------------------------------------------


def check_all(
    question: AnalyticalQuestion | None,
    claims: list[Claim] | None,
    findings: list[Finding] | None,
    evidence: list[Evidence] | None,
    hypotheses: list[Hypothesis] | None,
    limitations: list[Limitation] | None,
    recommendation: Recommendation | None,
    final_answer_text: str | None,
) -> list[str]:
    """Runs all 10 epistemic checks and returns the concatenated, deduplicated
    violation list. This is the function `orchestrator.py` is expected to call
    (a future wave's wiring job, not this module's) to populate
    `AnalysisResult.principle_violations`, using the same plain values it already
    threads through `verifier.build_findings`/`synthesizer.synthesize` today.

    Safe on a maximally-empty/default input (all empty lists / None) -- never raises.
    """
    violations: list[str] = []
    violations += check_observation_vs_interpretation(findings)
    violations += check_correlation_vs_causation(final_answer_text, hypotheses)
    violations += check_evidence_vs_assumption(findings, recommendation)
    violations += check_uncertainty_acknowledged(findings)
    violations += check_no_inference_beyond_data(recommendation, evidence)
    violations += check_alternative_explanations_considered(
        question.intent if question is not None else None, hypotheses
    )
    violations += check_falsifiability(hypotheses)
    violations += check_missing_evidence_identified(question, limitations, claims)
    violations += check_no_overconfidence(recommendation)
    violations += check_no_manufactured_certainty(findings)

    deduped: list[str] = []
    seen: set[str] = set()
    for v in violations:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped
