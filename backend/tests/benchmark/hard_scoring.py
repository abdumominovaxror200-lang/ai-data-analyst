"""Multi-dimensional scoring for the hard real-world professional-analyst benchmark.

Explicit design mandate for this benchmark (see the user's own instructions, preserved
verbatim in `.agent/hard_realworld_benchmark.md`): this must NOT be optimized for a high
pass rate, must NOT be made artificially easy, and a failure here is valuable evidence,
not a bug to make disappear. This module exists to make failure *legible* along 15
separate reasoning dimensions, not just collapse everything into one PASS/FAIL bit.

Deliberately reuses `tests.benchmark.scoring`'s real, unmodified structural checks and
the real, unmodified `causation_guard` / `epistemic_checks` / `Finding.cross_checked` /
`Uncertainty` mechanisms wherever they already cover a dimension -- new checks are added
only where those existing mechanisms genuinely don't reach (method-selection specificity,
cross-check requirement, refusal detection, communication-quality overclaim phrases,
scalability tool-choice). Every new check is built against real, source-verified fields
and functions (see each check's docstring for what it was checked against), after the
previous benchmark round's lesson that guessed field names silently never match anything.

The 15 dimensions (per the mission spec, section 17):
question_understanding, premise_validation, data_quality_awareness, tool_selection,
method_selection, numerical_correctness, statistical_correctness, evidence_grounding,
uncertainty_calibration, causal_restraint, hypothesis_quality, cross_checking,
recommendation_grounding, scalability, communication_quality.

Not every dimension applies to every case -- a dimension with no applicable check for a
given case is recorded as `None` (not counted for or against), exactly like
`scoring.py`'s existing `CheckResult.passed: bool | None` convention.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from app.agent.providers import MockProvider, ProviderResponse, ToolCall
from app.datasets.storage import DatasetRecord
from app.reasoning import causation_guard
from app.reasoning.contracts import AnalysisResult, AnalyticalQuestion
from app.reasoning.orchestrator import ReasoningOrchestrator
from app.reasoning.premise_validator import validate_question
from tests.benchmark.scoring import build_mock_provider_from_script

DIMENSIONS = [
    "question_understanding",
    "premise_validation",
    "data_quality_awareness",
    "tool_selection",
    "method_selection",
    "numerical_correctness",
    "statistical_correctness",
    "evidence_grounding",
    "uncertainty_calibration",
    "causal_restraint",
    "hypothesis_quality",
    "cross_checking",
    "recommendation_grounding",
    "scalability",
    "communication_quality",
]

Verdict = Literal["PASS", "PARTIAL", "FAIL", "UNMEASURED"]

# Overclaim phrases with no legitimate hedged use in a data-analysis answer -- distinct
# from causation_guard's causal-specific phrase table (this catches non-causal
# overconfidence: manufactured certainty about a future/general claim).
_OVERCLAIM_PHRASES = [
    "guaranteed", "100% certain", "100% confident", "will definitely",
    "always results in", "no doubt", "proven beyond question", "certainly will",
    "without question", "undeniably", "absolutely will",
]


@dataclass
class DimensionCheck:
    dimension: str
    passed: bool | None  # None = not applicable to this case
    detail: str = ""


@dataclass
class HardCaseResult:
    case_id: str
    category: str
    verdict: Verdict
    dimension_checks: list[DimensionCheck] = field(default_factory=list)
    explanation: str = ""
    result: AnalysisResult | None = None
    provider_failure: bool = False


def _word_stem_match(a: str, b: str) -> bool:
    """Two independent, deliberately narrow checks, either of which is enough:

    1. Bidirectional substring: catches a compound/prefixed word embedding the other
       word verbatim -- "duplicate" inside "deduplicate", "collinear" inside
       "multicollinearity", "unit" inside "units". This does NOT fire on suffix
       changes that alter the shared root's own spelling (e.g. "currency" is not a
       substring of "currencies", "definition" is not a substring of "define").
    2. Prefix comparison capped at 5 characters: catches exactly that remaining
       class -- "definition"/"define" share "defin", "currency"/"currencies" share
       "curre" -- while a 5-char cap keeps unrelated same-prefix words like "small"
       vs "sample" from falsely matching (they diverge by the 5th character).

    Verified against every real pluralization/inflection pair found while
    root-causing the bug this function replaces (see _keyword_overlap's docstring)."""
    if a == b:
        return True
    if len(a) >= 4 and len(b) >= 4 and (a in b or b in a):
        return True
    shortest = min(len(a), len(b))
    if shortest < 4:
        return False
    prefix_len = min(shortest, 5)
    return a[:prefix_len] == b[:prefix_len]


def _keyword_overlap(needle: str, haystacks: list[str]) -> bool:
    """Stem-aware word match against open-ended model prose, not exact-token-set
    intersection. Root-caused during this benchmark's first real run (see
    .agent/hard_realworld_benchmark.md): the original exact-word-match version
    (inherited from tests/benchmark/scoring.py, which only ever compares against
    short structural Claim/Limitation text, not open-ended model prose) missed real,
    correct hedges purely from ordinary English inflection ("duplicate" vs
    "duplicates", "definition" vs "define", "unit" vs "units") -- a
    scoring-infrastructure bug, not a reasoning-pipeline one; it alone produced ~56
    false FAILs across this suite's data_quality_awareness checks."""
    # len > 2 (not > 3): several deliberately-chosen trap words are meaningful
    # 3-letter tokens ("utc", "app", "dip", "mix") that a >3 filter silently dropped
    # entirely, a second real bug found alongside the inflection one -- 3-letter
    # words are still safe here since _word_stem_match only allows an EXACT match
    # (no fuzzy stemming) below its 4-character floor, so this can't introduce noisy
    # partial matches on short common words.
    needle_words = [w for w in re.sub(r"[^a-z0-9\s]", " ", needle.lower()).split() if len(w) > 2]
    if not needle_words:
        return False
    for h in haystacks:
        h_words = [w for w in re.sub(r"[^a-z0-9\s]", " ", h.lower()).split() if len(w) > 2]
        for nw in needle_words:
            if any(_word_stem_match(nw, hw) for hw in h_words):
                return True
    return False


def _text_haystacks(result: AnalysisResult) -> list[str]:
    return (
        [result.final_answer_text]
        + [c.text + " " + (c.note or "") for c in result.claims]
        + [l.text for l in result.limitations]
        + list(result.reasoning_trace)
    )


def _find_overclaim_phrases(text: str) -> list[str]:
    lowered = text.lower()
    return [p for p in _OVERCLAIM_PHRASES if p in lowered]


def run_hard_case(case: dict, record: DatasetRecord) -> HardCaseResult:
    exp = case.get("expected", {})
    checks: list[DimensionCheck] = []

    if not case.get("script"):
        # Deterministic-only: no LLM call at all, mirrors scoring.py's own pattern for
        # premise-validation-only cases (a genuinely missing column/population).
        question = AnalyticalQuestion(
            original_question=case["user_question"],
            intent=exp.get("expected_intent") or "descriptive",
            requested_metrics=case.get("_probe_metrics", []),
            requested_dimensions=case.get("_probe_dimensions", []),
            requested_time_range=case.get("_probe_time_range"),
            explicit_constraints=case.get("_probe_scale_constraints", []),
        )
        claims, limitations, _profile = validate_question(question, record.df)
        checks.append(_check_premise_validation(exp, claims, limitations))
        checks.append(_check_data_quality_awareness(exp, [c.text + " " + (c.note or "") for c in claims] + [l.text for l in limitations]))
        verdict = _verdict(checks)
        return HardCaseResult(case_id=case["case_id"], category=case["category"], verdict=verdict, dimension_checks=checks, explanation=_explain(checks))

    provider = build_mock_provider_from_script(case["script"])
    return _run_with_provider(case, record, provider)


def _run_with_provider(case: dict, record: DatasetRecord, provider) -> HardCaseResult:
    """Shared scoring core for any already-built provider (MockProvider for the
    scripted suite, or a real LLMProvider for a live spot-check) -- the exact same 15
    dimension checks apply regardless of what produced the AnalysisResult, so a real
    run is graded identically to a scripted one and the two are directly comparable."""
    exp = case.get("expected", {})
    orchestrator = ReasoningOrchestrator(provider)
    result = orchestrator.analyze(record, case["user_question"])

    checks: list[DimensionCheck] = [
        _check_question_understanding(exp, result),
        _check_premise_validation_result(exp, result),
        _check_data_quality_awareness(exp, _text_haystacks(result)),
        _check_tool_selection(exp, result),
        _check_method_selection(exp, result),
        _check_numerical_correctness(exp, result),
        _check_statistical_correctness(exp, result),
        _check_evidence_grounding(result),
        _check_uncertainty_calibration(exp, result),
        _check_causal_restraint(exp, result),
        _check_hypothesis_quality(exp, result),
        _check_cross_checking(exp, result),
        _check_recommendation_grounding(exp, result),
        _check_scalability(exp, result),
        _check_communication_quality(exp, result),
    ]

    verdict = _verdict(checks)
    return HardCaseResult(
        case_id=case["case_id"], category=case["category"], verdict=verdict,
        dimension_checks=checks, explanation=_explain(checks), result=result,
    )


def _verdict(checks: list[DimensionCheck]) -> Verdict:
    applicable = [c for c in checks if c.passed is not None]
    if not applicable:
        return "PARTIAL"
    if all(c.passed for c in applicable):
        return "PASS"
    if any(c.passed for c in applicable):
        return "PARTIAL"
    return "FAIL"


def _explain(checks: list[DimensionCheck]) -> str:
    failed = [c for c in checks if c.passed is False]
    if not failed:
        return "All applicable dimension checks passed."
    return "Failed: " + "; ".join(f"{c.dimension} ({c.detail})" for c in failed)


# --- Per-dimension checks ------------------------------------------------------------


def _check_question_understanding(exp: dict, result: AnalysisResult) -> DimensionCheck:
    expected_intent = exp.get("expected_intent")
    if not expected_intent:
        return DimensionCheck("question_understanding", None)
    ok = result.question.intent == expected_intent
    return DimensionCheck("question_understanding", ok, f"expected intent {expected_intent!r}, got {result.question.intent!r}" if not ok else "")


def _check_premise_validation(exp: dict, claims, limitations) -> DimensionCheck:
    required = exp.get("required_constraints")
    if not required:
        return DimensionCheck("premise_validation", None)
    haystacks = [c.text + " " + (c.note or "") for c in claims] + [l.text for l in limitations]
    found = any(_keyword_overlap(rc, haystacks) for rc in required)
    return DimensionCheck("premise_validation", found, "no matching claim/limitation found" if not found else "")


def _check_premise_validation_result(exp: dict, result: AnalysisResult) -> DimensionCheck:
    required = exp.get("required_constraints")
    if not required:
        return DimensionCheck("premise_validation", None)
    haystacks = [c.text + " " + (c.note or "") for c in result.claims] + [l.text for l in result.limitations]
    found = any(_keyword_overlap(rc, haystacks) for rc in required)
    return DimensionCheck("premise_validation", found, "no matching claim/limitation found" if not found else "")


def _check_data_quality_awareness(exp: dict, haystacks: list[str]) -> DimensionCheck:
    traps = exp.get("must_flag_traps")
    if not traps:
        return DimensionCheck("data_quality_awareness", None)
    missing = [t for t in traps if not _keyword_overlap(t.replace("_", " "), haystacks)]
    ok = not missing
    return DimensionCheck("data_quality_awareness", ok, f"traps not flagged anywhere in answer/claims/limitations/trace: {missing}" if not ok else "")


def _check_tool_selection(exp: dict, result: AnalysisResult) -> DimensionCheck:
    expected = exp.get("tool_category")
    if expected is None:
        return DimensionCheck("tool_selection", None)
    actual = set(result.plan.capability_categories) if result.plan else set()
    if not expected:
        ok = not actual
        return DimensionCheck("tool_selection", ok, f"expected no applicable category, got {actual}" if not ok else "")
    ok = bool(actual & set(expected))
    return DimensionCheck("tool_selection", ok, f"expected one of {expected}, got {actual}" if not ok else "")


def _check_method_selection(exp: dict, result: AnalysisResult) -> DimensionCheck:
    """Specific-tool check, stricter than tool_selection's category-level check --
    proves the SPECIFIC method (e.g. `t_test`, not just any STATISTICS tool) was used,
    since picking the wrong specific method within a correct category is exactly the
    'plausible but wrong' failure mode this benchmark targets (mean vs median,
    correlation vs regression, aggregate vs cohort)."""
    required_any = exp.get("required_tools_any")
    forbidden = exp.get("forbidden_tools_alone")
    used = {e.source_tool for e in result.evidence}
    if required_any:
        ok = bool(used & set(required_any))
        if not ok:
            return DimensionCheck("method_selection", False, f"expected one of {required_any}, tools used: {used}")
        if forbidden and used and used.issubset(set(forbidden)):
            return DimensionCheck("method_selection", False, f"only used forbidden/inappropriate tool(s): {used}")
        return DimensionCheck("method_selection", True)
    if forbidden:
        ok = not used.issubset(set(forbidden)) if used else True
        return DimensionCheck("method_selection", ok, f"only used forbidden/inappropriate tool(s): {used}" if not ok else "")
    return DimensionCheck("method_selection", None)


def _check_numerical_correctness(exp: dict, result: AnalysisResult) -> DimensionCheck:
    spec = exp.get("expected_numeric_result")
    if not spec:
        return DimensionCheck("numerical_correctness", None)
    found_value = None
    for ev in result.evidence:
        if spec["field"] in ev.result_summary:
            found_value = ev.result_summary[spec["field"]]
            break
    if found_value is None:
        return DimensionCheck("numerical_correctness", False, f"field '{spec['field']}' not found in any evidence")
    ok = abs(float(found_value) - float(spec["value"])) <= spec.get("tolerance", 0.01)
    return DimensionCheck("numerical_correctness", ok, f"expected ~{spec['value']}, got {found_value}" if not ok else "")


def _check_statistical_correctness(exp: dict, result: AnalysisResult) -> DimensionCheck:
    """For STATISTICAL_RESULT evidence, a real Uncertainty must have been extractable
    (verifier.py's `_extract_uncertainty` returns non-None for every STATISTICAL_RESULT
    evidence unconditionally -- this checks that mechanism actually fired, i.e. that a
    STATISTICAL_RESULT tool call really happened when the case expects statistical
    reasoning, not just a descriptive aggregate dressed up as one)."""
    if not exp.get("require_statistical_evidence"):
        return DimensionCheck("statistical_correctness", None)
    stat_findings = [f for f in result.findings if f.classification == "STATISTICAL_RESULT"]
    if not stat_findings:
        return DimensionCheck("statistical_correctness", False, "no STATISTICAL_RESULT finding produced")
    missing_uncertainty = [f.id for f in stat_findings if f.uncertainty is None]
    ok = not missing_uncertainty
    return DimensionCheck("statistical_correctness", ok, f"STATISTICAL_RESULT finding(s) with no extracted Uncertainty: {missing_uncertainty}" if not ok else "")


def _check_evidence_grounding(result: AnalysisResult) -> DimensionCheck:
    evidence_ids = {e.id for e in result.evidence}
    finding_ids = {f.id for f in result.findings}
    traceable = all(ev_id in evidence_ids for f in result.findings for ev_id in f.supporting_evidence)
    if result.recommendation:
        traceable = traceable and all(fid in finding_ids for fid in result.recommendation.supporting_findings)
    unsupported = [f for f in result.findings if f.classification in ("FACT", "CALCULATED_RESULT", "STATISTICAL_RESULT") and not f.supporting_evidence]
    ok = traceable and not unsupported
    detail = ""
    if not ok:
        parts = []
        if not traceable:
            parts.append("a finding/recommendation referenced a non-existent id")
        if unsupported:
            parts.append(f"{len(unsupported)} solid finding(s) with no supporting evidence")
        detail = "; ".join(parts)
    return DimensionCheck("evidence_grounding", ok, detail)


def _check_uncertainty_calibration(exp: dict, result: AnalysisResult) -> DimensionCheck:
    metric = exp.get("require_uncertainty_for_metric")
    if not metric:
        return DimensionCheck("uncertainty_calibration", None)
    matching_evidence_ids = {e.id for e in result.evidence if e.metric == metric}
    matching_findings = [f for f in result.findings if set(f.supporting_evidence) & matching_evidence_ids]
    if not matching_findings:
        return DimensionCheck("uncertainty_calibration", False, f"no finding supported by evidence for metric '{metric}'")
    ok = any(f.uncertainty is not None and f.uncertainty.level in ("known", "estimated") for f in matching_findings)
    return DimensionCheck("uncertainty_calibration", ok, f"no calibrated (known/estimated) uncertainty attached to any finding for '{metric}'" if not ok else "")


def _check_causal_restraint(exp: dict, result: AnalysisResult) -> DimensionCheck:
    behavior = exp.get("expected_causal_behavior", "not_applicable")
    if behavior == "must_hedge_unless_causal_hypothesis_supported":
        remaining = causation_guard.find_causal_phrases(result.final_answer_text)
        ok = not remaining
        return DimensionCheck("causal_restraint", ok, f"unhedged causal phrase(s) remained: {remaining}" if not ok else "")
    if behavior == "must_generate_2_to_4_competing_hypotheses":
        ok = 2 <= len(result.hypotheses) <= 4
        return DimensionCheck("causal_restraint", ok, f"expected 2-4 hypotheses, got {len(result.hypotheses)}" if not ok else "")
    if behavior == "causal_language_is_justified":
        # The mission explicitly requires cases where causal language SHOULD be
        # permitted (real, statistically supported hypothesis) -- so the guard doesn't
        # become "blindly conservative." Checks the guard did NOT strip a causal
        # relationship mention when a hypothesis genuinely reached "supported".
        mentions = causation_guard.classify_relationship_language(result.final_answer_text)
        ok = any(m.category in ("causal_hypothesis", "causal_unhedged") for m in mentions) or any(h.status == "supported" for h in result.hypotheses)
        return DimensionCheck("causal_restraint", ok, "expected causal language to be permitted (a hypothesis reached 'supported') but none was found" if not ok else "")
    return DimensionCheck("causal_restraint", None)


def _check_hypothesis_quality(exp: dict, result: AnalysisResult) -> DimensionCheck:
    """Real structural invariant (section 14 of the mission spec: never convert
    HYPOTHESIS/ASSUMPTION -> FACT): a Hypothesis can only be `status == "supported"` if
    it actually has evidence_for -- status is never manufactured without backing."""
    unsupported_but_marked_supported = [h.id for h in result.hypotheses if h.status == "supported" and not h.evidence_for]
    ok = not unsupported_but_marked_supported
    if not ok:
        return DimensionCheck("hypothesis_quality", False, f"hypothesis marked 'supported' with no evidence_for: {unsupported_but_marked_supported}")
    if exp.get("expected_hypotheses_range"):
        lo, hi = exp["expected_hypotheses_range"]
        ok2 = lo <= len(result.hypotheses) <= hi
        return DimensionCheck("hypothesis_quality", ok2, f"expected {lo}-{hi} hypotheses, got {len(result.hypotheses)}" if not ok2 else "")
    if not result.hypotheses:
        return DimensionCheck("hypothesis_quality", None)
    return DimensionCheck("hypothesis_quality", True)


def _check_cross_checking(exp: dict, result: AnalysisResult) -> DimensionCheck:
    if not exp.get("require_cross_check"):
        return DimensionCheck("cross_checking", None)
    any_cross_checked = any(f.cross_checked for f in result.findings)
    any_disagreement_limitation = any("disagree" in l.text.lower() for l in result.limitations)
    ok = any_cross_checked or any_disagreement_limitation
    return DimensionCheck("cross_checking", ok, "no cross-checked finding and no disagreement limitation -- a second, independent verification tool call was expected" if not ok else "")


def _check_recommendation_grounding(exp: dict, result: AnalysisResult) -> DimensionCheck:
    if exp.get("must_refuse"):
        ok = result.recommendation is None or result.recommendation.confidence is None
        return DimensionCheck("recommendation_grounding", ok, "expected no recommendation (or an honest null-confidence one) given insufficient evidence, but got a confident one" if not ok else "")
    if not result.recommendation:
        return DimensionCheck("recommendation_grounding", None)
    grounded = bool(result.recommendation.supporting_findings) or result.recommendation.confidence is None
    return DimensionCheck("recommendation_grounding", grounded, "recommendation has neither supporting findings nor an honest null confidence" if not grounded else "")


def _check_scalability(exp: dict, result: AnalysisResult) -> DimensionCheck:
    if not exp.get("scalability_case"):
        return DimensionCheck("scalability", None)
    used_sql = any(e.source_tool in ("run_sql_query", "explain_sql_query") for e in result.evidence)
    haystacks = _text_haystacks(result)
    acknowledged_scale = any(_keyword_overlap(kw, haystacks) for kw in ("scale", "memory", "sample", "chunk", "large"))
    ok = used_sql or acknowledged_scale
    return DimensionCheck("scalability", ok, "neither a SQL-pushdown tool call nor an explicit scale acknowledgment was found" if not ok else "")


def _check_communication_quality(exp: dict, result: AnalysisResult) -> DimensionCheck:
    found = _find_overclaim_phrases(result.final_answer_text)
    ok = not found
    return DimensionCheck("communication_quality", ok, f"overclaiming phrase(s) found: {found}" if not ok else "")


# --- Report aggregation ---------------------------------------------------------------


def summarize(case_results: list[HardCaseResult], cases_by_id: dict[str, dict]) -> dict[str, Any]:
    total = len(case_results)
    passed = sum(1 for r in case_results if r.verdict == "PASS")
    partial = sum(1 for r in case_results if r.verdict == "PARTIAL")
    failed = sum(1 for r in case_results if r.verdict == "FAIL")
    unmeasured = sum(1 for r in case_results if r.verdict == "UNMEASURED")
    provider_failures = sum(1 for r in case_results if r.provider_failure)
    scored = total - unmeasured - provider_failures
    overall_pct = round(100.0 * passed / scored, 1) if scored else 0.0

    by_category: dict[str, list[HardCaseResult]] = {}
    for r in case_results:
        by_category.setdefault(r.category, []).append(r)
    category_scores = {
        cat: round(100.0 * sum(1 for r in results if r.verdict == "PASS") / len(results), 1)
        for cat, results in by_category.items()
    }

    by_dimension: dict[str, list[bool]] = {d: [] for d in DIMENSIONS}
    for r in case_results:
        for dc in r.dimension_checks:
            if dc.passed is not None:
                by_dimension[dc.dimension].append(dc.passed)
    dimension_scores = {
        dim: (round(100.0 * sum(vals) / len(vals), 1) if vals else None)
        for dim, vals in by_dimension.items()
    }

    failure_reasons: dict[str, int] = {}
    for r in case_results:
        for dc in r.dimension_checks:
            if dc.passed is False:
                failure_reasons[dc.dimension] = failure_reasons.get(dc.dimension, 0) + 1

    return {
        "total_cases": total,
        "passed": passed,
        "partial": partial,
        "failed": failed,
        "unmeasured": unmeasured,
        "provider_failures": provider_failures,
        "overall_score_pct": overall_pct,
        "category_scores_pct": category_scores,
        "dimension_scores_pct": dimension_scores,
        "most_common_failure_dimensions": dict(sorted(failure_reasons.items(), key=lambda kv: -kv[1])),
        "cases": [
            {
                "case_id": r.case_id,
                "category": r.category,
                "verdict": r.verdict,
                "explanation": r.explanation,
                "failed_dimensions": [dc.dimension for dc in r.dimension_checks if dc.passed is False],
            }
            for r in case_results
        ],
    }
