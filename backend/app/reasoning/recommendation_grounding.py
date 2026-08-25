"""Recommendation grounding check (Phase 4 remediation of a Phase 3C QA finding).

Background (`.agent/completed_tasks.md`'s Phase 3C section, "Remaining limitations"):
the benchmark scoring framework's structural check #10 ("recommendation grounding",
`backend/tests/benchmark/scoring.py`) only verifies a `Recommendation` has *some*
supporting finding (or an honest `confidence=None`) -- it never checks whether that
finding is actually *strong enough* to justify the stated confidence. Concretely, a
recommendation citing a single weak, purely-correlational finding could claim
`confidence="high"` and nothing in the pipeline caught it (this is exactly the tie,
rather than a clear win, QA-BENCHMARK-ENGINEER found on the adv_05
honesty-vs-overclaiming pair).

This module fixes that at the source: given a `Recommendation` plus the pipeline's own
`Finding`/`Evidence`/`Hypothesis` objects, it deterministically computes the maximum
confidence the evidence actually supports and flags any mismatch. It performs no LLM
call and raises no exception on malformed/missing input -- it must be safe to run on
any real pipeline output, including a recommendation with no supporting findings at
all.

This module is intentionally read-only with respect to the rest of the pipeline: it
does not modify `synthesizer.py`, `orchestrator.py`, `verifier.py`, or
`causation_guard.py`. A future integration step (not this one) is expected to call
`evaluate_recommendation_grounding` from the orchestrator and substitute
`adjusted_confidence` for `recommendation.confidence` before the result reaches the
user -- this module only computes what that correction should be.

---

## The evidence-strength rule (deterministic, auditable -- read this before changing
## any threshold)

Every piece of `Evidence` reachable from the recommendation (via
`recommendation.supporting_findings` -> `Finding.supporting_evidence` -> `Evidence.id`)
is scored individually into a tier, and the recommendation's overall
`evidence_strength` is the *strongest* tier found among that evidence (a recommendation
is only as weak as its best piece of support requires -- if at least one strong,
well-powered statistical result backs it, that is real grounding even if other cited
findings are weaker).

Per-evidence fields are read from the SAME raw fields `verifier.py`'s
`_extract_uncertainty` and the real tool return shapes
(`app/tools/hypothesis.py`, `app/tools/regression.py`) already use -- no new
data-extraction convention is invented:
  - significance: `result_summary["significant"]` if present (already a bool computed
    by the tool); else `p_value < alpha` if both present; else a confidence interval
    that excludes zero (`lower_bound`/`upper_bound`); else `linear_regression`'s
    `significant_features` being non-empty. `None` if none of these are present
    (not a formal test at all, or an unrecognized result shape).
  - effect size: `magnitude` (from `effect_size`'s Cohen's d classification --
    "negligible"/"small" do NOT count as meaningful, only "medium"/"large" do) if
    present; else `r_squared` (from `linear_regression`) checked against
    `_R_SQUARED_MEANINGFUL_THRESHOLD = 0.3`. `None` if neither field is present.
  - sample size: `Evidence.sample_size` if set; else `result_summary["n"]` /
    `["n_observations"]`; else the sum of `group_a`/`group_b`'s `"n"` (two-sample
    t-test / effect_size shape); else the sum of each group's `"n"` in ANOVA's
    `"groups"` dict. `None` if none of these are present.

Per-evidence tier, in this exact priority order (first matching rule wins):
  1. **weak** -- sample size is known and below `_LOW_SAMPLE_SIZE_THRESHOLD = 10`
     (reusing `verifier.py`'s existing "tiny sample" floor as the *same* floor here,
     so the two modules never disagree about what counts as tiny). This overrides
     everything else, including a significant p-value -- a significant result from 4
     observations is not trustworthy regardless of its p-value.
  2. **weak** -- `evidence_type != "STATISTICAL_RESULT"` (purely `FACT`/
     `CALCULATED_RESULT` evidence -- no formal significance test was ever run, e.g. a
     plain aggregate or a bare correlation number).
  3. **weak** -- a formal test was run and it was NOT significant (`significant is
     False`).
  4. **strong** -- `STATISTICAL_RESULT`, significant (`True`), a meaningful effect
     size (`True`), AND an adequate sample size (known and
     `>= _ADEQUATE_SAMPLE_SIZE_THRESHOLD = 30`). All four conditions are required --
     a statistically significant but practically tiny effect (e.g. p=0.001 with
     Cohen's d magnitude "negligible", or r_squared=0.02) never reaches "strong"
     because the effect-size condition fails.
  5. **moderate** -- everything else that survives rules 1-3 (i.e. `STATISTICAL_RESULT`
     evidence that is significant or has unknown significance, and isn't tiny-sample).
     This covers all three of the task's named moderate cases: significant-but-small
     effect (fails rule 4's effect-size condition), a meaningful effect size without a
     formal significance test (fails rule 4's significance condition), and an
     adequate-but-not-large sample with `STATISTICAL_RESULT` evidence (fails rule 4's
     sample-size condition).

`evidence_strength` (report-level) = the strongest tier among all resolved evidence,
or `"none"` if `recommendation.supporting_findings` is empty or none of the referenced
findings resolve to any real `Evidence`.

`recommended_confidence_ceiling` is a pure function of `evidence_strength`:
`strong -> "high"`, `moderate -> "medium"`, `weak -> "low"`, `none -> None` (no
confidence claim at all is justified when there is no real evidence).

`adjusted_confidence`: `recommendation.confidence` capped at
`recommended_confidence_ceiling` (by rank low < medium < high) if it exceeded the
ceiling; the original value, unchanged, otherwise. If the ceiling is `None` (no
evidence), `adjusted_confidence` is always `None` regardless of what the recommendation
originally claimed -- missing evidence must produce no confidence claim, not invented
certainty.

`is_observational_only`: `True` when none of the resolved evidence is
`STATISTICAL_RESULT` (every piece is `FACT`/`CALCULATED_RESULT` -- purely descriptive
or correlational) AND the recommendation's own text (`recommendation` +
`expected_business_effect`) uses causal-sounding language (detected via
`causation_guard.find_causal_phrases`, reused rather than re-implemented, plus a small
supplementary keyword list for phrasing that guard doesn't cover, e.g. "will improve")
AND no `Hypothesis` in `hypotheses` actually justifies a causal claim (`is_causal=True`
and `status` in `{"supported", "weakly_supported"}` -- the same justification test
`causation_guard._has_justifying_causal_hypothesis` uses). If a justifying hypothesis
exists, causal language is not a violation of this check -- the same "restrict only
*unsupported* causal language" principle `causation_guard.py` already applies.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.reasoning.causation_guard import find_causal_phrases
from app.reasoning.contracts import Evidence, Finding, Hypothesis, Recommendation
from app.reasoning.verifier import _LOW_SAMPLE_SIZE_THRESHOLD as LOW_SAMPLE_SIZE_THRESHOLD

# "strong" requires materially more than "not tiny" -- see rule 4 in the module
# docstring. Chosen as a conventional textbook floor for a reasonably powered simple
# comparison; documented here (not buried in code) precisely because it's a judgment
# call, same spirit as `_CROSS_CHECK_RELATIVE_TOLERANCE` in verifier.py.
_ADEQUATE_SAMPLE_SIZE_THRESHOLD = 30

# r_squared >= this counts as a "meaningful" effect for linear_regression evidence
# (conventional small-to-medium boundary for explained variance in social-science/
# business-analytics contexts). Below this, a regression result is not treated as
# strong evidence even if individual coefficients are statistically significant.
_R_SQUARED_MEANINGFUL_THRESHOLD = 0.3

# Effect-size magnitudes (from `app/tools/hypothesis.py::effect_size`) that count as
# "meaningful". "negligible"/"small" explicitly do not, even if paired with a tiny
# p-value -- this is the module's answer to the task's required example 3.
_MEANINGFUL_MAGNITUDES = frozenset({"medium", "large"})

_EvidenceTier = Literal["strong", "moderate", "weak"]
_ConfidenceLevel = Literal["high", "medium", "low"]

_CONFIDENCE_RANK: dict[_ConfidenceLevel, int] = {"low": 1, "medium": 2, "high": 3}
_TIER_TO_CEILING: dict[_EvidenceTier, _ConfidenceLevel] = {
    "strong": "high",
    "moderate": "medium",
    "weak": "low",
}

# Causal-sounding phrasing that `causation_guard.find_causal_phrases` does not itself
# cover (it targets rewriting the model's *output* for a narrower set of phrases;
# this list widens detection for judging whether a *recommendation* implies a causal
# driver -- read-only use, causation_guard.py is not modified).
_SUPPLEMENTARY_CAUSAL_KEYWORDS = (
    "because",
    "will improve",
    "will increase",
    "will decrease",
    "will reduce",
    "will boost",
    "will drive",
    "will grow",
    "drives",
    "is driving",
    "is the driver",
    "the driver of",
    "responsible for",
)


class RecommendationGroundingReport(BaseModel):
    """The result of checking one `Recommendation` against the evidence that actually
    supports it. Pydantic (not a plain dataclass) for consistency with every other
    contract in this package (`contracts.py`) -- it gets the same free validation,
    `.model_dump()` JSON serialization, and immutable-by-convention field typing the
    rest of the reasoning layer already relies on, and callers that already work with
    `Recommendation`/`Finding`/etc. get a uniform object model rather than mixing
    dataclasses and pydantic models.
    """

    evidence_strength: Literal["strong", "moderate", "weak", "none"]
    is_observational_only: bool
    sample_size_adequate: bool | None = None
    statistically_significant: bool | None = None
    effect_size_meaningful: bool | None = None
    recommended_confidence_ceiling: Literal["high", "medium", "low"] | None = None
    violations: list[str] = Field(default_factory=list)
    adjusted_confidence: Literal["high", "medium", "low"] | None = None


def _effective_sample_size(evidence: Evidence) -> int | None:
    if isinstance(evidence.sample_size, int) and not isinstance(evidence.sample_size, bool):
        return evidence.sample_size
    r = evidence.result_summary or {}
    for key in ("n", "n_observations"):
        v = r.get(key)
        if isinstance(v, int) and not isinstance(v, bool):
            return v
    group_a, group_b = r.get("group_a"), r.get("group_b")
    if isinstance(group_a, dict) and isinstance(group_b, dict):
        na, nb = group_a.get("n"), group_b.get("n")
        if isinstance(na, int) and isinstance(nb, int) and not isinstance(na, bool) and not isinstance(nb, bool):
            return na + nb
    groups = r.get("groups")
    if isinstance(groups, dict):
        ns = [g.get("n") for g in groups.values() if isinstance(g, dict) and isinstance(g.get("n"), int) and not isinstance(g.get("n"), bool)]
        if ns:
            return sum(ns)
    return None


def _is_significant(evidence: Evidence) -> bool | None:
    r = evidence.result_summary or {}
    if isinstance(r.get("significant"), bool):
        return r["significant"]
    p_value, alpha = r.get("p_value"), r.get("alpha")
    if isinstance(p_value, (int, float)) and not isinstance(p_value, bool) and isinstance(alpha, (int, float)) and not isinstance(alpha, bool):
        return p_value < alpha
    lower, upper = r.get("lower_bound"), r.get("upper_bound")
    if isinstance(lower, (int, float)) and not isinstance(lower, bool) and isinstance(upper, (int, float)) and not isinstance(upper, bool):
        return not (lower <= 0 <= upper)
    significant_features = r.get("significant_features")
    if isinstance(significant_features, list):
        return len(significant_features) > 0
    return None


def _effect_size_meaningful(evidence: Evidence) -> bool | None:
    r = evidence.result_summary or {}
    magnitude = r.get("magnitude")
    if isinstance(magnitude, str):
        return magnitude in _MEANINGFUL_MAGNITUDES
    r_squared = r.get("r_squared")
    if isinstance(r_squared, (int, float)) and not isinstance(r_squared, bool):
        return r_squared >= _R_SQUARED_MEANINGFUL_THRESHOLD
    return None


def _tier_for_evidence(evidence: Evidence) -> _EvidenceTier:
    """Implements rules 1-5 of the module docstring's evidence-strength rule, in
    priority order. See the docstring for the full rationale of each branch."""
    n = _effective_sample_size(evidence)
    if n is not None and n < LOW_SAMPLE_SIZE_THRESHOLD:
        return "weak"
    if evidence.evidence_type != "STATISTICAL_RESULT":
        return "weak"

    significant = _is_significant(evidence)
    if significant is False:
        return "weak"

    effect_meaningful = _effect_size_meaningful(evidence)
    adequate_n = n is not None and n >= _ADEQUATE_SAMPLE_SIZE_THRESHOLD
    if significant is True and effect_meaningful is True and adequate_n:
        return "strong"
    return "moderate"


_TIER_ORDER: dict[_EvidenceTier, int] = {"weak": 1, "moderate": 2, "strong": 3}


def _resolve_evidence(
    recommendation: Recommendation,
    findings: list[Finding],
    evidence: list[Evidence],
) -> list[Evidence]:
    """Every distinct Evidence reachable from the recommendation's cited findings.
    Silently ignores dangling finding/evidence ids -- this module never raises on
    malformed pipeline output."""
    findings_by_id = {f.id: f for f in findings}
    evidence_by_id = {e.id: e for e in evidence}

    resolved: dict[str, Evidence] = {}
    for finding_id in recommendation.supporting_findings:
        finding = findings_by_id.get(finding_id)
        if finding is None:
            continue
        for evidence_id in finding.supporting_evidence:
            ev = evidence_by_id.get(evidence_id)
            if ev is not None:
                resolved[ev.id] = ev
    return list(resolved.values())


def _has_justifying_causal_hypothesis(hypotheses: list[Hypothesis]) -> bool:
    # Mirrors causation_guard._has_justifying_causal_hypothesis's exact test --
    # duplicated rather than imported since that name is private to that module, but
    # kept intentionally identical so the two modules never disagree about what
    # "justified" causation means.
    return any(h.is_causal and h.status in ("supported", "weakly_supported") for h in hypotheses)


def _recommendation_uses_causal_language(recommendation: Recommendation) -> bool:
    text = recommendation.recommendation or ""
    if recommendation.expected_business_effect:
        text = f"{text} {recommendation.expected_business_effect}"
    if find_causal_phrases(text):
        return True
    lowered = text.lower()
    return any(keyword in lowered for keyword in _SUPPLEMENTARY_CAUSAL_KEYWORDS)


def evaluate_recommendation_grounding(
    recommendation: Recommendation,
    findings: list[Finding],
    evidence: list[Evidence],
    hypotheses: list[Hypothesis],
) -> RecommendationGroundingReport:
    """Checks whether `recommendation.confidence` is actually justified by the
    evidence backing it. Never raises -- safe to call on any real pipeline output,
    including `recommendation.supporting_findings == []`.

    See the module docstring for the exact evidence-strength rule and every
    threshold used here.
    """
    resolved_evidence = _resolve_evidence(recommendation, findings, evidence)

    violations: list[str] = []

    if not resolved_evidence:
        report = RecommendationGroundingReport(
            evidence_strength="none",
            is_observational_only=False,
            sample_size_adequate=None,
            statistically_significant=None,
            effect_size_meaningful=None,
            recommended_confidence_ceiling=None,
            adjusted_confidence=None,
        )
        violations.append(
            "recommendation has no supporting findings or evidence backing it "
            "-- no confidence claim is justified"
        )
        if recommendation.confidence is not None:
            violations.append(
                "missing evidence must produce UNKNOWN/LOW confidence rather than "
                f"invented certainty (stated confidence: '{recommendation.confidence}')"
            )
        report.violations = violations
        return report

    tiers = [(_tier_for_evidence(e), e) for e in resolved_evidence]
    best_tier, primary_evidence = max(tiers, key=lambda pair: _TIER_ORDER[pair[0]])
    evidence_strength: Literal["strong", "moderate", "weak"] = best_tier

    sample_size_adequate: bool | None = None
    n = _effective_sample_size(primary_evidence)
    if n is not None:
        sample_size_adequate = n >= _ADEQUATE_SAMPLE_SIZE_THRESHOLD
        if not sample_size_adequate:
            violations.append(f"recommendation is based on a sample of only {n} observations")

    statistically_significant: bool | None = None
    effect_size_meaningful: bool | None = None
    if primary_evidence.evidence_type == "STATISTICAL_RESULT":
        statistically_significant = _is_significant(primary_evidence)
        effect_size_meaningful = _effect_size_meaningful(primary_evidence)
        if statistically_significant is False:
            violations.append(
                f"the strongest supporting statistical result ({primary_evidence.source_tool}) "
                "was not statistically significant"
            )
        if statistically_significant is True and effect_size_meaningful is False:
            violations.append(
                f"the strongest supporting statistical result ({primary_evidence.source_tool}) "
                "is statistically significant but has a negligible/small effect size -- "
                "this does not justify strong/high confidence on its own"
            )

    is_observational_only = (
        all(e.evidence_type != "STATISTICAL_RESULT" for e in resolved_evidence)
        and _recommendation_uses_causal_language(recommendation)
        and not _has_justifying_causal_hypothesis(hypotheses)
    )
    if is_observational_only:
        violations.append(
            "only observational/correlational evidence supports a recommendation "
            "implying causation"
        )

    ceiling = _TIER_TO_CEILING[evidence_strength]

    adjusted_confidence: Literal["high", "medium", "low"] | None = recommendation.confidence
    if recommendation.confidence is not None:
        if _CONFIDENCE_RANK[recommendation.confidence] > _CONFIDENCE_RANK[ceiling]:
            violations.append(
                f"stated confidence '{recommendation.confidence}' exceeds the evidence "
                f"ceiling '{ceiling}'"
            )
            adjusted_confidence = ceiling

    return RecommendationGroundingReport(
        evidence_strength=evidence_strength,
        is_observational_only=is_observational_only,
        sample_size_adequate=sample_size_adequate,
        statistically_significant=statistically_significant,
        effect_size_meaningful=effect_size_meaningful,
        recommended_confidence_ceiling=ceiling,
        violations=violations,
        adjusted_confidence=adjusted_confidence,
    )
