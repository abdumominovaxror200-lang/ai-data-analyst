"""Deterministic safeguards for inferential multiplicity and insufficient data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app.reasoning.contracts import AnalyticalQuestion, AnalysisPlan, Evidence, Limitation

_INFERENCE_TERMS = (
    "statistical", "significant", "significance", "confidence interval",
    "uncertainty", "p-value", "p value", "hypothesis test",
)
_COMPARISON_TERMS = (
    " versus ", " vs ", "compare", "comparison", "difference", "change from",
    "year over year", "year-over-year",
)
_ADJUSTMENT_KEYS = {
    "adjusted_p_value", "adjusted_p_values", "p_value_adjustment",
    "multiple_comparison_adjustment", "correction_method",
}


@dataclass(frozen=True)
class MultipleComparisonsAssessment:
    comparison_count: int
    adjusted: bool
    limitation: Limitation | None


def assess_multiple_comparisons(evidence: list[Evidence]) -> MultipleComparisonsAssessment:
    """Warn when one synthesis consumes a family of unadjusted inferential claims."""
    count = sum(_p_value_count(item.result_summary) for item in evidence if item.evidence_type == "STATISTICAL_RESULT")
    adjusted = count > 0 and all(
        _has_adjustment(item.result_summary)
        for item in evidence
        if item.evidence_type == "STATISTICAL_RESULT" and _p_value_count(item.result_summary)
    )
    if count < 2 or adjusted:
        return MultipleComparisonsAssessment(count, adjusted, None)
    affected = [f"finding_{index}" for index, item in enumerate(evidence) if _p_value_count(item.result_summary)]
    limitation = Limitation(
        category="multiple_comparisons",
        severity="reduces_confidence",
        text=(
            f"{count} unadjusted inferential comparisons were evaluated in this analysis. "
            "The chance of at least one false-positive result is higher than the per-test alpha; "
            "treat individual p-values as exploratory unless hypotheses were pre-specified or a "
            "documented correction such as Holm or Benjamini-Hochberg is applied."
        ),
        affected_findings=affected,
    )
    return MultipleComparisonsAssessment(count, False, limitation)


def insufficient_data_limitation(
    question: AnalyticalQuestion,
    plan: AnalysisPlan,
    evidence: list[Evidence],
) -> Limitation | None:
    """Return a blocking refusal when the requested inference has no valid evidence."""
    if not evidence:
        return Limitation(
            # Preserve the established API/scoring category for a total absence
            # of evidence; the stronger blocking severity is the T1.7 change.
            category="missing_data",
            severity="blocks_conclusion",
            text=(
                "Insufficient data: no analytical tool produced usable evidence for the requested "
                "question and scope. Additional observations or a supported analytical path are required."
            ),
        )
    if not _requires_formal_inference(question, plan):
        return None

    comparison_requested = _comparison_requested(question, plan)
    for item in evidence:
        if item.evidence_type != "STATISTICAL_RESULT" or not _has_valid_inference(item.result_summary):
            continue
        if comparison_requested and not _has_two_period_scope(item):
            continue
        return None

    scope_text = " matching the requested two-period scope" if comparison_requested else ""
    return Limitation(
        category="insufficient_data",
        severity="blocks_conclusion",
        text=(
            "Insufficient data for the requested statistical conclusion: no valid inferential evidence"
            f"{scope_text} was produced. Descriptive results alone cannot establish statistical uncertainty "
            "or significance."
        ),
    )


def ensure_multiple_comparisons_warning_visible(
    text: str, assessment: MultipleComparisonsAssessment
) -> str:
    """Ensure the structured warning is also present in the user-visible narrative."""
    limitation = assessment.limitation
    if limitation is None or limitation.text in text:
        return text
    return f"{text}\n\nStatistical caution: {limitation.text}"


def _p_value_count(value: Any, parent_key: str = "") -> int:
    if isinstance(value, dict):
        count = 0
        for key, nested in value.items():
            normalized = str(key).lower()
            if normalized in {"p_value", "f_p_value"} and _is_number(nested):
                # Assumption-test diagnostics are not business hypotheses and do not
                # expand the substantive comparison family.
                if parent_key not in {"residual_normality", "heteroscedasticity"}:
                    count += 1
            else:
                count += _p_value_count(nested, normalized)
        return count
    if isinstance(value, list):
        return sum(_p_value_count(item, parent_key) for item in value)
    return 0


def _has_adjustment(summary: dict[str, Any]) -> bool:
    return any(key in summary and summary[key] not in (None, "", False) for key in _ADJUSTMENT_KEYS)


def _has_valid_inference(summary: dict[str, Any]) -> bool:
    if _p_value_count(summary) > 0:
        return True
    low = summary.get("lower_bound", summary.get("ci_low"))
    high = summary.get("upper_bound", summary.get("ci_high"))
    if _is_number(low) and _is_number(high):
        return True
    interval = summary.get("confidence_interval")
    return isinstance(interval, (list, tuple)) and len(interval) == 2 and all(_is_number(value) for value in interval)


def _has_two_period_scope(evidence: Evidence) -> bool:
    temporal = evidence.scope.temporal if evidence.scope else None
    return bool(
        temporal
        and temporal.current_start and temporal.current_end
        and temporal.previous_start and temporal.previous_end
    )


def _requires_formal_inference(question: AnalyticalQuestion, plan: AnalysisPlan) -> bool:
    if question.required_confidence:
        return True
    return any(term in _analysis_text(question, plan) for term in _INFERENCE_TERMS)


def _comparison_requested(question: AnalyticalQuestion, plan: AnalysisPlan) -> bool:
    text = " " + _analysis_text(question, plan) + " "
    return bool(question.requested_time_range) and any(term in text for term in _COMPARISON_TERMS)


def _analysis_text(question: AnalyticalQuestion, plan: AnalysisPlan) -> str:
    values = [
        question.original_question,
        question.requested_time_range or "",
        *question.explicit_constraints,
        plan.objective,
        *plan.steps,
        *plan.expected_outputs,
        *plan.validation_steps,
    ]
    return " ".join(values).lower()


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
