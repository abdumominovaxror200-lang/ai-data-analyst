"""Evidence validation, cross-check, and finding classification (Phase 3B.5/3B.6
support). Entirely deterministic -- no LLM call, no new tool call by default.

`Evidence.evidence_type` was already fixed at gathering time by `executor.py` (which
tool produced it), not asked of an LLM here -- this module derives each `Finding`'s
classification and `Uncertainty` directly from that plus the tool's own result shape,
so a persuasive-sounding synthesis call downstream cannot talk a `CALCULATED_RESULT`
into being presented as a `FACT`, or a plain aggregate into carrying a confidence
interval it never actually computed.

Cross-checking (Phase 3B "cross-check" pipeline stage): implemented here as a
deterministic corroboration check -- a finding is marked `cross_checked=True` when
two *different* tools independently produced evidence about the same metric and
agree closely enough to be considered consistent, or flagged as disagreeing via a
Limitation if they materially differ. This is the "minimum viable" version of
cross-checking: it corroborates what the plan's execution phase already gathered,
rather than issuing extra tool calls per finding -- keeping this stage's cost at
zero additional tool/LLM calls, consistent with the "do not perform unnecessary
analysis" rule.
"""

from __future__ import annotations

from app.reasoning.contracts import Evidence, Finding, FindingClassification, Limitation, Uncertainty

_CROSS_CHECK_RELATIVE_TOLERANCE = 0.10  # 10%: close enough to call "agreeing"
_LOW_SAMPLE_SIZE_THRESHOLD = 10


def _classification_for(evidence_type: str) -> FindingClassification:
    # Evidence.evidence_type is already one of "FACT"/"CALCULATED_RESULT"/
    # "STATISTICAL_RESULT" -- these three map 1:1 onto three of the six Finding
    # classifications. HYPOTHESIS/ASSUMPTION/UNKNOWN are assigned elsewhere (by the
    # orchestrator, for hypotheses and for capability-unavailable cases respectively)
    # since no tool-derived Evidence is ever itself a hypothesis or an assumption.
    if evidence_type in ("FACT", "CALCULATED_RESULT", "STATISTICAL_RESULT"):
        return evidence_type  # type: ignore[return-value]
    return "UNKNOWN"


def _extract_uncertainty(evidence: Evidence) -> Uncertainty | None:
    if evidence.evidence_type != "STATISTICAL_RESULT":
        return None
    r = evidence.result_summary

    if "p_value" in r:
        alpha = r.get("alpha")
        return Uncertainty(
            level="known" if r.get("p_value") is not None else "unavailable",
            metric=evidence.metric,
            confidence_level=(1 - alpha) if isinstance(alpha, (int, float)) else None,
            method=r.get("test"),
        )
    if "lower_bound" in r and "upper_bound" in r:
        return Uncertainty(
            level="estimated",
            metric=evidence.metric,
            point_estimate=r.get("mean"),
            interval_low=r.get("lower_bound"),
            interval_high=r.get("upper_bound"),
            confidence_level=r.get("confidence"),
            method="confidence_interval",
        )
    forecast_points = r.get("forecast")
    if isinstance(forecast_points, list) and forecast_points:
        first = forecast_points[0]
        return Uncertainty(
            level="estimated",
            metric=evidence.metric,
            point_estimate=first.get("forecast"),
            interval_low=first.get("lower_bound"),
            interval_high=first.get("upper_bound"),
            confidence_level=r.get("confidence_level"),
            method=f"{r.get('method_used', 'forecast')} prediction interval",
        )
    if "silhouette_score" in r:
        return Uncertainty(level="estimated", metric="cluster fit quality", point_estimate=r.get("silhouette_score"), method="silhouette_score")
    if "r_squared" in r:
        return Uncertainty(level="estimated", metric="model fit (R-squared)", point_estimate=r.get("r_squared"), method="ordinary_least_squares")

    return Uncertainty(level="uncertain", metric=evidence.metric, method=f"{evidence.source_tool}: no recognized uncertainty field in this result shape")


def _statement_for(evidence: Evidence) -> str:
    if evidence.metric:
        return f"{evidence.source_tool} produced a result for '{evidence.metric}'."
    return f"{evidence.source_tool} produced a result relevant to the analysis."


def _numeric_point(evidence: Evidence) -> float | None:
    r = evidence.result_summary
    for key in ("mean", "value", "coefficient", "statistic"):
        v = r.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def _cross_check(evidence: list[Evidence]) -> tuple[set[str], list[Limitation]]:
    """Returns (ids of evidence that were corroborated, limitations for disagreements)."""
    corroborated: set[str] = set()
    limitations: list[Limitation] = []
    by_metric: dict[str, list[Evidence]] = {}
    for e in evidence:
        if e.metric:
            by_metric.setdefault(e.metric, []).append(e)

    for metric, items in by_metric.items():
        distinct_tools = {e.source_tool: e for e in items}.values()
        if len(distinct_tools) < 2:
            continue
        points = [(e, _numeric_point(e)) for e in distinct_tools]
        points = [(e, p) for e, p in points if p is not None]
        if len(points) < 2:
            continue
        base_e, base_p = points[0]
        for e, p in points[1:]:
            denom = max(abs(base_p), abs(p), 1e-9)
            if abs(p - base_p) / denom <= _CROSS_CHECK_RELATIVE_TOLERANCE:
                corroborated.add(base_e.id)
                corroborated.add(e.id)
            else:
                limitations.append(
                    Limitation(
                        category="methodological",
                        text=(
                            f"'{metric}' results disagree between {base_e.source_tool} "
                            f"({base_p}) and {e.source_tool} ({p}) -- treat with caution."
                        ),
                        severity="reduces_confidence",
                    )
                )
    return corroborated, limitations


def build_findings(evidence: list[Evidence]) -> tuple[list[Finding], list[Limitation]]:
    corroborated, cross_check_limitations = _cross_check(evidence)

    findings: list[Finding] = []
    sample_size_limitations: list[Limitation] = []
    for i, ev in enumerate(evidence):
        finding_id = f"finding_{i}"
        findings.append(
            Finding(
                id=finding_id,
                statement=_statement_for(ev),
                classification=_classification_for(ev.evidence_type),
                supporting_evidence=[ev.id],
                uncertainty=_extract_uncertainty(ev),
                cross_checked=ev.id in corroborated,
            )
        )
        if (
            ev.evidence_type == "STATISTICAL_RESULT"
            and ev.sample_size is not None
            and ev.sample_size < _LOW_SAMPLE_SIZE_THRESHOLD
        ):
            sample_size_limitations.append(
                Limitation(
                    category="sample_size",
                    text=f"{ev.source_tool}'s result for '{ev.metric or 'this analysis'}' is based on only {ev.sample_size} observations -- treat with caution.",
                    severity="reduces_confidence",
                    affected_findings=[finding_id],
                )
            )

    return findings, cross_check_limitations + sample_size_limitations
