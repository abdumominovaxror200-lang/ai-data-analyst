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

from typing import Callable

from app.reasoning.contracts import Evidence, Finding, FindingClassification, Limitation, Uncertainty
from app.reasoning.numerical_sanity import check_numerical_sanity

_CROSS_CHECK_RELATIVE_TOLERANCE = 0.10  # 10%: close enough to call "agreeing"
_LOW_SAMPLE_SIZE_THRESHOLD = 10
# A single extreme value (or a small handful) pulling the max this many standard
# deviations above the mean is a strong, cheap signal the mean is not representative
# -- found via real-LLM adversarial testing: a mean skewed by one $80,000 outlier
# among ~$800 typical orders was reported unhedged, with no outlier mention, because
# nothing forced the check; the LLM simply didn't choose to call detect_anomalies.
# This makes the check unconditional -- it reads describe_data's own already-computed
# mean/std/max (no extra tool call) rather than depending on the model electing to
# investigate further.
_OUTLIER_RATIO_THRESHOLD = 4.0


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
    """Reads a tool's point ESTIMATE of the metric's actual value -- deliberately
    excludes "statistic" (a test statistic like a t-value or F-value), which is not
    a value of the metric at all and is not on a comparable scale across different
    tools/tests. Found as a real bug via a real (unscripted) LLM run: a two-sample
    t_test call (no top-level "mean", only "statistic") and a confidence_interval
    call on the same metric got flagged as "disagreeing" because 7.1 (the
    t-statistic) was compared against $143 (the actual mean) as if they were the
    same kind of number -- a confusing, incorrect cross-check false positive, not a
    real disagreement."""
    r = evidence.result_summary
    for key in ("mean", "value", "coefficient"):
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


# Verification-style tools and how to read "this tool found no DISQUALIFYING problem"
# directly off each one's own real, unmodified return shape (field names confirmed
# against app/tools/anomaly.py and app/tools/data_quality.py, not guessed):
#   - duplicate_analysis:    duplicate_row_count == 0 (binary -- any exact duplicate
#                            is a real, unambiguous problem)
#   - data_quality_report:   quality_issues is empty (equivalently quality_score == 100)
#   - detect_anomalies:      anomaly_pct below _ANOMALY_CLEAN_THRESHOLD_PCT, NOT a
#                            literal zero -- IQR-based outlier detection on any
#                            moderately-skewed real distribution (revenue, deal size,
#                            etc.) routinely flags a nonzero baseline percentage with
#                            no genuine data-quality problem behind it (verified
#                            directly: a real run against this project's own demo
#                            dataset, filtered to one region, flagged 6.81% of rows via
#                            IQR on ordinary right-skewed revenue -- a literal
#                            `== 0` bar would almost never be satisfied on real data,
#                            making this corroboration signal nearly useless in
#                            practice). A double-digit anomaly rate is still a real,
#                            disqualifying signal; a single-digit one is ordinary noise.
_ANOMALY_CLEAN_THRESHOLD_PCT = 15.0
_VERIFICATION_TOOLS_CLEAN_CHECK: dict[str, Callable[[dict], bool]] = {
    "detect_anomalies": lambda r: (r.get("anomaly_pct") or 0.0) < _ANOMALY_CLEAN_THRESHOLD_PCT,
    "duplicate_analysis": lambda r: r.get("duplicate_row_count") == 0,
    "data_quality_report": lambda r: not r.get("quality_issues"),
}


def _investigation_cross_check(evidence: list[Evidence]) -> set[str]:
    """Broader corroboration than `_cross_check`'s literal-scalar-agreement check.

    Real gap found via the hard real-world benchmark (.agent/hard_realworld_benchmark.md
    finding #1): a genuine multi-step root-cause investigation -- e.g. `compare_periods`
    (does revenue look different?) + `group_and_aggregate` (is one category driving it?)
    + `detect_anomalies` (is a data artifact driving it?), all on the same metric --
    never produces two evidence items with a directly comparable flat scalar (`mean`/
    `value`/`coefficient`/`statistic`), so `_cross_check` never marks any of them
    `cross_checked=True`, even though the investigation is real, evidence-grounded, and
    exactly the "check outliers / check data quality before accepting a conclusion"
    discipline this project's own reasoning principles require.

    This function formalizes that discipline as an explicit, deterministic corroboration
    rule: when an independent verification tool (`detect_anomalies`/`duplicate_analysis`/
    `data_quality_report`) examined the SAME metric as another analytical tool and found
    no problem, that is real corroboration for the investigation as a whole -- distinct
    from (and additive to, never a replacement for) `_cross_check`'s narrower "two tools
    computed the same number" check, which remains unchanged and still catches genuine
    numeric disagreements `_investigation_cross_check` cannot see."""
    corroborated: set[str] = set()
    by_metric: dict[str, list[Evidence]] = {}
    for e in evidence:
        if e.metric:
            by_metric.setdefault(e.metric, []).append(e)

    for items in by_metric.values():
        distinct_tools = {e.source_tool: e for e in items}
        if len(distinct_tools) < 2:
            continue
        verification_ran_clean = [
            e for e in distinct_tools.values()
            if e.source_tool in _VERIFICATION_TOOLS_CLEAN_CHECK
            and _VERIFICATION_TOOLS_CLEAN_CHECK[e.source_tool](e.result_summary)
        ]
        other_analytical = [e for e in distinct_tools.values() if e.source_tool not in _VERIFICATION_TOOLS_CLEAN_CHECK]
        if verification_ran_clean and other_analytical:
            corroborated.update(e.id for e in distinct_tools.values())
    return corroborated


def _describe_data_outlier_limitations(evidence: list[Evidence]) -> list[Limitation]:
    """Deterministically flags a describe_data-reported mean at risk of being
    dominated by extreme outlier value(s), independent of whether the model chose to
    call detect_anomalies. Reads only fields describe_data already computed -- no
    extra tool call."""
    limitations: list[Limitation] = []
    for i, ev in enumerate(evidence):
        if ev.source_tool != "describe_data":
            continue
        columns = ev.result_summary.get("columns")
        if not isinstance(columns, dict):
            continue
        flagged_cols = []
        for col, stats in columns.items():
            if not isinstance(stats, dict):
                continue
            mean, std, max_v = stats.get("mean"), stats.get("std"), stats.get("max")
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (mean, std, max_v)):
                continue
            if not std:  # zero, None already excluded above, but guard div-by-zero explicitly
                continue
            if (max_v - mean) / std >= _OUTLIER_RATIO_THRESHOLD:
                flagged_cols.append(col)
        if flagged_cols:
            limitations.append(
                Limitation(
                    category="methodological",
                    text=(
                        f"describe_data's mean for {', '.join(flagged_cols)} may be distorted by an extreme "
                        f"outlier value (the maximum is at least {_OUTLIER_RATIO_THRESHOLD:.0f} standard "
                        "deviations above the mean) -- consider the median, or check for outliers, before "
                        "treating the mean as representative."
                    ),
                    severity="reduces_confidence",
                    affected_findings=[f"finding_{i}"],
                )
            )
    return limitations


def build_findings(evidence: list[Evidence]) -> tuple[list[Finding], list[Limitation]]:
    corroborated, cross_check_limitations = _cross_check(evidence)
    corroborated = corroborated | _investigation_cross_check(evidence)

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

    outlier_limitations = _describe_data_outlier_limitations(evidence)
    numerical_sanity_limitations = check_numerical_sanity(evidence)

    return findings, cross_check_limitations + sample_size_limitations + outlier_limitations + numerical_sanity_limitations
