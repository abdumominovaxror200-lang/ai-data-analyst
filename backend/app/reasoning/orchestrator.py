"""Reasoning pipeline orchestrator (Phase 3B.2/3B.7, extended Phase 4).

Composes the bounded, 3-structured-LLM-call pipeline:

    parse question (LLM call 1)
        -> validate premise (deterministic)
        -> [early stop: requested metric/dimension does not exist]
    plan analysis + select capability categories (LLM call 2)
        -> [early stop: no applicable capability category]
    execute plan via the EXISTING agent/tool_router loop (no new tool engine)
        -> build findings + cross-check (deterministic)
        -> derive hypothesis status from evidence (Phase 4 P1, deterministic)
    synthesize final answer (LLM call 3, includes the causation guard)
        -> strengthen/cap recommendation confidence by evidence strength (Phase 4 P0,
           deterministic)
        -> run machine-checkable epistemic-principle checks (Phase 4 P2, deterministic)

No stage here duplicates tool-execution logic (`executor.py` drives the existing
`DataAnalystAgent`), and no stage adds a second unbounded loop -- the two early-stop
branches use only 2 of the 3 reasoning calls (parse + synthesize), never more than the
documented 3 regardless of path taken. Every Phase 4 addition below is deterministic
(no new LLM call, no new tool call) -- it post-processes objects the existing 3 calls
already produced.
"""

from __future__ import annotations

import logging

from app.agent.providers import LLMProvider
from app.agent.tool_router import ToolRouter
from app.datasets.storage import DatasetRecord
from app.reasoning import confound_detection, contradiction_detection, epistemic_checks, executor, hypothesis_evaluator, planner, question_parser, verifier
from app.reasoning.analytical_audit import build_analytical_audit
from app.reasoning.coverage import CoverageAssessment, assess_coverage
from app.reasoning.conclusion_guard import sanitize_blocked_hypotheses
from app.reasoning.contracts import AnalysisResult, Finding, Limitation
from app.reasoning.premise_validator import validate_question
from app.reasoning.recommendation_grounding import evaluate_recommendation_grounding
from app.reasoning.synthesizer import synthesize
from app.tools.profiler import profile_dataset

logger = logging.getLogger(__name__)


class ReasoningOrchestrator:
    def __init__(self, provider: LLMProvider, tool_router: ToolRouter | None = None) -> None:
        self._provider = provider
        self._router = tool_router or ToolRouter()
        self.last_coverage: CoverageAssessment | None = None

    def analyze(self, record: DatasetRecord, question_text: str) -> AnalysisResult:
        trace: list[str] = []

        # --- LLM call 1: parse question + extract claims ---
        question, parsed_claims = question_parser.parse_question(
            self._provider, question_text, _dataset_summary(record)
        )
        trace.append(f"parsed question: intent={question.intent}")

        # --- deterministic: validate premise against the real data ---
        validation_claims, limitations, _profile = validate_question(question, record.df)
        claims = parsed_claims + validation_claims
        trace.append(f"premise validation: {len(limitations)} limitation(s) found")

        if _requested_field_missing(limitations):
            return self._stop_early_missing_field(question, claims, limitations, trace)

        # --- LLM call 2: analysis plan + capability-category selection ---
        plan = planner.plan_analysis(self._provider, question, claims, limitations)
        trace.append(f"plan: categories={plan.capability_categories}")

        if not plan.capability_categories:
            trace.append("stopping: no applicable capability category for this question")
            return self._stop_unavailable_capability(question, claims, plan, limitations, trace)

        # --- execute: existing agent/tool_router loop, unmodified ---
        evidence, _raw_narrative, executed_tools = executor.execute_plan(
            self._provider, record, question_text, plan, self._router
        )
        trace.append(f"execution: {len(evidence)} evidence item(s) gathered")

        if question.intent == "diagnostic":
            initial_coverage = assess_coverage(
                question, plan, evidence,
                date_columns=_profile["date_columns"], categorical_columns=_profile.get("categorical_columns", []), executed_tools=executed_tools,
                recovery_finished=False,
            )
            if initial_coverage.recovery_targets:
                trace.append("coverage recovery targets: " + ", ".join(initial_coverage.recovery_targets))
                recovered, recovery_attempts = executor.execute_recovery(
                    self._provider, record, question_text, initial_coverage.recovery_targets,
                    self._router, evidence_offset=len(evidence),
                )
                evidence = evidence + recovered
                executed_tools = executed_tools + recovery_attempts
                trace.append(f"coverage recovery: {len(recovered)} evidence item(s) gathered")
            self.last_coverage = assess_coverage(
                question, plan, evidence,
                date_columns=_profile["date_columns"], categorical_columns=_profile.get("categorical_columns", []), executed_tools=executed_tools,
                recovery_finished=True,
            )
            trace.extend(
                f"tool coverage: {item.tool_name}={item.stage} ({item.reason})"
                for item in self.last_coverage.tools
            )
            if not self.last_coverage.complete:
                gap = self.last_coverage.gap_explanation()
                limitations = limitations + [Limitation(
                    category="insufficient_coverage",
                    text="Required RCA coverage is unresolved: " + gap,
                    severity="blocks_conclusion",
                )]
                trace.append("coverage blocked: " + gap)
        else:
            self.last_coverage = assess_coverage(
                question, plan, evidence,
                date_columns=_profile["date_columns"], categorical_columns=_profile.get("categorical_columns", []), executed_tools=executed_tools,
                recovery_finished=False,
            )
        if not evidence:
            limitations = limitations + [
                Limitation(
                    category="missing_data",
                    text="No tool produced usable evidence for this question with the selected capabilities.",
                    severity="reduces_confidence",
                )
            ]

        # --- deterministic: validate results, cross-check, classify findings ---
        findings, verifier_limitations = verifier.build_findings(evidence)
        limitations = limitations + verifier_limitations
        trace.append(
            f"findings: {len(findings)}, cross-checked: {sum(1 for f in findings if f.cross_checked)}"
        )

        # --- deterministic (Phase 5): confounding-variable detection -- runs against
        # the real dataset (not just Evidence), so it lives here rather than in
        # verifier.py, which never sees record.df. See confound_detection.py's module
        # docstring for the real live-model failure that motivated this.
        confound_limitations = confound_detection.detect_confounds(record.df, evidence)
        if confound_limitations:
            limitations = limitations + confound_limitations
            trace.append(f"confound check: {len(confound_limitations)} possible confound(s) flagged")

        # --- deterministic (Contradiction Engine, now 3 checks -- see
        # contradiction_detection.py's module docstring): a mean-vs-median (or any
        # two differing aggregations) ranking contradiction, an overall-vs-subgroup
        # direction reversal, and conflicting data-quality signals across the same
        # scope, all over the same group comparison / dataset.
        contradiction_limitations = (
            contradiction_detection.detect_ranking_contradictions(evidence)
            + contradiction_detection.detect_overall_vs_subgroup_contradiction(evidence)
            + contradiction_detection.detect_data_quality_contradictions(evidence)
        )
        if contradiction_limitations:
            limitations = limitations + contradiction_limitations
            trace.append(f"contradiction check: {len(contradiction_limitations)} contradiction(s) flagged")

        # --- deterministic (Phase 4 P1): derive hypothesis status from gathered
        # evidence -- never from the LLM declaring itself "supported". This is what
        # makes the causation guard's "a supported causal hypothesis may use unhedged
        # language" branch reachable in production for the first time.
        hypotheses = hypothesis_evaluator.update_hypothesis_status(plan.hypotheses, evidence, findings)
        hypotheses = sanitize_blocked_hypotheses(hypotheses, limitations)
        if hypotheses:
            trace.append(f"hypotheses: {[(h.id, h.status) for h in hypotheses]}")

        # --- LLM call 3: synthesize final answer (includes the causation guard) ---
        final_text, recommendation, was_hedged, matched_phrases = synthesize(
            self._provider, question, claims, plan, evidence, findings, hypotheses, limitations
        )
        if was_hedged:
            trace.append(f"causation guard: hedged unsupported causal language {matched_phrases}")

        # --- deterministic (Phase 4 P0): cap recommendation confidence at what the
        # evidence actually supports -- never trust the LLM's own stated confidence.
        if question.intent == "diagnostic" and self.last_coverage and not self.last_coverage.complete:
            if recommendation is not None:
                trace.append("recommendation withheld: required RCA coverage is unresolved")
            recommendation = None

        grounding = None
        if recommendation is not None:
            grounding = evaluate_recommendation_grounding(recommendation, findings, evidence, hypotheses, limitations)
            if grounding.violations:
                trace.append(f"recommendation grounding violations: {grounding.violations}")
            if grounding.adjusted_confidence != recommendation.confidence:
                trace.append(
                    f"recommendation confidence capped: {recommendation.confidence!r} -> "
                    f"{grounding.adjusted_confidence!r} (evidence strength: {grounding.evidence_strength})"
                )
                recommendation = recommendation.model_copy(update={"confidence": grounding.adjusted_confidence})

        trace.append("stopping: synthesis complete, sufficient evidence gathered or exhausted")

        # --- deterministic (Phase 4 P2): machine-checkable epistemic-principle audit ---
        violations = epistemic_checks.check_all(
            question, claims, findings, evidence, hypotheses, limitations, recommendation, final_text
        )

        # --- deterministic (v2 reliability mission, Phase 3/5): assemble the
        # structured AnalyticalAudit from the pieces already computed above.
        audit = build_analytical_audit(
            limitations, confound_limitations, contradiction_limitations, verifier_limitations,
            grounding, was_hedged, matched_phrases,
        )
        trace.append(f"analytical audit: conclusion_status={audit.conclusion_status}")

        return AnalysisResult(
            question=question,
            claims=claims,
            plan=plan,
            evidence=evidence,
            findings=findings,
            hypotheses=hypotheses,
            limitations=limitations,
            recommendation=recommendation,
            final_answer_text=final_text,
            reasoning_trace=trace,
            principle_violations=violations,
            analytical_audit=audit,
        )

    # --- early-stop branches (Phase 3B.7) -----------------------------------------

    def _stop_early_missing_field(self, question, claims, limitations, trace) -> AnalysisResult:
        trace.append("stopping: a requested metric or dimension does not exist in this dataset")
        finding = Finding(
            id="finding_0",
            statement="The requested metric or dimension does not exist in this dataset.",
            classification="UNKNOWN",
        )
        final_text, _rec, _hedged, _matched = synthesize(
            self._provider, question, claims, None, [], [finding], [], limitations
        )
        violations = epistemic_checks.check_all(question, claims, [finding], [], [], limitations, None, final_text)
        audit = build_analytical_audit(limitations, [], [], [], None, False, [])
        return AnalysisResult(
            question=question,
            claims=claims,
            plan=None,
            evidence=[],
            findings=[finding],
            hypotheses=[],
            limitations=limitations,
            recommendation=None,
            final_answer_text=final_text,
            reasoning_trace=trace,
            principle_violations=violations,
            analytical_audit=audit,
        )

    def _stop_unavailable_capability(self, question, claims, plan, limitations, trace) -> AnalysisResult:
        finding = Finding(
            id="finding_0",
            statement="No analytical capability in this system can address this question.",
            classification="UNKNOWN",
        )
        unavailable = Limitation(
            category="unavailable_capability",
            text="No analytical capability category applicable to this question was found.",
            severity="blocks_conclusion",
        )
        all_limitations = limitations + [unavailable]
        final_text, _rec, _hedged, _matched = synthesize(
            self._provider, question, claims, plan, [], [finding], [], all_limitations
        )
        violations = epistemic_checks.check_all(
            question, claims, [finding], [], list(plan.hypotheses), all_limitations, None, final_text
        )
        audit = build_analytical_audit(all_limitations, [], [], [], None, False, [])
        return AnalysisResult(
            question=question,
            claims=claims,
            plan=plan,
            evidence=[],
            findings=[finding],
            hypotheses=sanitize_blocked_hypotheses(list(plan.hypotheses), all_limitations),
            limitations=all_limitations,
            recommendation=None,
            final_answer_text=final_text,
            reasoning_trace=trace,
            principle_violations=violations,
            analytical_audit=audit,
        )


def _requested_field_missing(limitations: list[Limitation]) -> bool:
    return any(l.category == "missing_data" for l in limitations)


def _dataset_summary(record: DatasetRecord) -> str:
    profile = profile_dataset(record.df)
    date_coverage = (
        ", ".join(f"{col} spans {rng['min']} to {rng['max']}" for col, rng in profile["date_ranges"].items())
        or "none"
    )
    other_columns = profile["text_columns"] + profile["boolean_columns"]
    return (
        f"{profile['rows']} rows, {profile['columns']} columns. "
        f"Numeric columns: {profile['numeric_columns']}. "
        f"Categorical columns: {profile['categorical_columns']}. "
        f"Date columns: {profile['date_columns']} (coverage: {date_coverage}). "
        f"Other columns (IDs, text, booleans): {other_columns}."
    )
