"""Phase 3C Part E: deterministic structural scoring framework for the professional
analyst benchmark.

Built once, centrally, before the parallel benchmark-authoring wave (BENCHMARK-ENGINEER,
QA-BENCHMARK-ENGINEER) starts -- this is the shared contract both agents author cases
against, so their work is compatible without either one seeing the other's worktree
(the same "foundation lands first" lesson from every previous wave in this project;
see .agent/decisions.md).

Design principle (Phase 3C Part F: no Groq dependency for the full suite): a case can
supply an optional `script` describing exactly what a scripted `MockProvider` should
return at each of the reasoning layer's 3 structured LLM calls plus its tool-execution
rounds. `build_mock_provider_from_script` turns that into a real `MockProvider`, and
`run_case` drives the REAL `ReasoningOrchestrator` against it -- so what's being tested
is whether the deterministic scaffolding (premise validation, category filtering,
finding classification, causation guard, evidence traceability) correctly turns a given
LLM behavior into the right structural result. This is honest and Groq-independent:
the script represents "a plausible LLM response for this case" (authored by the
benchmark writer as the case's assumption), not something the scoring framework
invents to guarantee a pass.

A case without a `script` still gets the deterministic-only checks that don't need any
LLM call at all (constraint/limitation detection via `premise_validator`, category
resolution via `categories.py`) -- narrower coverage, but still real and honest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from app.agent.providers import MockProvider, ProviderResponse, ToolCall
from app.datasets.storage import DatasetRecord
from app.reasoning import causation_guard
from app.reasoning.contracts import AnalysisResult, AnalyticalQuestion
from app.reasoning.orchestrator import ReasoningOrchestrator
from app.reasoning.premise_validator import validate_question

Verdict = Literal["PASS", "PARTIAL", "FAIL"]


@dataclass
class CheckResult:
    name: str
    passed: bool | None  # None = not applicable to this case
    detail: str = ""


@dataclass
class CaseResult:
    case_id: str
    verdict: Verdict
    checks: list[CheckResult] = field(default_factory=list)
    explanation: str = ""
    result: AnalysisResult | None = None  # None if the case had no script (deterministic-only)


# --- Building a MockProvider from a case's `script` block -----------------------


def build_mock_provider_from_script(script: dict) -> MockProvider:
    """`script` shape (all keys except `parsed_question` and `final_answer_text` are
    optional, to support cases that intentionally stop early):

    {
      "parsed_question": {<AnalyticalQuestion-shaped dict, matches question_parser's
                            expected raw JSON: intent, requested_metrics, ...,
                            "claims": [{"text":..., "source":...}]>},
      "plan": {<AnalysisPlan-shaped dict: objective, capability_categories, steps,
                 tools_required, expected_outputs, validation_steps,
                 stopping_conditions, hypotheses>} or omitted entirely if the case
               is expected to stop before planning (e.g. a missing-column case),
      "tool_calls": [{"tool": "t_test", "arguments": {...}}, ...] or omitted/empty
                     if the case is expected to stop before execution (e.g. an
                     unavailable-capability case, which still reaches planning),
      "final_answer_text": "...",
      "recommendation": {...} or null
    }
    """
    responses: list[ProviderResponse] = [ProviderResponse(content=json.dumps(script["parsed_question"]))]

    if "plan" in script and script["plan"] is not None:
        responses.append(ProviderResponse(content=json.dumps(script["plan"])))
        tool_calls = script.get("tool_calls") or []
        for i, call in enumerate(tool_calls):
            responses.append(
                ProviderResponse(
                    content=None,
                    tool_calls=[ToolCall(id=f"call_{i}", name=call["tool"], arguments=call["arguments"])],
                )
            )
        if tool_calls:
            responses.append(ProviderResponse(content="evidence gathered"))

    responses.append(
        ProviderResponse(
            content=json.dumps(
                {"final_answer_text": script["final_answer_text"], "recommendation": script.get("recommendation")}
            )
        )
    )
    return MockProvider(responses)


# --- Running a case ----------------------------------------------------------------


def run_case(case: dict, record: DatasetRecord) -> CaseResult:
    checks: list[CheckResult] = []

    if case.get("script"):
        provider = build_mock_provider_from_script(case["script"])
        orchestrator = ReasoningOrchestrator(provider)
        result = orchestrator.analyze(record, case["user_question"])
        checks.extend(_structural_checks(case, result))
    else:
        result = None
        checks.extend(_deterministic_only_checks(case, record))

    verdict = _verdict(checks)
    explanation = _explain(checks)
    return CaseResult(case_id=case["case_id"], verdict=verdict, checks=checks, explanation=explanation, result=result)


def _verdict(checks: list[CheckResult]) -> Verdict:
    applicable = [c for c in checks if c.passed is not None]
    if not applicable:
        return "PARTIAL"
    if all(c.passed for c in applicable):
        return "PASS"
    if any(c.passed for c in applicable):
        return "PARTIAL"
    return "FAIL"


def _explain(checks: list[CheckResult]) -> str:
    failed = [c for c in checks if c.passed is False]
    if not failed:
        return "All applicable structural checks passed."
    return "Failed: " + "; ".join(f"{c.name} ({c.detail})" for c in failed)


# --- Deterministic-only checks (no script, no LLM call) ----------------------------


def _deterministic_only_checks(case: dict, record: DatasetRecord) -> list[CheckResult]:
    checks = []
    question = AnalyticalQuestion(
        original_question=case["user_question"],
        intent="descriptive",
        requested_metrics=case.get("_probe_metrics", []),
        requested_dimensions=case.get("_probe_dimensions", []),
        requested_time_range=case.get("_probe_time_range"),
        explicit_constraints=case.get("_probe_scale_constraints", []),
    )
    claims, limitations, _profile = validate_question(question, record.df)

    if case["required_constraints"]:
        found = any(_keyword_overlap(rc, [c.text + " " + (c.note or "") for c in claims]) for rc in case["required_constraints"])
        checks.append(CheckResult("correct constraints detected", found, "no matching claim/note found" if not found else ""))

    if case["expected_limitations"]:
        expected_categories = {el["category"] for el in case["expected_limitations"]}
        actual_categories = {l.category for l in limitations}
        ok = expected_categories.issubset(actual_categories)
        checks.append(
            CheckResult(
                "correct limitation",
                ok,
                f"expected {expected_categories}, got {actual_categories}" if not ok else "",
            )
        )

    from app.reasoning.categories import valid_categories

    if case["expected_tool_category"]:
        resolved = set(valid_categories(case["expected_tool_category"]))
        ok = resolved == set(case["expected_tool_category"])
        checks.append(CheckResult("correct tool category", ok, "case references an unknown category" if not ok else ""))

    return checks


def _keyword_overlap(needle: str, haystacks: list[str]) -> bool:
    needle_words = {w.lower() for w in needle.split() if len(w) > 3}
    for h in haystacks:
        h_words = {w.lower() for w in h.split()}
        if needle_words & h_words:
            return True
    return False


# --- Full structural checks (script-driven, real ReasoningOrchestrator run) --------


def _structural_checks(case: dict, result: AnalysisResult) -> list[CheckResult]:
    checks: list[CheckResult] = []

    # 1. correct capability selected / 3. correct tool category (same underlying
    # signal in this codebase, since a ToolCategory IS the capability -- see
    # app/reasoning/categories.py)
    if case["expected_tool_category"]:
        actual = set(result.plan.capability_categories) if result.plan else set()
        expected = set(case["expected_tool_category"])
        ok = bool(actual & expected)
        checks.append(CheckResult("correct capability/tool category selected", ok, f"expected one of {expected}, got {actual}" if not ok else ""))
    elif result.plan is not None:
        # case expects NO capability (unavailable-capability scenario)
        ok = not result.plan.capability_categories
        checks.append(CheckResult("correctly found no applicable capability", ok, f"expected none, got {result.plan.capability_categories}" if not ok else ""))

    # 2. correct constraints detected
    if case["required_constraints"]:
        haystacks = [c.text + " " + (c.note or "") for c in result.claims] + [l.text for l in result.limitations]
        found = any(_keyword_overlap(rc, haystacks) for rc in case["required_constraints"])
        checks.append(CheckResult("correct constraints detected", found, "no matching claim/limitation found" if not found else ""))

    # 4. correct numerical result (optional field)
    if case.get("expected_numeric_result"):
        spec = case["expected_numeric_result"]
        found_value = None
        for ev in result.evidence:
            if spec["field"] in ev.result_summary:
                found_value = ev.result_summary[spec["field"]]
                break
        if found_value is None:
            checks.append(CheckResult("correct numerical result", False, f"field '{spec['field']}' not found in any evidence"))
        else:
            ok = abs(float(found_value) - float(spec["value"])) <= spec.get("tolerance", 0.01)
            checks.append(CheckResult("correct numerical result", ok, f"expected ~{spec['value']}, got {found_value}" if not ok else ""))

    # 5. correct finding classification
    if case["expected_classifications"]:
        actual = {f.classification for f in result.findings}
        expected = set(case["expected_classifications"])
        ok = expected.issubset(actual)
        checks.append(CheckResult("correct finding classification", ok, f"expected {expected}, got {actual}" if not ok else ""))

    # 6. correct limitation
    if case["expected_limitations"]:
        expected_categories = {el["category"] for el in case["expected_limitations"]}
        actual_categories = {l.category for l in result.limitations}
        ok = expected_categories.issubset(actual_categories)
        checks.append(CheckResult("correct limitation", ok, f"expected {expected_categories}, got {actual_categories}" if not ok else ""))

    # 7. correct causal language
    behavior = case["expected_causal_behavior"]
    if behavior == "must_hedge_unless_causal_hypothesis_supported":
        remaining = causation_guard.find_causal_phrases(result.final_answer_text)
        ok = not remaining
        checks.append(CheckResult("correct causal language", ok, f"unhedged causal phrase(s) remained: {remaining}" if not ok else ""))
    elif behavior == "must_generate_2_to_4_competing_hypotheses":
        ok = 2 <= len(result.hypotheses) <= 4
        checks.append(CheckResult("correct causal language", ok, f"expected 2-4 hypotheses, got {len(result.hypotheses)}" if not ok else ""))

    # 8. evidence traceability
    evidence_ids = {e.id for e in result.evidence}
    finding_ids = {f.id for f in result.findings}
    traceable = all(ev_id in evidence_ids for f in result.findings for ev_id in f.supporting_evidence)
    if result.recommendation:
        traceable = traceable and all(fid in finding_ids for fid in result.recommendation.supporting_findings)
    checks.append(CheckResult("evidence traceability", traceable, "" if traceable else "a finding or recommendation referenced a non-existent id"))

    # 9. unsupported claims: nothing FACT/CALCULATED_RESULT/STATISTICAL_RESULT without backing evidence
    unsupported = [
        f for f in result.findings if f.classification in ("FACT", "CALCULATED_RESULT", "STATISTICAL_RESULT") and not f.supporting_evidence
    ]
    checks.append(CheckResult("no unsupported solid claims", not unsupported, f"{len(unsupported)} finding(s) with no supporting evidence" if unsupported else ""))

    # 10. recommendation grounding
    if result.recommendation:
        grounded = bool(result.recommendation.supporting_findings) or result.recommendation.confidence is None
        checks.append(CheckResult("recommendation grounding", grounded, "recommendation has neither supporting findings nor an honest null confidence" if not grounded else ""))

    return checks


# --- Report aggregation (Phase 3C Part G) -------------------------------------------


def summarize(case_results: list[CaseResult], cases_by_id: dict[str, dict]) -> dict[str, Any]:
    total = len(case_results)
    passed = sum(1 for r in case_results if r.verdict == "PASS")
    partial = sum(1 for r in case_results if r.verdict == "PARTIAL")
    failed = sum(1 for r in case_results if r.verdict == "FAIL")
    overall_pct = round(100.0 * passed / total, 1) if total else 0.0

    by_category: dict[str, list[CaseResult]] = {}
    for r in case_results:
        cat = cases_by_id[r.case_id]["category"]
        by_category.setdefault(cat, []).append(r)

    category_scores = {
        cat: round(100.0 * sum(1 for r in results if r.verdict == "PASS") / len(results), 1)
        for cat, results in by_category.items()
    }

    return {
        "total_tasks": total,
        "passed": passed,
        "partial": partial,
        "failed": failed,
        "overall_score_pct": overall_pct,
        "category_scores_pct": category_scores,
        "cases": [
            {"case_id": r.case_id, "verdict": r.verdict, "explanation": r.explanation}
            for r in case_results
        ],
    }
