# Evidence-Based Analytical Reasoning Layer — Design

Status: **PROPOSED — architecture only, not implemented.** Per explicit user instruction,
no subagent has been launched for this yet. This document is the deliverable the user
asked for before approving any implementation wave.

Not called a "Philosophy Agent" per the user's instruction — philosophy (Socratic
questioning, falsificationism, epistemic humility) is the *source* of the reasoning
principles below, not the name or framing of the component. This is an engineering
control loop with typed contracts and bounded LLM calls, not a persona.

## 1. Why this is needed (audit finding)

Read `backend/app/agent/agent.py` and `backend/app/agent/tool_router.py` in full before
writing this. Current state:

- `DataAnalystAgent.ask()` is a single flat LLM tool-calling loop. The model picks a
  tool, a tool executes deterministically, the result goes back in context, repeat until
  the model stops calling tools or `MAX_TOOL_ITERATIONS=10` is hit. This is solid
  *execution* machinery (dedup via `_canonical_signature`, stagnation-stop, the
  `[UNTRUSTED DATA]` trust boundary) but it has **no structure for reasoning** — no
  explicit claim extraction, no premise validation beyond one prose paragraph in
  `SYSTEM_PROMPT`, no hypothesis generation, no forced uncertainty reporting, no
  fact/hypothesis/assumption classification. All of that currently lives entirely in the
  LLM's own unstructured judgment, steered only by system-prompt prose.
- `tool_router.py` currently registers **10 tools**. Wave 1 built 5 more subsystems
  (`app/data/`, `app/sql/`, `app/large_data/`, `hypothesis.py`, `regression.py`) and
  Wave 2 (this session) added 5 more tool modules (`forecasting.py`, `clustering.py`,
  `segmentation.py`, `regression_diagnostics.py`, `eda.py`) — **none of these 15 tools
  are registered in `TOOL_SCHEMAS`/`_HANDLERS` yet.** The LLM cannot call any of them
  today. This is a known, already-logged gap (`integration_status.md`), not new.
- Consequence: the reasoning layer described below is being designed against a tool
  catalog that is currently much larger on disk than what's actually callable. **Wiring
  the 15 unregistered tools into `tool_router.py` is a prerequisite for the planner
  (§4) to be useful, and should happen before or in parallel with Phase 3a below** —
  see §9.

## 2. Contracts (new package `backend/app/reasoning/contracts.py`)

Pydantic models, additive, no existing schema touched. All fields JSON-safe (same
convention as `app/tools/serialization.py`).

```
AnalyticalQuestion
  raw_text: str
  intent: Literal["descriptive","diagnostic","comparative","predictive","prescriptive"]
  target_metrics: list[str]        # e.g. ["revenue"]
  implied_scope: dict              # any row-count/date-range/segment claim found in the text

Claim
  text: str
  claim_type: Literal["premise","scope","causal","comparative"]
  status: Literal["unverified","verified_true","verified_false","partially_true","unverifiable"]
  evidence_refs: list[str]         # Evidence.id, once checked

Evidence
  id: str
  source_tool: str                 # matches a ToolRouter tool name
  params: dict
  result_summary: dict             # bounded, not the raw payload
  evidence_type: Literal["FACT","CALCULATED_RESULT","STATISTICAL_RESULT"]
  tool_call_ref: str                # links to the existing ToolCallRecord

Hypothesis
  text: str
  is_causal: bool                  # explicit — forces the causal/correlation guard
  supporting_evidence: list[str]   # Evidence.id
  contradicting_evidence: list[str]
  status: Literal["supported","refuted","inconclusive"]

AnalysisPlan
  question_ref: str
  claims: list[Claim]
  planned_calls: list[dict]        # [{tool, params, rationale}], drawn only from
                                    # ToolRouter.available_tools() — planner never
                                    # invents a tool name
  competing_hypotheses: list[Hypothesis]   # populated only when intent == "diagnostic"

Uncertainty
  metric: str
  point_estimate: float | None
  interval_low: float | None
  interval_high: float | None
  confidence_level: float | None
  method: str                      # e.g. "95% CI via hypothesis.confidence_interval"

Limitation
  text: str
  severity: Literal["blocks_conclusion","reduces_confidence","minor"]
  affected_findings: list[str]     # Finding.id

Finding
  id: str
  statement: str
  classification: Literal["FACT","CALCULATED_RESULT","STATISTICAL_RESULT",
                           "HYPOTHESIS","ASSUMPTION","UNKNOWN"]
  supporting_evidence: list[str]   # Evidence.id
  uncertainty: Uncertainty | None

Recommendation
  text: str
  supporting_findings: list[str]   # Finding.id
  confidence_level: Literal["high","medium","low"]
  caveats: list[str]

AnalysisResult                      # top-level output of one reasoning pass
  question: AnalyticalQuestion
  plan: AnalysisPlan
  evidence: list[Evidence]
  findings: list[Finding]
  hypotheses: list[Hypothesis]
  uncertainties: list[Uncertainty]
  limitations: list[Limitation]
  recommendation: Recommendation | None
  final_answer_text: str
  reasoning_trace: list[str]        # ordered log of which of the 11 steps ran/skipped,
                                     # for debugging and benchmark scoring
```

These names differ slightly from the user's example list (`Claim`/`Evidence`/
`Hypothesis`/`AnalysisPlan`/`Finding`/`Uncertainty`/`Limitation`/`Recommendation` kept
verbatim; `AnalyticalQuestion` kept) because the existing codebase already has a
`ToolCallRecord` in `schemas.py` — `Evidence.tool_call_ref` links to it rather than
duplicating it, per "orchestrate thinking, don't duplicate existing tools."

## 3. Control loop → concrete steps

Mirrors the user's 11-step loop exactly. Each step names what's LLM-backed (bounded,
constrained-output) vs. pure Python (deterministic, free).

| # | Step | Implementation | Cost |
|---|---|---|---|
| 1 | Parse user question | `question_parser.py` — 1 constrained LLM call → `AnalyticalQuestion` + `Claim[]` | 1 LLM call |
| 2 | Extract claims/constraints | folded into step 1's same call | 0 extra |
| 3 | Validate premise against data | `premise_validator.py` — pure Python, reuses `profile_dataset`/`eda.automated_eda`/`analyze_cardinality` output already computed for the session; checks each `Claim.implied_scope` against real row count / `date_ranges` / column buckets | 0 LLM calls, 0 *new* tool calls (reuses profile already fetched at session start, same as today's `dataset_context`) |
| 4 | Determine required evidence | folded into step 5's plan call | 0 extra |
| 5 | Generate analysis plan | `planner.py` — 1 constrained LLM call, tool catalog = `ToolRouter.available_tools()` (planner cannot name a tool that doesn't exist — schema-enforced) → `AnalysisPlan`. Hypothesis generation (`competing_hypotheses`) is **gated**: only requested when `AnalyticalQuestion.intent == "diagnostic"` | 1 LLM call |
| 6 | Execute tools | `executor.py` — drives the **existing** `DataAnalystAgent` tool-loop machinery (`_canonical_signature`, `_wrap_tool_payload`, dedup, `MAX_TOOL_ITERATIONS`) against `plan.planned_calls`. Not reimplemented. | up to 10 tool calls (unchanged ceiling) |
| 7 | Validate results | `verifier.py` — deterministic shape/sanity checks always; a **capped** cross-check tool call (≤1 per Finding that will feed the Recommendation, only for findings flagged high-stakes by the plan) | 0–N tool calls, N ≤ number of Recommendation-feeding findings |
| 8 | Generate alternative explanations | same gate as step 5 — only for diagnostic questions; `Hypothesis[]` grounded in `Evidence[]` already gathered | 0 extra LLM calls (produced alongside step 5, refined in step 9) |
| 9 | Cross-check important findings | `verifier.py` continued — any `Finding` behind the final `Recommendation` must pass a consistency check before being classified `FACT`/`CALCULATED_RESULT` | part of step 7's budget |
| 10 | Assess uncertainty/limitations | `uncertainty.py` — pure Python, reads CI/p-value/effect-size already present in `hypothesis.py`/`regression.py`/`regression_diagnostics.py`/`forecasting.py` results; auto-generates a `Limitation` when no statistical backing exists for a claim that needs it | 0 LLM calls |
| 11 | Produce final answer | `synthesizer.py` — 1 constrained LLM call that may **only** state things present in `Finding[]`/`Recommendation`, each carrying its classification | 1 LLM call |

**Fixed LLM-call budget: 3 structured calls (parse, plan, synthesize) plus the existing
≤10-iteration tool loop, unchanged from today.** This is the answer to the user's
explicit "avoid infinite reasoning loops / unnecessary tool calls / excessive context
growth" requirement — there is no new open-ended loop anywhere in this design; every
step is either O(1) deterministic Python or one bounded, schema-constrained LLM call.

## 4. Where this sits relative to `agent.py` (explicit answer to "don't rebuild")

**`DataAnalystAgent.ask()` is not modified or replaced.** All 505 existing tests keep
passing against unchanged behavior. The reasoning layer is a new, additive package:

```
backend/app/reasoning/
    contracts.py       # §2
    question_parser.py
    premise_validator.py
    planner.py
    executor.py         # imports and drives DataAnalystAgent + ToolRouter — does not
                         # reimplement tool execution or touch app/sql, app/data,
                         # or pandas directly (same layering rule agent.py already follows)
    verifier.py
    uncertainty.py
    synthesizer.py
    orchestrator.py     # composes the 11 steps
```

New API surface (open decision, not resolved here — flagged in `decisions.md`): either
a new route (`POST /api/analyze`) or a `mode=deep` flag on the existing chat route. The
existing route/behavior is untouched either way. Routing signal into the new pipeline
can itself just be `AnalyticalQuestion.intent` — trivial descriptive lookups can stay on
the fast, cheap existing path; this is an implementation-time decision for
AGENT-ARCHITECT + REASONING-ARCHITECT, not something to pre-decide here.

Ownership (extends `agent_registry.md`'s single-writer rule): a new
**REASONING-ARCHITECT** role owns `contracts.py` + `orchestrator.py` (lands first,
sequential). Everything else in the package is owned by the Phase-3b agents in §9, each
a distinct file. `agent.py` and `tool_router.py` remain solely owned by
AGENT-ARCHITECT / TOOLING-ENGINEER as today — the reasoning layer only *calls* them.

## 5. Component reuse vs. new components

**Reused, not duplicated:**
- `profile_dataset`, `eda.automated_eda`, `eda.analyze_cardinality` — premise validation (step 3)
- `hypothesis.py` (t-test/chi-square/ANOVA/CI), `regression.py`, `regression_diagnostics.py` (VIF/heteroscedasticity/normality) — statistical backing for `Uncertainty` and the causal-language guard
- `correlation_analysis` — the exact tool the causation guard (§6) governs
- `forecasting.py`'s prediction intervals — `Uncertainty` for predictive questions
- `clustering.py`/`segmentation.py` — evidence source for diagnostic "which segment" hypotheses
- SQL layer (`app/sql/`) and `app/large_data/` — evidence sources like any other tool, once registered (§1)
- `DataAnalystAgent`'s tool loop (dedup, stagnation-stop, `_wrap_tool_payload`) — the execution substrate for step 6, called not reimplemented
- `ToolCallRecord` (`schemas.py`) — `Evidence.tool_call_ref` links here instead of a duplicate record type

**New:**
- `backend/app/reasoning/**` (package above)
- New pydantic contracts (§2)
- New API route/flag (decision pending)
- New reasoning-quality benchmark fixtures (§8) — distinct from the tool-capability
  professional benchmark already planned for QA-PROFESSIONAL-BENCHMARK-ENGINEER

## 6. The specific reasoning principles → concrete mechanisms

| Principle | Mechanism |
|---|---|
| Socratic questioning | `question_parser.py`'s output schema *forces* `target_metrics` + `implied_scope` to be populated — the LLM cannot skip stating what's being asked |
| Epistemic humility | Every `Finding` has a mandatory `classification` field (§10 below); `synthesizer.py`'s system prompt forbids stating a `HYPOTHESIS`/`ASSUMPTION` with the confidence of a `FACT` |
| Falsification | `premise_validator.py` runs *before* any evidence-gathering, checking the claim against real data, deterministically — a claim isn't just "accepted," it's assigned `verified_true`/`verified_false`/etc. before the plan proceeds |
| Alternative hypotheses | `competing_hypotheses` is populated (capped at 4, schema-enforced array length) whenever `intent == "diagnostic"` — the planner is structurally prevented from returning zero or one hypothesis for a "why" question |
| Correlation vs. causation | `Hypothesis.is_causal: bool` is a required field, not inferred after the fact — `synthesizer.py` is instructed to only use causal language when `is_causal=True` AND the supporting evidence includes something beyond `correlation_analysis` alone (e.g. a controlled comparison via `compare_periods` with a documented confound check, or an explicit user-supplied experimental design) — for purely observational correlation evidence, `is_causal` must be `False` and the synthesizer must use hedged language |
| Evidence hierarchy | `Evidence.evidence_type` (`FACT`/`CALCULATED_RESULT`/`STATISTICAL_RESULT`) is set by which tool produced it — a raw `profile_dataset` field is `FACT`, a `group_and_aggregate` sum is `CALCULATED_RESULT`, a `hypothesis.py` p-value is `STATISTICAL_RESULT`; `verifier.py` prefers higher-hierarchy evidence when two sources conflict |
| Counterfactual reasoning | Explicitly **not claimed as available** on observational data — `synthesizer.py`'s prompt states plainly that this system has no experimental/counterfactual tool; any "what would have happened without X" question gets answered as a `HYPOTHESIS`/`UNKNOWN`, never presented as computed |
| Uncertainty | `uncertainty.py` (step 10) is mandatory, not optional — every `Finding` with `classification == STATISTICAL_RESULT` must have a non-null `Uncertainty`; if the underlying tool result has no CI/p-value, a `Limitation` is auto-generated instead of silently omitting it |
| Data coverage validation | Reuses the *already-shipped* `date_ranges`/coverage-warning machinery from `agent.py`/`compare_periods` — `premise_validator.py` is the deterministic, testable version of what's currently a prose instruction in `SYSTEM_PROMPT` |
| Conclusion discipline | The `Finding.classification` enum *is* the six-way bucket the user specified verbatim (`FACT`/`CALCULATED_RESULT`/`STATISTICAL_RESULT`/`HYPOTHESIS`/`ASSUMPTION`/`UNKNOWN`) |

## 7. Failure modes and mitigations

- **Malformed structured LLM output** at any of the 3 call sites → validate against the
  pydantic schema, retry once with an error-correction system message, then fall back to
  today's plain `DataAnalystAgent.ask()` path — never a hard failure, graceful
  degradation to current behavior.
- **Planner names a tool/param that doesn't exist** → same `ToolExecutionError` path as
  today; surfaced as an `Evidence` entry with an error, doesn't crash the pipeline.
- **Hypothesis runaway** → schema caps `competing_hypotheses` at 4.
- **Verifier's cross-check disagrees with the original result** → surfaced as an
  explicit `Limitation` ("initial and cross-check results disagree: X vs Y"), never
  silently resolved by picking one — this is itself a valuable, honest output.
- **Double-counting LLM/tool budget** → total ceiling is 3 reasoning LLM calls + ≤10
  tool-loop iterations (unchanged), a concrete number for PERFORMANCE-ENGINEER (Wave 4)
  to plan latency/cost against.
- **Premise-validator false positives** (flagging a valid claim as mismatched) →
  conservative thresholds only (order-of-magnitude row-count mismatch, dates fully
  outside `date_ranges`, columns absent from all four profiler buckets) — same
  conservative bar already proven in the current `SYSTEM_PROMPT` paragraph, now made
  deterministic and unit-testable instead of LLM-judgment-based.
- **Trust-boundary regression** — any new LLM call site that sees tool-derived text
  (mainly `synthesizer.py`, since it consumes `Evidence.result_summary`) must reuse
  `_wrap_tool_payload`/`_UNTRUSTED_DATA_MARKER` exactly as `agent.py` does today, not
  reinvent it. Tested by extending `test_prompt_injection_mitigation.py`'s pattern to
  the new call sites.

## 8. Security considerations

- No new write/network side effects — reasoning layer stays read-only, so the residual
  risk noted in `backend/docs/security/prompt-injection-trust-boundary.md` ("revisit
  when a tool gains write/network capability") still does not trigger.
- `backend/app/reasoning/**` must import only from `app.tools.*`, `app.agent.tool_router`,
  `app.agent.agent` — never `app.sql`, `app.data`, or pandas directly. Same layering
  discipline `agent.py` already follows; enforceable with a simple import-lint check in
  CI later if desired.
- Structured/constrained-output LLM calls (JSON-schema-bound, not free chat) have a
  narrower injection surface than open-ended completions — a secondary benefit of the
  bounded design, not just a cost control.
- Any dataset-derived text reaching `synthesizer.py` must carry the untrusted-data
  marker — explicit new test requirement, not assumed.

## 9. Test strategy

- Unit tests per module, `MockProvider`-based (mirrors existing convention):
  `question_parser` produces correct `AnalyticalQuestion.intent`/`Claim[]` for known
  inputs; `premise_validator` correctly flags the exact "10-million-row question against
  a 4,000-row dataset" scenario from this session's original benchmark, kept as a
  permanent regression fixture; `planner` respects the diagnostic-only hypothesis gate;
  `verifier` classification logic; `uncertainty.py` correctly extracts CI from known
  `hypothesis.py`/`regression.py`/`forecasting.py` result shapes.
- Integration test: full `AnalysisResult` built end-to-end against a scripted
  `MockProvider` (deterministic, no real LLM), asserting all object linkages resolve
  (`Evidence.tool_call_ref` → real `ToolCallRecord`, `Finding.supporting_evidence` →
  real `Evidence.id`, `Recommendation.supporting_findings` → real `Finding.id`) and the
  `reasoning_trace` shows steps skipped correctly for a descriptive (non-diagnostic)
  question.
- One-time real-LLM sanity pass (same pattern as the existing prompt-injection live
  verification) — not part of CI — running a few of this session's original 7 Uzbek
  benchmark questions plus 2–3 new diagnostic ones through real Groq, hand-checking
  classification and hypothesis quality.
- Full existing 505-test suite must stay green throughout — the reasoning layer is
  purely additive and does not touch `agent.py`/`tool_router.py`.

## 10. Benchmark strategy (reasoning-quality benchmark — distinct from the
tool-capability professional benchmark already planned)

New fixture file, e.g. `backend/tests/benchmark/reasoning_questions.json`. Categories,
each with structural (not free-text) ground truth checked against `AnalysisResult`
fields directly — the payoff of having typed contracts:

1. **Direct/descriptive** — expect `intent=="descriptive"`, `hypotheses==[]`,
   findings classified `CALCULATED_RESULT`.
2. **Diagnostic/"why"** — expect `intent=="diagnostic"`, `len(hypotheses) >= 2`.
3. **Premise-mismatch** — question references a column/scale/date-range the dataset
   doesn't have; expect at least one `Claim.status in {"verified_false","unverifiable"}`
   and a `Limitation` with `severity != "minor"`.
4. **Correlation-trap** — questions phrased to invite a causal answer ("does X cause
   Y") on purely observational data; expect every `Hypothesis.is_causal == False` and
   no unhedged causal language in `final_answer_text` (checked via a forbidden-phrase
   list: "causes", "because of", "leads to", "due to" absent unless hedged).
5. **Uncertainty-required** — forecast/statistical-test questions; expect a non-null
   `Uncertainty` on the relevant `Finding`.
6. **Unanswerable** — the claim genuinely can't be supported by the data; expect
   `classification=="UNKNOWN"` on the relevant finding and a refusal in
   `final_answer_text` rather than a fabricated number.

This benchmark complements, not replaces, QA-PROFESSIONAL-BENCHMARK-ENGINEER's planned
100–150-task tool-capability benchmark ("can it compute the right number"). This one
measures "does it reason soundly about what the number means." Both scores should be
reported together once both exist.

## 11. Parallel subagent plan (Wave 3 proposal — NOT launched)

Not fully parallel like Wave 1/2, because every downstream file depends on `contracts.py`
existing first — same shape as Wave 1's DATA-ARCHITECT-first pattern.

**Phase 3a (sequential, blocking, one agent):**
- **REASONING-ARCHITECT** — lands `backend/app/reasoning/contracts.py` +
  `orchestrator.py` skeleton (steps wired as stubs) + resolves the new-route-vs-flag
  API decision. Must be reviewed/merged before 3b starts.

**Phase 3b (parallel, once contracts.py exists — no file overlap):**
- **QUESTION-PARSING-ENGINEER** — `question_parser.py` + `premise_validator.py`
- **PLANNING-ENGINEER** — `planner.py` (incl. hypothesis gating)
- **VERIFICATION-ENGINEER** — `verifier.py` + `uncertainty.py`
- **SYNTHESIS-ENGINEER** — `synthesizer.py`
- **REASONING-QA-ENGINEER** — cross-cutting tests + the reasoning-quality benchmark fixtures (§10)
- **REASONING-SECURITY-ENGINEER** — review-only, checks every new LLM call site against §8

**Phase 3c (sequential, blocking, after 3b lands):**
- Orchestrator (me, directly — per the Wave 1 lesson that cross-cutting wiring benefits
  from single-owner care) wires `executor.py` to the real `DataAnalystAgent`/
  `ToolRouter`, adds the API route, runs full regression + the new reasoning benchmark,
  reconciles any contract drift.

**Independent prerequisite, not blocked by any of the above:**
- **TOOLING-ENGINEER** — register the 15 already-built Wave 1+2 tools into
  `tool_router.py` (§1). Small, mechanical, already-anticipated by every Wave 1/2
  agent's final report. The Phase 3b **planner** genuinely needs this done first to have
  a useful tool catalog to plan against — recommend running this *before or alongside*
  Phase 3a, not after.

## 12. Estimated implementation sequence

1. TOOLING-ENGINEER registers the 15 unregistered tools (small, unblocks everything, no reasoning-layer dependency).
2. Phase 3a: REASONING-ARCHITECT lands contracts + orchestrator skeleton + API decision.
3. Phase 3b: 6 parallel subagents build the pipeline stages + tests + benchmark fixtures.
4. Phase 3c: orchestrator integrates, runs full regression + reasoning benchmark, reports two new headline numbers (reasoning-benchmark score, updated total test count) alongside the tool-capability benchmark from Wave 2's QA-PROFESSIONAL-BENCHMARK-ENGINEER (not yet launched — on hold, see roadmap.md).
5. Wave 4 (already on roadmap) absorbs latency/cost tuning for the reasoning layer's added LLM calls.

STOP HERE per user instruction — no subagent for this wave has been launched. Waiting
for explicit orchestrator/user approval before Phase 3a starts.
