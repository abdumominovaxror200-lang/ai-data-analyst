# AI Data Analyst v2 — Baseline Report (before this mission's changes)

Written per this mission's explicit "preserve baseline" requirement, before any code
changes. HEAD: `246dba2`. Working tree clean (only an untracked, unrelated local
`.claude/` directory).

## Current test count

Full backend suite: **1020 passed, 74 skipped (real-LLM/100M-row gated), 0 failed**
(verified by a fresh run this session, not recalled from memory).

## Current benchmark scores

- Hard real-world benchmark (`tests/benchmark/hard_realworld_results.json`): **100.0%,
  102/102 PASS, 0 PARTIAL, 0 FAIL**.
- `final_100`, `professional`, `adversarial` benchmark suites: all passing, unchanged
  since the last mission.

## Current architecture (the 3-call reasoning pipeline)

```
parse question (LLM call 1: question_parser.py)
    -> validate premise (deterministic: premise_validator.py)
    -> [early stop: requested metric/dimension does not exist]
plan analysis + select capability categories (LLM call 2: planner.py)
    -> [early stop: no applicable capability category]
execute plan via the existing agent/tool_router loop (executor.py, no new tool engine)
    -> build findings + cross-check (deterministic: verifier.py)
    -> confound detection (deterministic: confound_detection.py)
    -> contradiction detection (deterministic: contradiction_detection.py)
    -> derive hypothesis status from evidence (deterministic: hypothesis_evaluator.py)
synthesize final answer (LLM call 3: synthesizer.py)
    -> causation guard (deterministic: causation_guard.py)
    -> conclusion guard (deterministic: conclusion_guard.py -- prepends a caveat when
       any blocks_conclusion limitation exists)
    -> cap recommendation confidence by evidence strength (deterministic:
       recommendation_grounding.py)
    -> machine-checkable epistemic-principle checks (deterministic: epistemic_checks.py)
```

Orchestrated by `app/reasoning/orchestrator.py`. Exactly 3 structured LLM calls
regardless of dataset size or tool-loop length — a deliberate, repeatedly-preserved
architectural invariant (see `.agent/reasoning-layer-design.md`).

## Current critical modules

| Module | Responsibility |
|---|---|
| `question_parser.py` | LLM call 1 — intent, requested metrics/dimensions, claims |
| `premise_validator.py` | Deterministic — checks requested columns/scale/time-range exist |
| `planner.py` | LLM call 2 — capability-category selection, hypothesis generation |
| `executor.py` | Bridges to the existing `DataAnalystAgent`/`ToolRouter` tool-calling loop |
| `verifier.py` | Deterministic — finding classification, cross-check, sample-size/outlier limitations |
| `confound_detection.py` | Deterministic — categorical confound detection (2-group, 3+-group/ANOVA, nested-relationship guard) |
| `contradiction_detection.py` | Deterministic — mean-vs-median ranking-reversal detection only (built last mission) |
| `numerical_sanity.py` | Deterministic — impossible percentages, population mismatch, group-magnitude outliers, group-size imbalance, unusual baseline window |
| `hypothesis_evaluator.py` | Deterministic — derives hypothesis status from evidence, never from the LLM's own claim |
| `synthesizer.py` | LLM call 3 — final answer + recommendation |
| `causation_guard.py` | Deterministic — rewrites unhedged causal language absent a supporting hypothesis |
| `conclusion_guard.py` | Deterministic — prepends an unmissable caveat when `blocks_conclusion` severity is present (built last mission) |
| `recommendation_grounding.py` | Deterministic — caps `Recommendation.confidence` by evidence strength; `blocks_conclusion` forces `None` regardless of evidence strength |
| `epistemic_checks.py` | Deterministic — 10 machine-checkable principle canaries, surfaced as `principle_violations` |

## Current known limitations (from `FINAL_STRESS_TEST_MISSION.md`, still true)

1. Impossible-percentage check has no real trigger path in the current toolset — no
   tool computes a cross-aggregation ratio (e.g. a funnel conversion rate) as a
   structured field.
2. Contradiction detection covers only mean-vs-median ranking reversal — this
   mission's Phase 2 target.
3. Confound detection is categorical-only, and candidate-column selection is a
   simple "scan every low-cardinality column" pass, not intelligent/targeted — this
   mission's Phase 1 target.
4. Two of the (prior mission's) 14 self-challenge questions — fragile-assumption
   identification, uncertainty-reduction suggestions — are prompt-only, not
   deterministic.
5. The SQL bridge requires full in-memory materialization; does not itself scale to
   100M-row SQL queries (chunked pandas operations do; SQL does not).
6. Real-LLM validation is quota-limited; broad coverage exists from prior sessions,
   this mission should prioritize previously-unmeasured or newly-changed cases only.
7. The 100M-row large-data numbers are from a prior verified run
   (`app/large_data/benchmark_100m_results.json`), not re-measured every session.

This report is the pre-change baseline this mission's regression checks compare
against. Do not edit it after this point — subsequent progress goes in
`.agent/V2_MISSION_REPORT.md`.
