# Production Readiness — Gap Matrix & Roadmap

Status as of HEAD `0f1d484` (before this document's own commit). 896 tests passing,
74 real-LLM tests correctly gated. This document is the authoritative, continuously
updated tracker for the "Master Mission" — verify every claim below against actual
code before trusting it; do not assume prior `.agent/` docs are current (several
were found stale during this audit, see below).

## Phase 0 audit — gap matrix

**A. Genuinely complete** (real code, real tests, most with real-LLM evidence):
- Reasoning pipeline: contracts → question_parser → premise_validator → planner →
  executor → verifier → hypothesis_evaluator → causation_guard →
  recommendation_grounding → synthesizer → epistemic_checks, all wired into
  `orchestrator.py`. 39 tools registered, categorized, filtered.
- SQL layer (DuckDB + SQLite, read-only enforcement, resource limits) — tested,
  security-reviewed.
- Prompt-injection trust boundary — tested, verified live against real Groq.
- `/api/reason` and `/api/chat` both functional, both integration-tested.
- Causation guard v2 (layered), recommendation grounding, evidence-derived
  hypothesis status — all fixed and verified this session, including one real
  cross-cutting bug found and closed during integration (weakly_supported
  excusing unhedged causal claims).

**B. Partially complete**:
- Real-LLM benchmark coverage: 18/37 professional + 3/15 adversarial cases got a
  real model response before hitting a sustained provider-capacity wall; **100%
  of real responses passed**. The remaining cases are unmeasured, not failed.
- 100M-row large-data engine: chunking/sampling/aggregation/memory-guard all
  real, tested, benchmarked at 100M rows (see `benchmark_100m_results.json`) —
  **but see finding C1 below: none of it is reachable from the actual product.**

**C. Implemented but NOT integrated (the most important findings this audit made)**:
1. **`app/large_data/` is completely disconnected from the real upload path.**
   `DatasetStore.save()` (`app/datasets/storage.py`) still does a single
   `pd.read_csv(io.BytesIO(content))` / `pd.read_excel(...)` — a full, non-chunked,
   in-memory parse — gated only by `max_upload_mb=25` / `max_rows=500_000`. The
   100M-row chunking/DuckDB-direct capability that was benchmarked this session
   is real and correct as a standalone package, but a real user cannot reach it
   through the product today. **This is the single largest gap between what's been
   built and what's actually usable.**
2. **`/api/reason`'s response never exposes real computed values.** `FindingOut.
   statement` is a generic auto-generated string (e.g. "describe_data produced a
   result for 'revenue'.") — the actual number lives in `Evidence.result_summary`,
   which has NO corresponding field in `schemas.py`/`routes_reasoning.py` at all.
   A caller gets classification/limitations/hypotheses metadata but no traceable
   numeric evidence outside the free-text `answer` string. This directly
   undermines this project's own "evidence > confidence, traceable evidence"
   principle at the actual product surface.
3. **`AnalysisResult.principle_violations` (Phase 4 P2's epistemic checks
   output) is computed but never surfaced through the API** — confirmed via
   `grep`, zero references in `schemas.py`/`routes_reasoning.py`.
4. **Frontend never calls `/api/reason` at all.** `frontend/src/api/client.ts`
   only wraps `/datasets/upload`, `/analysis`, `/chat`, `/reports` — the entire
   reasoning-layer output (findings, limitations, hypotheses, recommendation
   grounding, principle violations) has no UI surface whatsoever. A user can only
   see it via a raw API call.
5. **`business_diagnosis.py`/`data_quality.py`/`advanced_charts.py` tools** are
   registered and LLM-callable, but have no dedicated frontend rendering beyond
   whatever the generic chart/markdown components already do for `/api/chat`.

**D. Tested only with mocks**: the large majority of the 75 scripted benchmark
cases (60 professional + 15 adversarial from Phase 3C) — real behavior spot-checked
live for a subset only (21 real cases total across all live runs this session).

**E. Real-LLM evidence exists for**: prompt-injection resistance, the row-count/
planner fix, the outlier-limitation fix, 21 benchmark cases (18 professional + 3
adversarial), one manually-verified causal-hedging case.

**F. No evidence at all**:
- Frontend behavior with any reasoning-layer output (never wired, never tested).
- The actual upload path at any scale beyond ~500K rows (only the standalone
  `large_data` package was tested, never through `/api/datasets/upload`).
- Deployment: **zero Dockerfile, docker-compose, or CI config exists anywhere in
  the repo** (confirmed via `find`).
- Observability beyond basic request-ID-tagged logging (`logging_config.py`) —
  no metrics, no tracing, no structured error taxonomy beyond the 3 exception
  handlers in `main.py`.

**G. Known limitations (already honestly documented in `decisions.md`)**:
causation_guard is regex-based not semantic; hypothesis-evidence linking is a
bag-of-words heuristic; SQL bridge/`profile_dataset` require full in-memory
materialization; real-LLM benchmark coverage is partial due to provider quota.

**H. Production blockers (real, prioritized below)**: C1–C5 above, plus the
missing deployment configuration and the stale README (see I).

**I. Merely polish**: additional chart types, more adversarial cases beyond the
current 15+12(new), additional documentation detail.

**Stale documentation found**: `README.md` describes only the original ~10-tool
MVP with zero mention of the reasoning layer, SQL, forecasting, clustering,
segmentation, causation guard, or anything from Waves 1–2 / Phases 3A–5. Anyone
evaluating this project from the README alone would miss ~90% of what exists.

## Prioritized roadmap

| Pri | Item | Status |
|---|---|---|
| P0 | Security re-verification (no new work expected — confirm, don't rebuild) | Verify only |
| P1 | Expose real evidence in `/api/reason` response (C2, C3 above) | **In progress this session** |
| P1 | Wire `app/large_data/` into the real upload/analysis path (C1) | Scoped below — large, sequenced separately |
| P2 | Scalability: none beyond P1's large-data wiring is currently blocking | — |
| P3 | Analyst workflow (auto-profile → decompose → plan → verify → charts → exec summary) | Deferred pending P1 completion — the API must carry real evidence before a richer workflow is worth building on top of it |
| P4 | Frontend reasoning-layer integration (C4, C5) | Scoped for parallel execution |
| P5 | Deployment config (Dockerfile, compose, CI) + observability | Scoped for parallel execution |
| P6 | README/docs overhaul | This session |

**Sequencing rationale**: P1's evidence-exposure gap is fixed first because P3
(analyst workflow: charts, executive summary, technical report) and P4 (frontend)
both depend on the API actually carrying real data — building a richer workflow
or UI on top of today's evidence-less response would just propagate the same gap
further, not fix it.

**The C1 large-data-wiring gap is real and important but is NOT being rushed into
a partial architecture change this session** — it requires `DatasetRecord` to
support a "reference a file, not necessarily a full DataFrame" mode, and every one
of the 39 tools' `record.df` access pattern to be reconciled with that, which is a
genuine multi-file, high-conflict-risk architectural change (touches
`storage.py`, `tool_router.py`, potentially every tool). Per the mission's own
instruction to write high-conflict architectural decisions into `decisions.md`
*before* implementing, this is logged as a proposed design below, not yet built.

### Proposed design for C1 (large-data upload wiring) — NOT YET IMPLEMENTED

Documented here per the mission's explicit rule (architecture decisions go into
`decisions.md`/here before a high-conflict rewrite): `DatasetStore.save()` would
need a size-based branch — under a configurable row-count threshold (e.g. current
500K), keep today's full in-memory path unchanged (zero regression risk, this is
the well-tested, common case). Above it, store the raw file and defer full
materialization: `DatasetRecord` gains an optional `large_file_path` alongside
`df`, and tools that can operate via `app/large_data/` (aggregation, sampling,
count) or the SQL bridge could route through those instead of `record.df` when
present. This is a real design, not a stub, but it changes the calling contract
for a large fraction of `app/tools/*.py`, which is why it isn't attempted as a
same-session addition to everything else already changed.

## Real-LLM benchmark discipline for this mission (per explicit instruction)

Sequential only, never concurrent, small representative batches, cache results,
never fabricate missing numbers, always distinguish PROVIDER_FAILURE from
MODEL_FAILURE from TEST_INFRASTRUCTURE_FAILURE.
