# Completed Tasks

## Phase 3A — TOOLING INTEGRATION — COMPLETE
- OWNER: orchestrator (direct — mechanical, well-scoped registration work, no
  subagent launched per explicit user instruction)
- STATUS: COMPLETE, committed.
- FILES_CHANGED: `backend/app/agent/tool_router.py` (22 new tool schemas +
  handlers + a new SQL bridge — `_run_sql_query`/`_explain_sql_query` — added
  directly in this file, the existing `app/sql/` engines were NOT modified),
  `backend/tests/test_tool_registration.py` (new, 63 tests),
  `.agent/tool_inventory.md` (new — full per-tool contract/security/failure-mode
  documentation for all 32 registered tools).
- IMPLEMENTATION: registered every already-built, already-tested Wave 1 + Wave 2
  analytical tool that was sitting unreachable from the LLM: 5 hypothesis-testing
  tools, `linear_regression` + `regression_diagnostics` + `outlier_analysis_multivariate`,
  4 forecasting tools, `kmeans_cluster` + `pca_reduce`, 3 segmentation tools
  (RFM/cohort/churn), 3 EDA tools (`automated_eda`/`analyze_cardinality`/
  `analyze_distributions`), and 2 new SQL-bridge tools (`run_sql_query`/
  `explain_sql_query` — the only genuinely new code this pass, since the SQL
  engines had no existing agent-facing entry point). `agent.py`'s core loop was
  NOT modified — registration alone required no change to it (dedup, stagnation-stop,
  and the `[UNTRUSTED DATA]` wrapping are generic over any tool). 10 pre-existing +
  22 newly wired = 32 total tools.
  Deliberately NOT registered: `app/large_data/**` (ingestion-layer, operates on
  file paths not loaded DataFrames, and isn't wired into the upload path either —
  see `tool_inventory.md`) and `app/data/**` (internal contracts, not LLM-facing).
- SECURITY: no bypass of any existing control. SQL bridge reuses the unmodified
  `DuckDBDataSource`/`SQLiteDataSource` read-only engines verbatim; adds one new,
  tool-specific constraint (`_SQL_TOOL_MAX_ROWS=500`, tighter than the engines'
  own 10,000-row default) to bound payload size back to the LLM, per the
  project's documented 413-payload history. Every tool result — new or
  pre-existing — is still wrapped by `agent.py`'s untrusted-data marker with no
  change to that mechanism.
- TESTS: baseline 505 passed → 568 passed (505 + 63 new, zero regressions),
  independently re-run by the orchestrator. New tests prove: every schema has a
  matching handler and vice versa (no orphans), every schema is well-formed,
  missing/invalid arguments are rejected for a sample of new tools, 8 dangerous
  SQL statements are blocked on both engines via the new bridge, the SQL
  row-cap is real (verified truncation), new tool results carry the
  untrusted-data marker through the full agent loop, and — the item 7
  integration test — the agent can discover *and actually execute* a
  previously-unreachable Wave 2 tool (`forecast`) end-to-end via a scripted
  `MockProvider`, plus three more tool classes (stats/clustering/SQL) each
  independently reachable through the same loop.
- COMMIT: `(see git log — "Phase 3A: register Wave 1+2 tools into tool_router.py")`.
- REMAINING ARCHITECTURAL BLOCKER: none for this phase's stated goal (make
  built capabilities reachable). Two things intentionally deferred, not blocking:
  (1) the tool catalog is now 32 entries offered to a single flat
  function-calling completion — fine for Phase 3A's goal, but the
  reasoning-layer's planner (Phase 3B) will need to filter/rank this catalog
  rather than relying on raw LLM free choice as today, per
  `reasoning-layer-design.md`; (2) `app/large_data/` is still not reachable by
  either the upload path or the agent — registering it needs an ingestion-path
  decision first, not just a tool-router entry.

## Wave 2 — 3 of 7 launched agents landed; remaining 4 put ON HOLD by user pivot
- FORECASTING-ENGINEER: COMPLETE, integrated (merge commit `705833b`). Worktree was
  stale (branched from the initial commit); agent self-detected via `git log`/
  `git merge-base`, had zero divergent commits, fast-forwarded itself to true main
  before starting work — no orchestrator reconciliation needed. `forecasting.py`
  (train_test_split_timeseries, decompose_timeseries, forecast, backtest_forecast),
  38 new tests.
- ADVANCED-ANALYTICS-ENGINEER: COMPLETE, integrated (merge commit `ec67573`). Same
  stale-worktree pattern, same self-fix (fast-forward, zero lost work).
  `clustering.py`, `segmentation.py`, `regression_diagnostics.py`, 42 new tests.
- EDA-PROFILING-ENGINEER: COMPLETE, integrated (commit `fb2aa8d`). Worktree was
  stale (branched from the *very first* commit — `git merge-base` would have been
  destructive to reconcile), and the agent correctly did NOT self-fast-forward since
  its situation differed; it flagged staleness prominently instead. Orchestrator
  integrated by copying its two new, non-conflicting files directly onto current main
  rather than merging the outdated branch history. `eda.py` (automated_eda,
  analyze_cardinality, analyze_distributions), 19 new tests.
- **BUSINESS-ANALYTICS-ENGINEER, VISUALIZATION-ENGINEER, DATA-QUALITY-ENGINEER,
  QA-PROFESSIONAL-BENCHMARK-ENGINEER: NOT LAUNCHED.** User sent an explicit pivot
  instruction mid-Wave-2 to design an "Evidence-Based Analytical Reasoning Layer"
  first (see `reasoning-layer-design.md`) and explicitly said not to implement more
  until that design is approved. These 4 remain valid, well-scoped Wave 2 tasks and
  can resume once the user gives the go-ahead — nothing about them is invalidated by
  the pivot, they're simply paused.

Full regression suite after the 3-agent Wave 2 integration: **505/505 passing**
(406 pre-Wave-2 baseline + 38 + 42 + 19 new, independently re-run by the orchestrator).

**Recurring finding across all 3 landed Wave 2 agents**: every worktree branched from
the repo's *initial* commit instead of the current `f72a93a` HEAD at launch time — the
same worktree-staleness failure mode first seen in Wave 1, now confirmed systemic
rather than a one-off. Two agents (FORECASTING, ADVANCED-ANALYTICS) had zero divergent
commits of their own and safely self-recovered via `git merge <main-commit> --ff-only`
before starting work — this should be added to the standard Wave-launch instructions
as a first step every agent runs, rather than something each one has to independently
think of. The third (EDA-PROFILING) correctly recognized its situation was different
(deep staleness, no easy fast-forward) and instead flagged for orchestrator handling —
also the right call. Both responses were correct given their respective situations.

## Wave 1 — 4 of 6 landed (commit b9701e5), 2 in progress
- DATA-ARCHITECT: COMPLETE, integrated. Data contracts in `backend/app/data/`.
  Orchestrator reconciled two format mismatches (the agent's worktree branched
  before the P0/P1 date-coverage fixes landed on main) — see decisions.md.
- STATISTICS-ENGINEER: COMPLETE, integrated. `hypothesis.py` + `regression.py`,
  38 new tests, zero regressions.
- LARGE-DATA-ENGINEER: COMPLETE, integrated. `backend/app/large_data/`, 24 new
  tests, real 100K/1M/10M-row benchmark (see benchmark_status.md).
- QA-ENGINEER: COMPLETE, integrated. Ground-truth benchmark regression suite +
  SQLite/DuckDB test fixtures, 15 new tests.
- SQL-ENGINEER: IN PROGRESS. Note: this agent's isolated worktree was never
  created (infrastructure issue) — it worked directly in the main checkout.
  No existing tracked file was touched (confirmed via `git status`), only new
  files under `backend/app/sql/` plus some stray debug artifacts
  (`backend/evil.db`, `backend/out.csv`) that will be cleaned up, not committed,
  once its final report lands.
- SECURITY-ENGINEER: IN PROGRESS (proper worktree isolation).

Full regression suite after the 4-agent integration: **337/337 passing**
(independently re-run by the orchestrator, not just trusted from agent reports).

## Wave 0 — Audit
- OWNER: Claude Code (Orchestrator)
- STATUS: COMPLETE
- OUTPUT: `architecture.md`, `agent_registry.md`, `dependency_graph.md`, `decisions.md`,
  `roadmap.md`

## Pre-Wave-0 (this session, before the multi-agent structure was proposed)
- Backend MVP built: FastAPI + 10 analysis tools + agent loop + React/TS frontend.
  86 tests passing.
- Real end-to-end verification against Groq (`openai/gpt-oss-120b`).
- 4 reliability fixes shipped and benchmark-verified: error-message sanitization,
  global exception safety net, tool-loop duplicate/stagnation control, date-coverage
  constraint validation — plus two bugs found and fixed along the way.
- GitHub publication audit + push completed.
