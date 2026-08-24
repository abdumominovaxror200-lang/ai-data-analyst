# Integration Status

## Wave 2 — PARTIAL (3 of 7 agents). Remaining 4 on hold, see roadmap.md.

| Commit(s) | Agent | Tests after |
|---|---|---|
| `705833b` | FORECASTING-ENGINEER (clean merge, `--no-ff`) | 444 |
| `ec67573` | ADVANCED-ANALYTICS-ENGINEER (clean merge, `--no-ff`) | 448 (note: agent's own worktree measured 448 before this integration step; orchestrator merge landed the same net result) |
| `fb2aa8d` | EDA-PROFILING-ENGINEER (files copied directly from a stale worktree, not merged — see completed_tasks.md) | 505 |

**Final state: 505/505 tests passing**, independently re-run by the orchestrator after
the full integration (not just trusted from agent reports). Zero regressions on the
406-test pre-Wave-2 baseline. All 3 worktrees removed (`git worktree remove --force`)
and their branches deleted after integration.

**Known, accepted technical debt from this integration**: none of the 5 new tool
modules (`forecasting.py`, `clustering.py`, `segmentation.py`,
`regression_diagnostics.py`, `eda.py`) are wired into `tool_router.py` yet — same
already-logged pattern as Wave 1's 5 subsystems below. This is now flagged as the
recommended *first* step of the next implementation wave in
`reasoning-layer-design.md` §1/§9, since the reasoning layer's planner needs the real
tool catalog to be complete to be useful.

Remaining Wave 2 agents (BUSINESS-ANALYTICS, VISUALIZATION, DATA-QUALITY,
QA-PROFESSIONAL-BENCHMARK) were never launched — user pivoted mid-wave to request the
reasoning-layer architecture design first. See `roadmap.md` and
`reasoning-layer-design.md`.

## Wave 1 — COMPLETE. All 6 agents merged to main.

| Commit | Agent(s) | Tests after |
|---|---|---|
| `b9701e5` | DATA-ARCHITECT (reconciled), STATISTICS-ENGINEER, LARGE-DATA-ENGINEER, QA-ENGINEER | 337 |
| `04df1bd` | SQL-ENGINEER | 380 (146 of these tests were already present as uncommitted files at the b9701e5 checkpoint — see decisions.md's worktree-isolation note) |
| `1a5d00e` | SECURITY-ENGINEER | 380 |

**Final state: 380/380 tests passing**, independently re-run by the orchestrator
after every merge (not just trusted from agent reports). Zero regressions on the
pre-Wave-1 baseline (86 tests) at any point.

## Orchestrator cross-review performed after merge

Per both SQL-ENGINEER's and SECURITY-ENGINEER's reports requesting it: reviewed
`backend/app/sql/` against `backend/docs/security/sql-layer-threat-model.md`
section 4's checklist. 8/9 items met. One gap found and logged: no query
timeout/resource-limit enforcement (only output row count is capped). See
`decisions.md` for the full finding — tracked as a Wave 2 task in `roadmap.md`.

## Known, accepted technical debt from this integration

- `backend/app/data/` (DATA-ARCHITECT) was reconciled by the orchestrator, not the
  original agent, to match the current `profile_dataset` shape (date_ranges,
  text_columns, min_date/max_date) — the agent's worktree branched before those
  fields existed on main. Fix was mechanical (add the missing fields, match the
  date-format string) and is covered by the existing test suite post-fix.
- None of the 6 new subsystems (`app/data/`, `app/sql/`, `app/large_data/`,
  `app/tools/hypothesis.py`, `app/tools/regression.py`) are wired into
  `tool_router.py`/`agent.py` yet — that's explicitly Wave 3 (TOOLING-ENGINEER +
  AGENT-ARCHITECT), out of scope for Wave 1 by design.
