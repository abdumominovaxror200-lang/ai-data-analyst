# Integration Status

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
