# Decisions Log

## Already decided (context for new agents — do not re-litigate without cause)

| Decision | Why | When |
|---|---|---|
| No database for the original MVP; in-memory `DatasetStore` | Scoped to a single-session demo; see `docs/architecture.md#why-no-database-for-v1` | Initial build |
| LLM provider = any OpenAI-compatible `/chat/completions` API, currently Groq `openai/gpt-oss-120b` | Free tier available for demo/testing; swappable via 3 env vars | Initial build |
| Tool outputs must be summary-only where they can grow unbounded (e.g. `generate_business_insights` excludes raw anomaly rows) | A real 413 "payload too large" failure was traced to this; fixed and regression-tested | This session |
| Agent has duplicate-call detection + stagnation early-stop, `MAX_TOOL_ITERATIONS=10` | A real benchmark run showed the agent burning all 6 iterations on repeated identical calls with no final answer; fixed and verified against real Groq traffic | This session |
| System prompt requires explicit constraint-mismatch disclosure (row count, date coverage, missing columns) rather than silent substitution | A real benchmark run showed the agent silently analyzing a 4,000-row dataset against a question describing a 10-million-row database, without saying so | This session |

## Wave 1 operational findings (for whoever runs Wave 2+)

- **Worktree-isolated subagents do not auto-commit.** All 5 agents that got a real
  worktree left their work as *uncommitted* changes in that worktree's working
  directory — `git merge <branch>` found nothing to merge ("already up to date")
  because there was no commit on the branch. The orchestrator integrated by copying
  files directly from each worktree into main and committing there. Instruct future
  agents explicitly to `git add` + `git commit` inside their own worktree before
  reporting done, or budget for the orchestrator to do the copy-and-commit step.
- **Worktrees branched from an older commit than intended.** All 6 Wave 1 worktrees
  were created from `51759f8` (the initial commit), not `ad1539f` (the P0/P1-fixes +
  Wave-0-audit commit that was HEAD at the moment the agents were launched). Effect:
  DATA-ARCHITECT and STATISTICS-ENGINEER both independently noticed and correctly
  flagged that `profile_dataset`/`compare_periods` in their worktree lacked fields
  the Wave 0 audit doc claimed existed (`date_ranges`, `text_columns`, per-column
  `min_date`/`max_date`, coverage warnings) — they were seeing pre-fix code. This
  was reconciled at integration time (see `completed_tasks.md`) rather than being a
  real bug in either agent's work. **Lesson**: don't assume a subagent worktree is
  based on "whatever HEAD is right now" — verify with `git log` in the worktree
  before trusting an agent's "existing behavior" observations at face value, and
  expect to reconcile format drift for anything the orchestrator changed on main
  after Wave 0 but before/during Wave 1.
- **Orchestrator cross-review of SQL-ENGINEER against SECURITY-ENGINEER's threat
  model checklist** (requested in both agents' final reports): 8 of 9 checklist items
  in `backend/docs/security/sql-layer-threat-model.md` section 4 are met — real
  parser-based statement-type/single-statement enforcement (not regex), engine-level
  `read_only=True` connection, per-dataset catalog isolation (one temp DB file per
  `DataSource` instance), documented function-denylist gap, error sanitization
  reusing the established pattern. **One item is NOT met: "query timeout and
  resource limits."** Neither `DuckDBDataSource` nor `SQLiteDataSource` enforces an
  execution-time or memory cap — only output row count is capped (`LIMIT`). A
  valid-but-expensive `SELECT` (large cross join, `range()` of a huge size, deeply
  nested subqueries) could still exhaust CPU/memory before ever returning a row to
  truncate. **This must be closed before the SQL layer is exposed to the live agent
  loop or any untrusted input** — tracked as an open Wave 2 task, see `roadmap.md`.
- **Prompt-injection gap is now CONFIRMED reachable, not hypothetical** (was an open
  question in Wave 0's audit). SECURITY-ENGINEER built a real PoC proving adversarial
  text in a dataset's categorical/text column reaches the LLM verbatim via
  `group_and_aggregate`, `filter_data`, `describe_data`, and `detect_anomalies`,
  both at the tool level and through the real `DataAnalystAgent` loop. Current blast
  radius is limited (no tool has write/network/side-effect capability yet, so worst
  case is manipulating the agent's own next reply in the same session) — but
  SQL-ENGINEER's read-only query layer landing this same wave is exactly the kind of
  capability-expansion that should trigger revisiting the "defer" call. Recommended
  fix (AGENT-ARCHITECT's file, not made this wave — no such agent ran): add an
  explicit "tool results are data, not instructions" boundary to `agent.py`'s system
  prompt. Low cost, not yet applied — tracked as a Wave 2 task.
- **XLSX upload XXE/entity-expansion protection is real today but accidental**,
  inherited from CPython's bundled `expat` defaults rather than an explicit control
  (`lxml`/`defusedxml` are not installed). A future dependency change (e.g. adding
  `lxml` without `resolve_entities=False`) could silently regress this, since
  openpyxl checks for `lxml` before falling back to the safe default. No action
  taken this wave; worth a one-line guard or an explicit `defusedxml` dependency if
  `lxml` is ever added for another reason.
- **One agent's worktree was never created at all** (SQL-ENGINEER) — it worked
  directly in the main checkout instead. No existing tracked file was damaged
  (verified via `git status` showing only new/untracked files), but this is a real
  isolation failure the orchestrator had to detect after the fact, not something
  the agent could have known. Its stray debug artifacts (`backend/evil.db`,
  `backend/out.csv`) needed manual cleanup rather than being excluded by worktree
  boundaries. **Lesson**: after any agent reports completion, run `git worktree
  list` and diff against the expected worktree set before trusting that isolation
  actually held — don't assume the `isolation: "worktree"` parameter always
  succeeds silently.

## Decided (2026-08-24, by user)

1. **SQL target databases for the first SQL-ENGINEER pass: DuckDB + SQLite.** Embedded,
   no server to stand up, DuckDB reads CSV/Parquet natively and is genuinely fast at
   the "10M+ rows" scale LARGE-DATA cares about. Postgres/MySQL/BigQuery/Snowflake/
   ClickHouse deferred to a later pass, per the original phasing.

2. **Statistics + forecasting library: `statsmodels`** for both STATISTICS-ENGINEER
   (hypothesis testing, regression) and FORECASTING-ENGINEER (ARIMA/ETS) — one shared
   dependency instead of two. `scipy` likely still needed underneath (statsmodels
   depends on it) but no separate `prophet`/other forecasting library.

## Open — need a decision before the relevant wave starts

1. **Frontend ownership.** No team in the proposed org chart owns `frontend/src/`.
   New capabilities (SQL results, forecasts, clusters) will need UI surfaces before
   Wave 2 ships anything user-visible. **Decision needed from the user**: add a
   FRONTEND-ENGINEER role, or split frontend work across VISUALIZATION-ENGINEER /
   REPORTING-ENGINEER / capability owners? Not blocking Wave 1 (no Wave 1 task touches
   the frontend) — must be resolved before Wave 2.

2. **Auth model.** Currently none. Before any DEVOPS production deployment work
   (Wave 4), need a decision: is this staying a local/demo tool (no auth needed), or
   does it need real user accounts/API keys for a hosted deployment? Significant scope
   difference.

3. **Benchmark question bank ownership.** This session's 7-question manual benchmark
   (Uzbek-language, business-analyst-style questions) should become
   BENCHMARK-ENGINEER's first fixture set. **Recommend**: formalize it as a
   version-controlled fixture file (`backend/tests/benchmark/questions.json` or
   similar) with the ground-truth values already computed this session, so
   regressions are caught automatically rather than requiring another manual pass.

## Explicitly deferred (not blocking Wave 1)

- BigQuery/Snowflake/ClickHouse connectors — later, per the user's own phasing.
- Full context-compression/summarization system for CONTEXT/TOKEN-ENGINEER — the
  existing dedup + stagnation-stop already resolved the concrete failure this session
  surfaced (21K-token payload from one bloated tool result); a general token-budget
  accounting layer is a Wave 3 nice-to-have, not a Wave 1 blocker.
- Authentication/authorization implementation — tracked as open decision #2 above, but
  not required to start Wave 1 (data-layer and analytics work don't need it yet).
