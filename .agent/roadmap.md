# Roadmap — Wave Overview + Proposed Wave 1 Tasks

Status: Wave 0 (audit) and Wave 1 (build-out) both complete — see
`completed_tasks.md`/`integration_status.md`. Two concrete follow-up tasks surfaced
during Wave 1 integration are listed at the end of this file, ready to assign before
or alongside Wave 2.

## Wave overview (confirmed against the audit)

| Wave | Teams | Status |
|---|---|---|
| 0 | Audit (this document set) | ✅ Complete |
| 1 | DATA-ARCHITECT, SQL-ENGINEER, LARGE-DATA, STATISTICS-ENGINEER, QA, SECURITY | ✅ Complete — merged, 380/380 tests passing |
| 2 | EDA, FORECASTING, ADVANCED-ANALYTICS, VISUALIZATION, BUSINESS-ANALYTICS | Not yet planned in detail — depends on Wave 1 output |
| 3 | AGENT-ARCHITECT, TOOLING, REASONING/VALIDATION, CONTEXT/TOKEN | Not yet planned — note: agent.py/tool_router.py already exist and work; this wave *extends* them, doesn't replace. Two Wave-1-discovered fixes below belong here. |
| 4 | PERFORMANCE, SECURITY, DEVOPS, REPORTING, DOCUMENTATION | Not yet planned |
| 5 | BENCHMARK (full suite) | Not yet planned — should absorb this session's 7-question run (now automated in `backend/tests/test_benchmark_ground_truth.py`) as its seed |

## Proposed Wave 1 tasks

Each ready to hand to a subagent once approved. Format follows the reporting contract
from the architecture spec (TASK/OWNER/... at completion) but these are task
*assignments*, not completion reports.

---

**TASK: Data contracts (`DataSource`, `Dataset`, `Table`, `Column`, `Schema`,
`QueryResult`, `DataProfile`)**
OWNER: DATA-ARCHITECT
FILES: new `backend/app/data/` package (contracts only — pydantic models / dataclasses,
no implementation logic yet)
DEPENDS ON: nothing (foundational)
BLOCKS: SQL-ENGINEER, LARGE-DATA-ENGINEER, DATA-VALIDATION-ENGINEER
ACCEPTANCE: existing `DatasetRecord`/`DatasetStore` (current in-memory CSV/XLSX model)
must fit the new contracts without breaking any of the 86 existing tests — this is an
abstraction layer *above* the current storage, not a replacement of it in Wave 1.
NOTES: Decision #3 in `decisions.md` (DuckDB/SQLite first) should inform the `DataSource`
shape so SQL-ENGINEER doesn't need a redesign later.

---

**TASK: SQL layer — generation, validation, read-only execution (DuckDB + SQLite, per
`decisions.md`)**
OWNER: SQL-ENGINEER
FILES: new `backend/app/sql/` package; extends `backend/requirements.txt` with
`duckdb` (SQLite needs no extra dependency — stdlib `sqlite3`)
DEPENDS ON: DATA-ARCHITECT's contracts (can start against a stub, integrate once real)
ACCEPTANCE: read-only enforcement is non-negotiable (no INSERT/UPDATE/DELETE/DDL can
execute, tested with adversarial inputs); JOIN/CTE/window-function/aggregation support;
`EXPLAIN` support for cost visibility before execution.
SECURITY REVIEW REQUIRED before integration (SQL injection is the primary risk this
introduces to a system that currently has none).

---

**TASK: Large-data handling (chunking, sampling, pushdown, memory control)**
OWNER: LARGE-DATA-ENGINEER
FILES: new `backend/app/large_data/` package
DEPENDS ON: DATA-ARCHITECT's contracts
ACCEPTANCE: must define and hit concrete benchmarks at 100K / 1M / 10M row scale (the
current system has *never* been tested past 4,000 rows — this is the first real data
point, not a regression check). Memory ceiling must be explicit and enforced, not
"whatever fits."

---

**TASK: Statistical testing (t-test, chi-square, ANOVA, confidence intervals, effect
size, regression)**
OWNER: STATISTICS-ENGINEER
FILES: new `backend/app/tools/hypothesis.py`, `backend/app/tools/regression.py`;
extends `backend/requirements.txt` with `statsmodels` (pulls in `scipy`), per
`decisions.md` — same dependency Wave 2's FORECASTING-ENGINEER will reuse for
ARIMA/ETS, so add it once here rather than twice.
DEPENDS ON: nothing structural — can start immediately
ACCEPTANCE: same discipline as every existing tool — deterministic Python/SciPy
computation only, LLM never computes a p-value itself (per the project's core "LLM
never invents numbers" rule, `docs/agent-tools.md`). Every new tool needs unit tests
with hand-checked expected values, same pattern as `backend/tests/test_*.py`.

---

**TASK: QA infrastructure for the new subsystems**
OWNER: QA-ENGINEER
FILES: `backend/tests/conftest.py` (extend, don't rewrite), new fixtures for
DB-backed tests (SQL-ENGINEER) and large-data tests (LARGE-DATA-ENGINEER)
DEPENDS ON: DATA-ARCHITECT contracts (for fixture shapes)
ACCEPTANCE: every Wave 1 deliverable ships with tests before being marked complete —
per Rule 1 ("Hech qachon test o'tmasdan feature'ni complete deb hisoblamaydi"). No
existing test may be weakened or deleted to make a new one pass.

---

**TASK: Security review of the SQL layer + prompt-injection assessment**
OWNER: SECURITY-ENGINEER
FILES: review only (no writes) on SQL-ENGINEER's new package; new
`backend/tests/test_sql_security.py` if SQL-ENGINEER doesn't already cover it
DEPENDS ON: SQL-ENGINEER's first draft
ACCEPTANCE: must explicitly test SQL injection resistance and confirm read-only
enforcement can't be bypassed (parameterized queries, no string-concatenated user
input, no DDL/DML verbs reachable). Also: formally assess the prompt-injection gap this
audit flagged (adversarial cell content reaching the LLM's context via tool results) —
decide whether it's in scope for Wave 1 or explicitly deferred with a documented reason.

---

## What happens after Wave 1

Once these six land (with passing tests, security review, and no regression on the
existing 86-test suite + the 7-question benchmark), Wave 2 planning happens against
*actual* DATA-ARCHITECT contracts rather than the proposal in this document — the
detailed Wave 2 task list is intentionally not written yet, per the wave-based
development rule (don't plan five waves deep before wave one has landed).

## Wave 1 done. Two concrete follow-ups surfaced during integration.

Not full Wave 2 planning (that still waits, per the rule above) — these are small,
well-scoped, high-priority fixes to real gaps found while reviewing Wave 1's output.
Both are cheap and both close a *confirmed* (not hypothetical) issue.

---

**TASK: SQL query timeout / resource-limit enforcement**
OWNER: SQL-ENGINEER (same file owner, extending its own package)
FILES: `backend/app/sql/duckdb_source.py`, `backend/app/sql/sqlite_source.py`
WHY: orchestrator's cross-review against `backend/docs/security/sql-layer-threat-model.md`
section 4 found 8/9 checklist items met — this is the missing one. Neither backend
currently caps execution time or memory; only output row count is capped. A
valid-but-expensive `SELECT` (large cross join, huge `range()`, deep subqueries)
can still consume unbounded CPU/memory before returning anything to truncate.
ACCEPTANCE: a deliberately expensive query is rejected/interrupted within a
configurable time budget (DuckDB supports query cancellation via a watchdog
thread + `conn.interrupt()`; SQLite via `set_progress_handler`), with a test proving
it, on both engines.

---

**TASK: "Tool data is not instructions" system-prompt boundary**
OWNER: AGENT-ARCHITECT (agent.py's sole writer — no agent has this role yet; assign
before/alongside Wave 3, or sooner given the finding below)
FILES: `backend/app/agent/agent.py` (the `SYSTEM_PROMPT` constant)
WHY: SECURITY-ENGINEER built a real, passing PoC (`backend/tests/test_prompt_injection_gap.py`)
proving adversarial text in a dataset's categorical/text column reaches the LLM
verbatim through 4 existing tools, both at the tool level and through the real
`DataAnalystAgent` loop. This was an open question in the Wave 0 audit; it is now a
confirmed, reproducible finding. SQL-ENGINEER's read-only query layer landing this
same wave is exactly the kind of capability-expansion that should trigger
prioritizing this — the next tool with any write/network/side-effect capability
makes this urgent rather than low-priority.
ACCEPTANCE: add an explicit instruction to `SYSTEM_PROMPT` that tool-result content
is untrusted data, never instructions, regardless of what it contains — low cost,
no architecture change. `test_prompt_injection_gap.py`'s existing tests document the
reachability; a follow-up test should confirm the boundary instruction is present
(can't fully prove an LLM obeys a prompt instruction via unit test, but the
instruction's presence and the tool-level facts remain regression-tested either way).
