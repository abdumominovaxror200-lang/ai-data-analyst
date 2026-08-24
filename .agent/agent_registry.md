# Agent Registry — Proposed File Ownership (Single Writer Rule)

Status: **PROPOSED**, not yet approved for Wave 1. No subagent has been assigned or
launched. One writer per file; every other team may review but not modify.

## Existing files — owner reassignment (no conflicts found)

| File | Proposed owner (writer) | Reviewers |
|---|---|---|
| `backend/app/datasets/storage.py`, `validation.py` | DATA-ARCHITECT | SECURITY, QA |
| `backend/app/tools/profiler.py`, `filtering.py`, `aggregation.py`, `comparison.py` | EDA-ANALYST | QA, DATA-VALIDATION |
| `backend/app/tools/statistics.py`, `correlation.py` | STATISTICS-ENGINEER | QA, DATA-VALIDATION |
| `backend/app/tools/anomaly.py` | STATISTICS-ENGINEER (owns the method), EDA-ANALYST (consumer) | QA |
| `backend/app/tools/charts.py` | VISUALIZATION-ENGINEER | QA |
| `backend/app/tools/insights.py`, `report.py` | BUSINESS-ANALYST | REPORTING-ENGINEER, QA |
| `backend/app/tools/serialization.py`, `errors.py` | TOOLING-ENGINEER (shared utility, low-churn) | all tool teams (read) |
| `backend/app/agent/agent.py` | **AGENT-ARCHITECT** (sole writer — this is the file the "single writer" rule most matters for; it's the tool-calling loop every team's new tools flow through) | REASONING/VALIDATION, CONTEXT/TOKEN, QA, SECURITY |
| `backend/app/agent/tool_router.py` | **TOOLING-ENGINEER** (sole writer — the tool registry every new capability must register through) | AGENT-ARCHITECT, all capability teams (propose additions via contract, not direct edits) |
| `backend/app/agent/providers.py` | AGENT-ARCHITECT | SECURITY, PERFORMANCE |
| `backend/app/api/routes_datasets.py` | DATA-ARCHITECT | QA |
| `backend/app/api/routes_analysis.py`, `routes_chat.py` | AGENT-ARCHITECT | QA |
| `backend/app/api/routes_reports.py` | REPORTING-ENGINEER | QA |
| `backend/app/api/routes_health.py`, `main.py`, `config.py`, `logging_config.py` | DEVOPS-ENGINEER | SECURITY |
| `backend/app/schemas.py` | Shared contract file — **DATA-ARCHITECT** owns structure, additive-only changes from other teams (append fields, never remove/rename without an RFC in `decisions.md`) | all |
| `backend/tests/**` | Each team owns tests for its own files (paired with the implementation, same PR/task); QA-ENGINEER owns `conftest.py` and test infrastructure/conventions | — |
| `backend/requirements.txt`, `pytest.ini` | DEVOPS-ENGINEER | all (propose additions) |
| `frontend/src/**` | **Not covered by the proposed org chart.** Flagged in `decisions.md` — needs an explicit FRONTEND-ENGINEER (or split between VISUALIZATION-ENGINEER for chart components and REPORTING-ENGINEER for report UI) before Wave 2, since new backend capabilities (SQL results, forecasts, clusters) will need frontend surfaces. |
| `README.md`, `docs/**`, `reports/**` | DOCUMENTATION-ENGINEER | all (source of truth, others provide input) |
| `.gitignore`, CI config (future) | DEVOPS-ENGINEER | SECURITY |

## New (greenfield) subsystems — no existing file, owner assigned by capability

| Subsystem | Owner | Notes |
|---|---|---|
| `backend/app/data/` (proposed: `DataSource`, `Dataset`, `Schema`, `Table`, `Column`, `QueryResult` contracts) | DATA-ARCHITECT | Must land *before* SQL-ENGINEER or LARGE-DATA can start — see dependency_graph.md |
| `backend/app/sql/` (query generation/validation/execution) | SQL-ENGINEER | Depends on DATA-ARCHITECT's contracts |
| `backend/app/large_data/` (chunking, pushdown, sampling, streaming) | LARGE-DATA-ENGINEER | Depends on DATA-ARCHITECT's contracts |
| `backend/app/tools/hypothesis.py`, `regression.py` (t-test, chi-square, ANOVA, CI, effect size) | STATISTICS-ENGINEER | Needs `scipy`/`statsmodels` added to requirements.txt (DEVOPS reviews) |
| `backend/app/tools/forecasting.py` | FORECASTING-ENGINEER | Needs a time-series library decision (see decisions.md — open question) |
| `backend/app/tools/clustering.py`, `segmentation.py` (RFM, churn, cohort, CLV) | ADVANCED-ANALYTICS | Needs `scikit-learn` added; depends on DATA-ARCHITECT if it needs multi-table joins (RFM/cohort often do) |
| Independent validation/ground-truth harness | DATA-VALIDATION-ENGINEER | New, not a modification of existing tools |
| Repeatable benchmark suite (this session's 7-question run, formalized) | BENCHMARK-ENGINEER | Should absorb the 7 questions already run manually + this session's ground-truth cross-checks as its first fixtures |
| Dockerfile, CI/CD pipeline, monitoring | DEVOPS-ENGINEER | Entirely new |

## Rule reminder (per user's architecture spec, section 13)

No two agents write the same file in the same wave. `agent.py` and `tool_router.py` are
the highest-risk convergence points — every new capability needs a tool registered
there, but only TOOLING-ENGINEER (tool_router.py) and AGENT-ARCHITECT (agent.py) may
edit those files. Other teams submit their tool function + schema for TOOLING-ENGINEER
to wire in, rather than editing the router themselves.
