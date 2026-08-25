# Active Tasks

## Phase 3C Part H — 5 parallel worktree agents launched, branched from f545bcb

| Agent | Scope | Status |
|---|---|---|
| BENCHMARK-ENGINEER | `backend/tests/benchmark/professional_benchmark.json` (50+ cases, Part C) + `test_professional_benchmark.py` | IN PROGRESS |
| QA-BENCHMARK-ENGINEER | `backend/tests/benchmark/adversarial_cases.json` (15 cases, Part D) + `test_adversarial_benchmark.py` + honesty-audit of scoring.py | IN PROGRESS |
| BUSINESS-ANALYTICS-ENGINEER | `backend/app/tools/business_diagnosis.py` (contribution_analysis, executive_summary) + tests | IN PROGRESS |
| DATA-QUALITY-ENGINEER | `backend/app/tools/data_quality.py` (duplicate_analysis, data_quality_report) + tests | IN PROGRESS |
| VISUALIZATION-ENGINEER | `backend/app/tools/advanced_charts.py` (correlation_heatmap_data, boxplot_data, pareto_chart_data) + tests | IN PROGRESS |

INTEGRATION-ENGINEER's scope (Part A/B) was NOT launched as a 6th subagent — done directly
by the orchestrator instead (commit 4955ba1), same small/foundational/high-conflict-risk
judgment applied in Phases 3A/3B. Disclosed in the Phase 3C completion report.

No agent may modify another's files (see prompts for exact ownership). None may touch
`tool_router.py`, `categories.py`, `app/reasoning/**`, or `scoring.py` — the orchestrator
registers new tools and runs final integration/scoring after all 5 land.
