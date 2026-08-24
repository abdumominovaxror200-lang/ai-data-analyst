# Wave 0 Audit — Current Architecture Map

Date: 2026-08-24
Auditor: Claude Code (Orchestrator)
Scope: read-only. No code changed as part of this audit.

## 1. Current architecture (as-built)

```
User → React/TS dashboard → FastAPI → DataAnalystAgent → ToolRouter → pandas/NumPy tools
                                                                              ↓
                                                        structured JSON result (numbers, charts)
                                                                              ↓
                                                   Agent narrates the result → Dashboard / Report
```

- Single FastAPI process, single in-memory `DatasetStore` (uuid → pandas DataFrame),
  process-lifetime only — no database, no persistence.
- One dataset per request; no joins, no multi-table model.
- One LLM provider abstraction (`OpenAICompatibleProvider`), currently pointed at Groq
  (`openai/gpt-oss-120b`), swappable via `.env`.
- React/TS SPA, no routing library, six tabs (Overview/Chat/Charts/Insights/Anomalies/Report).

Total backend code: **1,611 lines** across `app/agent/`, `app/tools/`, `app/datasets/`,
`app/api/`. 86 backend tests (pytest), 0 frontend tests.

## 2. Existing tools (the entire current tool surface)

All in `backend/app/tools/`, registered in `backend/app/agent/tool_router.py`:

| Tool | File | Capability |
|---|---|---|
| `profile_dataset` | profiler.py | shape, column roles, missing/duplicate counts, **date coverage (min/max)** |
| `describe_data` | statistics.py | mean/median/std/min/max/sum (numeric), top values (categorical) |
| `filter_data` | filtering.py | row filtering + preview |
| `group_and_aggregate` | aggregation.py | groupby + sum/mean/median/count/min/max |
| `compare_periods` | comparison.py | two-date-range comparison, **with data-coverage warnings** |
| `correlation_analysis` | correlation.py | Pearson/Spearman/Kendall pairwise correlation |
| `detect_anomalies` | anomaly.py | IQR or z-score outlier detection, single column |
| `generate_chart` | charts.py | bar/line/histogram/scatter/pie chart-ready data |
| `generate_business_insights` | insights.py | bundles profile+stats+anomalies+correlations (summary-only, no raw rows) |
| `generate_report` | report.py | structured report composed from the above |

**This is the entire analytical capability of the system today.** Every capability in
sections 5–7 of the proposed multi-agent architecture (statistics/hypothesis testing,
forecasting, clustering/PCA/RFM/churn/cohort, SQL, large-data) **does not exist yet**.

## 3. Existing "analytics" — precisely what's covered vs. not

| Capability (from proposed org chart) | Status |
|---|---|
| Descriptive statistics | ✅ `describe_data` |
| Correlation | ✅ `correlation_analysis` (no significance/p-value) |
| Anomaly detection | ✅ IQR/z-score only — no isolation forest, no seasonal-aware detection |
| Confidence intervals, hypothesis testing (t-test, chi-square, ANOVA) | ❌ none |
| Effect size, regression | ❌ none |
| Time-series decomposition, forecasting, backtesting, prediction intervals | ❌ none |
| Clustering, PCA, RFM, churn, retention, cohort, funnel, CLV | ❌ none |
| SQL generation/execution against a real database | ❌ none — no DB driver installed at all |
| Large-data (chunking, pushdown, sampling, streaming) | ❌ none — whole file loaded into one in-memory DataFrame |

No `scipy`, `statsmodels`, `scikit-learn`, `sqlalchemy`, or any DB driver appears in
`requirements.txt`. These are genuinely greenfield, not partially-built.

## 4. Existing database support

**None.** The system only ingests uploaded `.csv`/`.xlsx` files, parsed once into a
`pandas.DataFrame` held in a process-lifetime in-memory dict (`app/datasets/storage.py`).
There is no SQL layer, no connection pooling, no query interface, no read-only
enforcement mechanism (because there's nothing to enforce it against yet). The
DATA-ARCHITECT / SQL-ENGINEER / LARGE-DATA roles described in the target architecture
have no existing code to build on — this is a new subsystem, not a refactor.

## 5. Existing agent loop

`backend/app/agent/agent.py` (`DataAnalystAgent.ask`) — already implements several of
the capabilities the target architecture assigns to AGENT-ARCHITECT and
CONTEXT/TOKEN-ENGINEER:

- Tool-calling loop, `MAX_TOOL_ITERATIONS = 10`.
- **Duplicate tool-call detection** (`_canonical_signature`): identical calls are not
  re-executed; a short reuse notice replaces the (potentially large) repeated payload.
- **Stagnation stop**: if a round produces zero new information, the loop forces one
  final no-tools completion instead of running to the hard cap.
- **Constraint-validation system prompt**: instructs the model to flag — not silently
  substitute — when a request's scale/timeframe/columns don't match the dataset's
  actual metadata (row count, date coverage, column list including non-numeric/date
  "text" columns).
- Dataset context message includes row/column counts and **date range coverage** per
  date column, computed once per request from `profile_dataset`.

This is a working, benchmark-validated first version of what the target architecture
calls "Reasoning/Validation Engineer" + "Context/Token Engineer" — not a blank slate.
It is **not** a general context-compression/pruning system (no token-budget accounting,
no summarization of old turns, no caching across requests) — those remain open gaps.

Provider layer (`providers.py`): retry/backoff on 429, sanitized user-facing error
messages (no internal provider details ever returned to the client), optional
`reasoning_effort` passthrough. Single provider implementation
(`OpenAICompatibleProvider`) plus a `MockProvider` for tests.

## 6. Existing tests

86 backend tests, 0 frontend tests, 0 load/performance tests, 0 SQL tests (nothing to
test), 0 statistical-significance tests (no such tool). Coverage breakdown:

- Tool correctness (aggregation, anomaly, charts, correlation/comparison, filtering,
  statistics, profiler, insights, report): unit tests per tool, happy path + validation
  errors.
- Agent behavior: no-hallucination boundary (LLM never sees raw row values before a
  tool call), duplicate-call short-circuit, hard-iteration-cap fallback, tool-error
  surfacing without crashing.
- API: 404/400/422/503 paths, malicious-file handling (path traversal, wrong
  extension, oversized, malformed content), JSON-serialization edge cases (datetime
  columns through the LLM's message payload).
- Provider: rate-limit backoff parsing, error-message sanitization (verified the exact
  leaked-detail string from a real incident never reaches the user-facing message).

No end-to-end (browser) tests are automated — the multi-tab user journey and the real
Groq integration were verified manually in this session, not via a repeatable suite.

## 7. Existing security posture

| Control | Status |
|---|---|
| Upload extension/size/row-count limits | ✅ |
| Path traversal protection (UUID-only storage paths) | ✅ tested |
| No `pickle`/macro execution on uploads | ✅ (pandas/openpyxl only) |
| CORS restricted to configured origin(s) | ✅ |
| `.env` gitignored, API key never logged/returned | ✅ verified via grep + git check-ignore |
| Generic exception → clean JSON (no traceback leak) | ✅ global FastAPI handler |
| Provider error sanitization (no internal IDs/jargon to client) | ✅ tested with the real leaked string |
| Authentication / authorization | ❌ none — any network-reachable client can upload/query |
| SQL injection / write defense (`app/sql/`, added Wave 1) | ✅ layered: engine-level read-only connection (DuckDB `read_only=True`, SQLite `set_authorizer`) + parser-based statement-type/single-statement validation + function denylist for file-reading table functions. 146 adversarial tests across 10 attack categories, all blocked. One documented, accepted-risk gap: the file-reading-function block is a denylist, not an allowlist (see `backend/docs/security/sql-layer-threat-model.md`). |
| SQL resource-exhaustion / DoS defense | ✅ (P0 remediation, closed this session) — execution-time timeout (DuckDB: watchdog thread + `conn.interrupt()`; SQLite: `set_progress_handler`) and a memory ceiling (DuckDB: connection-level `memory_limit` config, applies to data load too, not just queries; SQLite: best-effort `PRAGMA soft_heap_limit`, process-global). 12 tests verifying both engines under a deliberately expensive query, normal queries unaffected, read-only guarantees undisturbed. |
| Prompt-injection defense (malicious cell/column content) | ✅ (P0 remediation, closed this session) — see `backend/docs/security/prompt-injection-trust-boundary.md`. System-prompt trust-boundary instruction + a per-message "untrusted data" wrapper around every tool result. Verified both by automated tests (14 new, covering cell-value, column-name, and SQL-result/GROUP-BY injection vectors) and by one live run against the real Groq LLM with the exact adversarial payload — the model summarized the data correctly without obeying the embedded instruction. Residual risk: prompt-level mitigation is not a hard technical guarantee; revisit when any tool gains write/network side effects (see the doc's "Blast radius" section). |
| Rate limiting / abuse protection on the API itself | ❌ none (only the upstream LLM provider's own rate limit applies) |
| Read-only enforcement at a data-source level | N/A — no writable data source exists yet; relevant once SQL lands |

## 8. Existing performance profile

- Validated at dataset scale: **4,000 rows** (the demo dataset). `max_rows` config caps
  uploads at 500,000 rows, but **that ceiling has never been exercised or benchmarked**
  — no data point exists for 100K/1M/10M+ row behavior.
- Upload pipeline: ~600–700ms server-side for the demo file, dominated by `openpyxl`'s
  pure-Python XML parsing (measured this session; see `reports/final-qa-report.md`).
- Individual tool calls: 1–9ms (in-memory pandas ops on 4,000 rows).
- Real LLM chat request: 3–20s typical (real Groq call), free-tier token-budget
  constraints previously caused failures now mitigated (see `reports/` history) but not
  eliminated — a sufficiently complex question can still need multiple round-trips.
- Whole dataset lives in one process's RAM as a single DataFrame — no chunking, no
  pushdown, no sampling, no streaming. This is the direct blocker for the "100 million
  rows" target scenario in the proposed architecture; it isn't a matter of tuning, it's
  an architecture the current system doesn't have at all.
- Single Uvicorn process (`--reload` in dev), no worker pool, no caching layer, no load
  testing performed.
- Frontend production bundle: 833KB JS (247KB gzip), not code-split (Recharts is the
  bulk of it) — flagged, not yet addressed.

## 9. File-ownership map (proposed, for Wave 1 discussion)

See `.agent/agent_registry.md` for the full proposed team/file mapping. Summary: every
existing file already has a clear, single natural owner under the proposed team
structure — no file currently needs two owners. The friction point is entirely
*greenfield* work (SQL, forecasting, clustering) that has no existing file to anchor to,
and the **shared core** (`agent.py`, `tool_router.py`, `providers.py`) that every new
tool-providing team needs to integrate with without stepping on each other — see
`.agent/dependency_graph.md`.

## 10. Summary of gaps vs. the proposed target architecture

| Team (proposed) | Existing foundation | Gap |
|---|---|---|
| DATA-ARCHITECT | `app/datasets/storage.py` (single-table, in-memory) | No multi-source/multi-table model, no formal `DataSource`/`Dataset`/`Schema` contracts |
| SQL-ENGINEER | none | Entirely new — no DB driver, no query builder/validator |
| LARGE-DATA | none | Entirely new — no chunking/pushdown/streaming/sampling |
| EDA-ANALYST | profiler/filtering/aggregation/comparison tools | Solid foundation, could be extended (segmentation, comparative analysis) |
| STATISTICS-ENGINEER | describe_data, correlation_analysis | No hypothesis testing, no CI, no effect size, no regression |
| FORECASTING-ENGINEER | none | Entirely new |
| ADVANCED-ANALYTICS | none | Entirely new |
| VISUALIZATION-ENGINEER | generate_chart (5 chart types) | No automatic chart-type selection heuristic, no box plot/heatmap/dashboard |
| AGENT-ARCHITECT | agent.py (loop, dedup, stagnation-stop, constraint prompt) | Working v1 — needs to absorb new tools without regressing benchmark |
| TOOLING-ENGINEER | tool_router.py (10 tools, JSON-schema registry) | Pattern exists and scales; needs a contract doc so new tools stay consistent |
| REASONING/VALIDATION-ENGINEER | partially: constraint-checking system prompt + compare_periods coverage warnings | No independent/automated validation layer — currently relies on prompting, not a separate verification pass |
| CONTEXT/TOKEN-ENGINEER | partially: dedup + reuse notices + summary-only insights bundle | No token-budget accounting, no context pruning/summarization across turns |
| BUSINESS-ANALYST | insights.py, report.py | Reasonable v1; no formal root-cause-analysis or contribution-analysis tooling |
| REPORTING-ENGINEER | report.py (structured JSON), frontend Markdown export | No PDF/Excel export, no provenance/methodology section |
| QA-ENGINEER | 86 pytest tests, no e2e automation | Solid unit coverage; no browser/e2e suite, no load tests |
| DATA-VALIDATION-ENGINEER | ad hoc (this session's manual cross-checks against independent pandas calls) | No systematic ground-truth benchmark harness yet |
| BENCHMARK-ENGINEER | 7-question manual benchmark run this session (documented in conversation, not in a file) | No repeatable benchmark suite/runner exists in the repo |
| SECURITY-ENGINEER | upload/path/CORS/error-sanitization controls, plus (as of this session) SQL injection/read-only/resource-exhaustion defense and prompt-injection mitigation — see section 7 | No authentication/authorization still |
| PERFORMANCE-ENGINEER | ad hoc timing this session | No formal benchmark harness, no tests beyond 4,000 rows |
| DEVOPS-ENGINEER | none | No Dockerfile, no CI/CD, no monitoring/health-check beyond `/api/health` |
| DOCUMENTATION-ENGINEER | README.md, docs/architecture.md, docs/agent-tools.md, reports/ | Good foundation at MVP scale; will need updates as new capabilities land |

No feature work has been implemented as part of this audit, per Wave 0 rules.
