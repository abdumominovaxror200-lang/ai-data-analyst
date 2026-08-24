# Decisions Log

## Already decided (context for new agents — do not re-litigate without cause)

| Decision | Why | When |
|---|---|---|
| No database for the original MVP; in-memory `DatasetStore` | Scoped to a single-session demo; see `docs/architecture.md#why-no-database-for-v1` | Initial build |
| LLM provider = any OpenAI-compatible `/chat/completions` API, currently Groq `openai/gpt-oss-120b` | Free tier available for demo/testing; swappable via 3 env vars | Initial build |
| Tool outputs must be summary-only where they can grow unbounded (e.g. `generate_business_insights` excludes raw anomaly rows) | A real 413 "payload too large" failure was traced to this; fixed and regression-tested | This session |
| Agent has duplicate-call detection + stagnation early-stop, `MAX_TOOL_ITERATIONS=10` | A real benchmark run showed the agent burning all 6 iterations on repeated identical calls with no final answer; fixed and verified against real Groq traffic | This session |
| System prompt requires explicit constraint-mismatch disclosure (row count, date coverage, missing columns) rather than silent substitution | A real benchmark run showed the agent silently analyzing a 4,000-row dataset against a question describing a 10-million-row database, without saying so | This session |

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
