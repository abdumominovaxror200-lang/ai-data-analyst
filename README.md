# AI Data Analyst

An AI agent that analyzes CSV/Excel datasets like a business analyst would: it profiles the
data, answers natural-language questions, finds anomalies, charts trends, and writes up
findings — with every number backed by a real Python calculation, never guessed by the LLM.

39 deterministic analysis tools (profiling, SQL, EDA, statistics, regression, forecasting,
clustering, customer segmentation, data quality, general-purpose aggregation/reporting) sit
behind two API surfaces: a fast direct tool-calling agent (`/api/chat`), and a bounded,
evidence-classified reasoning layer (`/api/reason`) that separates parsing, planning,
execution, verification, hypothesis evaluation, causal-language guarding, recommendation
grounding, and epistemic self-checking into distinct, individually testable stages. See
[Two ways to ask a question](#two-ways-to-ask-a-question) below.

## This is an agent, not a chatbot

The distinction matters and it's the whole point of this project. A chatbot given a dataset
description will happily *generate plausible-looking numbers* — a hallucination risk that makes
LLMs unsafe for real data analysis on their own. This project instead wires the LLM into an
**agent loop**: the model's only job is to decide *which* deterministic tool to call and *how to
explain* what that tool returns. It never does arithmetic itself.

```
User
 ↓
AI Analyst Agent
 ↓
Tool Selection
 ↓
Deterministic Python/Pandas Analysis
 ↓
Structured Results
 ↓
LLM Interpretation
 ↓
Charts + Insights + Recommendations
```

**LLMs do not perform the numerical calculations directly. The agent calls deterministic
Python/Pandas tools and uses their results to generate explanations.**

This has been verified end-to-end against a real LLM provider (Groq): every number the model
cited in a live multi-part analysis ("find the top 5 products by revenue, identify anomalies,
explain the biggest problems, give 3 recommendations") was cross-checked against an independent,
direct pandas computation — with zero deviation. See
[`reports/final-qa-report.md`](reports/final-qa-report.md) for the full verification.

## What it does

Upload a spreadsheet and:

- See an automatic profile — row/column counts, types, missing data, duplicates.
- Ask questions in plain language ("Which region underperformed?", "Find anomalies",
  "What's correlated with revenue?") and get answers grounded in computed results.
- Build charts (line, bar, histogram, scatter, pie) from any columns.
- Run outlier detection (IQR or z-score) on any numeric column.
- Get a structured business report with key findings, exportable as Markdown.

## Problem solved

Spreadsheets are easy to collect and hard to interrogate — most people can't write a pandas
`groupby` or a z-score outlier check, and pasting numbers into a general-purpose chatbot risks
the AI silently inventing plausible-looking figures. This project gives a non-technical user a
conversational interface to real statistical analysis, while guaranteeing every number they see
came from deterministic code, not from the LLM's imagination.

## Architecture

```
User → React/TS dashboard → FastAPI → AI Analyst Agent → Tool Router → pandas/numpy tools
                                                                              ↓
                                                        structured JSON result (numbers, charts)
                                                                              ↓
                                                   Agent narrates the result → Dashboard / Report
```

- **Frontend**: React + TypeScript (Vite), Tailwind CSS, Recharts. A single-page dashboard —
  Upload, Overview, AI Analyst chat, Deep Reasoning, Charts, Insights, Anomalies, Report.
- **Backend**: FastAPI. Thin route handlers over a service layer; Pydantic schemas for every
  request/response.
- **Data**: pandas + NumPy, openpyxl for `.xlsx`, DuckDB for read-only ad hoc SQL. No database —
  datasets live in memory for the process lifetime, addressed by a generated UUID (see
  [`docs/architecture.md`](docs/architecture.md) for the reasoning). A separate, tested
  `app/large_data/` package (chunked reading, reservoir/Bernoulli sampling, chunked aggregation,
  memory guarding — see [Large-data engine](#large-data-engine-not-yet-wired-into-upload) below)
  exists for datasets far larger than fit in memory, but is **not yet connected to the upload
  path** — documented honestly, not silently implied.
- **AI**: an OpenAI-compatible tool-calling agent. The provider is swappable — OpenAI,
  OpenRouter, Groq, DeepSeek, or any other OpenAI-compatible `/chat/completions` API — via three
  environment variables, no code change.
- **Deployment**: Dockerfiles for backend/frontend, `docker-compose.yml`, and a GitHub Actions CI
  workflow (backend pytest + frontend typecheck/build/lint) — see [`docs/deployment.md`](docs/deployment.md).

See [`docs/architecture.md`](docs/architecture.md) for the full system design.

## AI agent architecture

`DataAnalystAgent` (`backend/app/agent/agent.py`) runs a standard tool-calling loop: it sends the
user's question plus dataset *metadata only* (shape, column names/types — never raw rows) to the
LLM along with a JSON-schema list of available tools. If the LLM requests a tool call, the
`ToolRouter` executes the corresponding deterministic Python function against the real dataset
and the result is fed back into the conversation. This repeats (bounded iterations) until
the LLM produces a final answer. See [`docs/agent-tools.md`](docs/agent-tools.md) for the full
tool catalogue.

## Two ways to ask a question

| | `/api/chat` | `/api/reason` |
|---|---|---|
| What it is | The original direct tool-calling loop (above) | A bounded, multi-stage reasoning pipeline built on top of the same tools |
| LLM calls | As many as the tool loop needs | Exactly 3 structured calls (parse / plan / synthesize) plus the tool-execution loop |
| Output | An answer + the tool calls that produced it | Answer **plus** structured `Evidence`, classified `Finding`s (FACT/CALCULATED_RESULT/STATISTICAL_RESULT/HYPOTHESIS/ASSUMPTION/UNKNOWN), `Limitation`s, `Hypothesis` objects with evidence-derived status, a grounded `Recommendation` (confidence capped by evidence strength), and a list of any `principle_violations` a set of 10 machine-checkable epistemic-honesty checks flagged |
| Guardrails | Tool-level only (SQL read-only, resource limits) | All of the above, plus: category-filtered tool access (a hallucinated or overly-eager LLM category choice can never reach an out-of-category tool), a 3-layer causal-language guard (unhedged causal claims like "X caused Y" are rejected unless a specific hypothesis reached `status: "supported"` from real statistical evidence), and recommendation-confidence capping tied to evidence strength |
| Where in the UI | **AI Analyst** tab | **Deep Reasoning** tab |
| When to use | Fast, general-purpose questions | Business-critical questions where you want to see *why* the system believes something, not just the answer |

Both paths call the same 39 real tools and never let the LLM compute a number itself — `/api/reason`
adds an evidence/finding/hypothesis bookkeeping layer and several honesty guardrails on top, at
the cost of being slower (3 fixed LLM calls vs. as many as the loop needs).

## How numerical accuracy is maintained

**The LLM never computes numbers — it only selects tools, reads their output, and narrates it.**

Concretely:

1. The agent's dataset context message contains only metadata (row/column counts, column
   names and types) — the raw DataFrame is never serialized into a prompt.
2. Every numeric claim the agent can make has to originate from a `tool_call` result appended
   to the conversation by `ToolRouter.execute(...)`, which calls straight into pandas/NumPy.
3. `backend/tests/test_agent_tools.py` asserts this architecturally: it checks that a
   distinctive token present only in raw cell values never appears in what's sent to the LLM
   before a tool call, and that the number in a scripted final answer matches the number an
   independent, direct call to the same tool produces.
4. Every other tool (`backend/app/tools/*.py`) has its own unit tests asserting the computed
   values match hand-checked expectations.

## Tech stack

| Layer | Choice |
|---|---|
| Frontend | React 19, TypeScript, Vite, Tailwind CSS, Recharts, Axios, react-markdown |
| Backend | FastAPI, Pydantic v2, Uvicorn |
| Data | pandas, NumPy, openpyxl |
| AI | Any OpenAI-compatible chat-completions API (OpenAI / OpenRouter / Groq / DeepSeek) |
| Tests | pytest, FastAPI `TestClient` |

## How to run locally

Prerequisites: Python 3.11+, Node 18+.

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate      # Windows; use `source venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
copy .env.example .env     # then set LLM_API_KEY to enable the AI Analyst chat tab
uvicorn app.main:app --reload --port 8010
```

The API is now at `http://localhost:8010/api` (docs at `/docs`). Without an `LLM_API_KEY`,
every tab except **AI Analyst** works fully — that tab shows a clear "AI provider not
configured" message instead of failing silently.

**Getting an API key (free option):** [Groq](https://console.groq.com/keys) has a free tier and
was used to verify this project end-to-end. In `backend/.env`:

```
LLM_API_KEY=gsk_...your key...
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b
LLM_REASONING_EFFORT=low   # cuts reasoning-token overhead ~90%, keeps you inside free-tier limits
```

Any other OpenAI-compatible provider (OpenAI, OpenRouter, DeepSeek) works the same way — just
change these three-to-four values, no code change.

### Frontend

```bash
cd frontend
npm install
copy .env.example .env     # VITE_API_BASE_URL should point at the backend above
npm run dev
```

Open `http://localhost:5173`.

### Demo dataset

[`data/demo/sales_data.xlsx`](data/demo/sales_data.xlsx) is **entirely synthetic** — 4,000
generated B2C retail transactions (Jan 2024–Dec 2025), no real business or personal data.
It's seeded with intentional anomalies so the analysis tools have something real to find:

- 6 bulk-order revenue spikes (unrealistically large single-day quantities)
- 8 data-entry errors where cost exceeds revenue (negative profit)
- 25 rows with a missing `customer_id` (a realistic data-quality gap)

Upload it to try every feature immediately. Regenerate it (same seed, so results stay
reproducible) with:

```bash
cd backend
venv\Scripts\python scripts\generate_demo_data.py
```

### Running tests

```bash
cd backend
venv\Scripts\python -m pytest
```

913 tests cover upload validation, every one of the 39 analysis tools, API error handling,
malicious-file handling, the agent's tool-calling / no-hallucination guarantees,
JSON-serialization edge cases, the full reasoning pipeline (premise validation, causal-language
guarding, recommendation grounding, epistemic checks), and two scripted end-to-end benchmarks
(`professional_benchmark.json`, `final_100_cases.json` — 100 and 102 cases respectively across
the full analyst-capability taxonomy; see [`.agent/final_100_case_benchmark.md`](.agent/final_100_case_benchmark.md)
for the latest measured results and honesty caveats). A further 74 tests (real-LLM-provider
benchmark cases and the 100M-row large-data benchmark) are gated behind opt-in environment
variables (`RUN_REAL_LLM_BENCHMARK=1`, `RUN_100M_BENCHMARK=1`) since they need a live API key
or several minutes and multiple GB of disk/RAM — skipped by default, not silently absent.

## Example questions

- "Analyze this dataset."
- "What are the most important business insights?"
- "Which products generate the most revenue?"
- "Which region performs worst?"
- "Show revenue trends over time."
- "Find anomalies."
- "Which variables are strongly correlated?"
- "What are the biggest problems in this dataset?"

## Demo walkthrough

1. Start both servers (above) and open `http://localhost:5173`.
2. Upload `data/demo/sales_data.xlsx`.
3. **Overview** tab: confirm 4,000 rows / 11 columns and column type breakdown.
4. **AI Analyst** tab: ask "Analyze this dataset and summarize the key findings." (requires
   `LLM_API_KEY` in `backend/.env` — without it the tab shows a clear "AI provider not
   configured" message rather than failing silently).
5. **Deep Reasoning** tab: ask "Did our new pricing strategy cause this quarter's revenue
   increase?" — watch the answer come with explicit Findings, Limitations, and Hypotheses
   sections, and notice unhedged causal language gets rejected unless the evidence actually
   supports it.
6. **Charts** tab: build a bar chart of `product` vs `revenue` (sum).
7. **Anomalies** tab: run IQR detection on `profit` — it finds the seeded negative-profit
   data-entry error.
8. **Report** tab: generate and download the Markdown report.

## Large-data engine (not yet wired into upload)

`backend/app/large_data/` is a separate, independently tested package for datasets far larger
than comfortably fit in memory: chunked CSV reading, exact reservoir/Bernoulli sampling from a
file stream, chunked group-aggregation, and a memory guard that raises before the process is at
real risk of OOM. It has been benchmarked for real at 100K/1M/10M/100M synthetic rows (see
`backend/app/large_data/benchmark_100m_results.json`) — real measured numbers, not projections,
for every stage except the full naive-load-to-100M-rows ceiling (extrapolated from a 2M-row
measurement, since actually loading 100M rows naively would OOM the benchmark machine by design).

**This package is not yet connected to the dataset-upload path** — today's upload endpoint
(`app/datasets/storage.py`) still does one full `pd.read_csv`/`pd.read_excel`, capped at
`MAX_UPLOAD_MB`/`max_rows` (25 MB / 500,000 rows by default). Wiring `large_data` into the real
upload flow is the single largest documented gap in this project — see
[`.agent/production-readiness.md`](.agent/production-readiness.md) for the proposed design.

## Security considerations

- Uploads are restricted to `.csv`/`.xlsx`, capped at 25 MB and 500,000 rows by default
  (`MAX_UPLOAD_MB`, `max_rows` in `app/config.py`).
- Filenames are never used as filesystem paths — storage paths are built entirely from a
  server-generated UUID, so path traversal via a crafted filename is structurally impossible
  (see `backend/tests/test_malicious_files.py`).
- Files are parsed with pandas/openpyxl only — no `pickle`, no macro execution, no code from
  the uploaded file is ever run.
- Ad hoc SQL (`run_sql_query`/`explain_sql_query` tools) runs against a read-only DuckDB/SQLite
  view — a single `SELECT` statement only; write/DDL statements and stacked statements are
  rejected before execution, verified by dedicated tests and by adversarial benchmark cases that
  literally attempt `DELETE`/`DROP TABLE` and confirm they're refused, not silently no-opped.
- Data from the uploaded file (cell values, column names) is treated as untrusted input
  throughout the LLM-facing prompt construction — a prompt-injection payload embedded in a cell
  is inert data to the synthesizer, never an instruction, verified by dedicated adversarial cases.
- Datasets are cached in memory and persisted only in the server-owned
  `STORAGE_DIR/uploads/<uuid>.<ext>` namespace with SQLite metadata and integrity hashes.
- `.env` files are gitignored; nothing in `app/` logs API keys, secrets, or full dataset
  contents (see the logging calls in `app/api/*.py`).
- CORS is restricted to the configured frontend origin(s) only.

## Limitations

- **Local persistence only.** Retained datasets survive backend restart and can be re-analyzed
  through their existing dataset ID. Reasoning transcripts and multi-user history are not persisted.
- **Single dataset per session.** No joins across multiple uploaded files yet.
- **Upload path caps out well below the large-data engine's tested scale.** See
  [Large-data engine](#large-data-engine-not-yet-wired-into-upload) above.
- **Free-tier LLM rate limits.** Groq's free tier caps at 8,000 tokens/minute and a daily token
  quota per account. A complex multi-part question, or `/api/reason`'s 3 fixed structured calls,
  can brush against those ceilings. The agent retries with backoff and `LLM_REASONING_EFFORT=low`
  trims token usage substantially, but a paid tier (or a less token-hungry model) removes the
  constraint entirely.
- **No streaming.** Chat responses arrive as one complete response, not token-by-token.
- **No auth.** Anyone with network access to the backend can upload/query — fine for a local
  demo, not for a multi-tenant deployment as-is.
- **Some structural gaps remain in the reasoning layer**, honestly documented rather than
  silently absent — see [`.agent/PROFESSIONAL_ANALYST_CAPABILITY_AUDIT.md`](.agent/PROFESSIONAL_ANALYST_CAPABILITY_AUDIT.md)
  for the full 20-area capability audit, including a fixed root-cause-analysis sequence for
  diagnostic questions, a general-purpose numerical sanity checker, and deterministic (not
  purely LLM-discretion) chart-type selection — all still open, prioritized items.

## Future roadmap

- Wire `app/large_data/` into the real upload path (the largest single documented gap).
- Persist datasets (PostgreSQL + object storage) for multi-session / multi-user use beyond a
  single process lifetime.
- Streaming chat responses (SSE) instead of a single request/response per message.
- Authentication and per-user dataset isolation.
- Support for joining multiple uploaded datasets.
- Column-level PII detection/redaction before any data reaches the LLM's context.
- A fixed, deterministic root-cause-analysis sequence for diagnostic questions (profile → decompose → compare periods → check segments → check outliers → test hypotheses), rather than leaving that sequencing entirely to the planner LLM.
- A general-purpose numerical sanity checker (magnitude/sign/units/denominator/order-of-magnitude) applied to every numeric finding, not only the specific checks that exist today.
