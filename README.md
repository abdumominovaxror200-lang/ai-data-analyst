# AI Data Analyst

An AI agent that analyzes CSV/Excel datasets like a business analyst would: it profiles the
data, answers natural-language questions, finds anomalies, charts trends, and writes up
findings — with every number backed by a real Python calculation, never guessed by the LLM.

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
  Upload, Overview, AI Analyst chat, Charts, Insights, Anomalies, Report.
- **Backend**: FastAPI. Thin route handlers over a service layer; Pydantic schemas for every
  request/response.
- **Data**: pandas + NumPy, openpyxl for `.xlsx`. No database — datasets live in memory for the
  process lifetime, addressed by a generated UUID (see [`docs/architecture.md`](docs/architecture.md)
  for the reasoning).
- **AI**: an OpenAI-compatible tool-calling agent. The provider is swappable — OpenAI,
  OpenRouter, Groq, DeepSeek, or any other OpenAI-compatible `/chat/completions` API — via three
  environment variables, no code change.

See [`docs/architecture.md`](docs/architecture.md) for the full system design.

## AI agent architecture

`DataAnalystAgent` (`backend/app/agent/agent.py`) runs a standard tool-calling loop: it sends the
user's question plus dataset *metadata only* (shape, column names/types — never raw rows) to the
LLM along with a JSON-schema list of available tools. If the LLM requests a tool call, the
`ToolRouter` executes the corresponding deterministic Python function against the real dataset
and the result is fed back into the conversation. This repeats (bounded at 6 iterations) until
the LLM produces a final answer. See [`docs/agent-tools.md`](docs/agent-tools.md) for the full
tool catalogue.

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

70 tests cover upload validation, every analysis tool, API error handling, malicious-file
handling, the agent's tool-calling / no-hallucination guarantees, and JSON-serialization edge
cases (e.g. datetime columns flowing through tool results).

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
5. **Charts** tab: build a bar chart of `product` vs `revenue` (sum).
6. **Anomalies** tab: run IQR detection on `profit` — it finds the seeded negative-profit
   data-entry error.
7. **Report** tab: generate and download the Markdown report.

## Security considerations

- Uploads are restricted to `.csv`/`.xlsx`, capped at 25 MB and 500,000 rows by default
  (`MAX_UPLOAD_MB`, `max_rows` in `app/config.py`).
- Filenames are never used as filesystem paths — storage paths are built entirely from a
  server-generated UUID, so path traversal via a crafted filename is structurally impossible
  (see `backend/tests/test_malicious_files.py`).
- Files are parsed with pandas/openpyxl only — no `pickle`, no macro execution, no code from
  the uploaded file is ever run.
- Datasets are process-lifetime, in-memory, and never written anywhere outside
  `STORAGE_DIR/uploads/<uuid>.<ext>`.
- `.env` files are gitignored; nothing in `app/` logs API keys, secrets, or full dataset
  contents (see the logging calls in `app/api/*.py`).
- CORS is restricted to the configured frontend origin(s) only.

## Limitations

- **No persistence.** Datasets live in memory for the backend process's lifetime — restarting
  the server or reloading the frontend loses the current dataset (by design for an MVP; see
  [`docs/architecture.md`](docs/architecture.md#why-no-database-for-v1)).
- **Single dataset per session.** No joins across multiple uploaded files yet.
- **Free-tier LLM rate limits.** Groq's free tier caps at 8,000 tokens/minute per account. A
  complex multi-part question can need 3-5 sequential tool-calling round-trips, which can brush
  against that ceiling. The agent retries with backoff and `LLM_REASONING_EFFORT=low` trims
  token usage substantially, but a paid tier (or a less token-hungry model) removes the
  constraint entirely.
- **No streaming.** Chat responses arrive as one complete response, not token-by-token.
- **No auth.** Anyone with network access to the backend can upload/query — fine for a local
  demo, not for a multi-tenant deployment as-is.

## Future roadmap

- Persist datasets (PostgreSQL + object storage) for multi-session / multi-user use beyond a
  single process lifetime.
- Streaming chat responses (SSE) instead of a single request/response per message.
- Authentication and per-user dataset isolation.
- Support for joining multiple uploaded datasets.
- Column-level PII detection/redaction before any data reaches the LLM's context.
- CI pipeline running the backend test suite + frontend build on every push.
- Dockerfile / docker-compose for one-command deployment.
