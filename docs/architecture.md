# Architecture

## System overview

```
┌─────────────┐      HTTP/JSON      ┌──────────────┐
│   Browser   │ ──────────────────▶ │   FastAPI    │
│ React + TS  │ ◀────────────────── │   backend    │
└─────────────┘                     └──────┬───────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
              DatasetStore            DataAnalystAgent          ToolRouter
           (in-memory, uuid-keyed)   (tool-calling loop)     (dispatch by name)
                    │                       │                       │
                    │                 LLMProvider              pandas/NumPy
                    │            (OpenAI-compatible API)        tool functions
                    └───────────────────────┴───────────────────────┘
                                  operate on the same pd.DataFrame
```

## Request flow

1. **Upload** (`POST /api/datasets/upload`): file is validated (extension, size), parsed into a
   `pandas.DataFrame`, and stored in an in-memory `DatasetStore` keyed by a generated UUID. The
   raw bytes are also written to `storage/uploads/<uuid>.<ext>` — never at a user-controlled path.
2. **Direct analysis** (`POST /api/analysis`): the frontend can call any tool directly by name
   (used by the Charts/Anomalies tabs, which need a specific, deterministic tool call rather than
   an LLM's judgment about which tool to use).
3. **Chat** (`POST /api/chat`): the message goes to `DataAnalystAgent.ask()`, which runs the
   tool-calling loop against the configured `LLMProvider` and returns a narrated answer plus the
   tool calls and any chart data produced along the way.
4. **Report** (`POST /api/reports`): directly composes `profile_dataset` +
   `generate_business_insights` into a structured report — no LLM involved, so it's available
   even without an `LLM_API_KEY` configured.

## Why no database for v1

The brief explicitly asks not to introduce a database unless genuinely necessary for the MVP.
A single analyst session — upload a file, explore it, close the tab — doesn't need durability
across process restarts, and adding PostgreSQL would mean migrations, connection pooling, and
ORM boilerplate with no payoff for a portfolio demo. `DatasetStore` is a small, swappable
abstraction (`backend/app/datasets/storage.py`) — if persistence becomes a real requirement, it
can be re-implemented behind the same `save()`/`get()` interface without touching any route or
tool code.

## LLM provider abstraction

`backend/app/agent/providers.py` defines `LLMProvider` as an abstract `complete(messages, tools)
-> ProviderResponse`. The concrete `OpenAICompatibleProvider` implementation talks to any
`/chat/completions`-compatible endpoint — this covers OpenAI, OpenRouter, Groq, and DeepSeek
without provider-specific code, because they all speak the same wire format. Switching providers
is three environment variables (`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`) in `backend/.env`.

A second implementation, `MockProvider`, returns a scripted sequence of responses with no network
call — this is what the test suite uses to verify the tool-calling loop and the
no-hallucination guarantee deterministically, without needing a live API key in CI.

## Data safety

| Concern | Mitigation |
|---|---|
| Path traversal via filename | Storage path is built from a server-generated UUID + validated extension only; the original filename is sanitized (`sanitize_display_name`) and used for *display* metadata only, never as a path component. |
| Oversized uploads | `MAX_UPLOAD_MB` (default 25) checked before parsing; `max_rows` (default 500,000) checked after parsing. |
| Malformed files | Parsing is wrapped in try/except; any pandas/openpyxl exception becomes a clean `400` with a message, never a crash. |
| Arbitrary code execution | Files are only ever passed to `pandas.read_csv` / `pandas.read_excel(engine="openpyxl")` — no `pickle`, no macro execution, no `eval`. |
| Secret exposure | `.env` is gitignored; nothing in the codebase logs `LLM_API_KEY` or full dataset contents (see the `logger.info(...)` calls in `app/api/*.py`, which log ids/shapes only). |

## Frontend structure

```
frontend/src/
  api/client.ts        typed Axios wrappers for every backend endpoint
  types/index.ts        TypeScript mirrors of the backend Pydantic schemas
  components/
    UploadView.tsx       landing page / drag-drop upload
    DashboardShell.tsx    sidebar nav + header once a dataset is loaded
    ChartRenderer.tsx     maps backend ChartData -> Recharts components
    primitives.tsx        Card/Button/Badge/StatTile/EmptyState/ErrorBanner
    tabs/
      OverviewTab.tsx      profile + column table
      ChatTab.tsx          AI Analyst conversation
      ChartsTab.tsx        build-your-own chart
      InsightsTab.tsx      generate_business_insights, rendered
      AnomaliesTab.tsx     detect_anomalies, rendered
      ReportTab.tsx        generate_report + Markdown export
```

State lives in `App.tsx` (the uploaded dataset's profile + active tab) — there's no global state
library because a single dataset and six tabs don't warrant one.
