# Initial Audit — AI Data Analyst

Date: 2026-08-24

## 1. Context

`C:\Users\Admin2025\ai_agent` is the root of an **existing, unrelated project**: "Super Agent AI", a
Uzbek-language personal Telegram bot / content-generation tool (video/image/audio generation,
a Cyberpunk desktop GUI, a web GUI, 42 tools in `tools.py`, `second_brain.py` memory, live API
keys in `.env`). That repository is not under git version control and contains large generated
media files (multi-MB `.mp4`/`.mp3`) unrelated to data analysis.

The task requested here — an **AI Data Analyst** (FastAPI + React, CSV/Excel statistical
analysis agent) — shares no code, dependencies, or purpose with that project. Building it in the
repo root would mix two unrelated products in one uncontrolled directory with no git safety net.

**Decision (confirmed with user):** build the new project in an isolated subfolder,
`ai_agent/ai-data-analyst/`, with its own git repository, its own `.env`, its own dependencies.
Nothing in the parent directory is read, modified, or referenced (in particular, the parent
`.env` and API keys are never inspected, per the parent project's own CLAUDE.md rule and this
project's need for a clean, portfolio-ready boundary).

## 2. Existing reusable components

None. This is a greenfield project. The parent repo's `ai_core.py` shows a useful *pattern*
(multi-provider LLM abstraction: Gemini/OpenRouter/Groq/DeepSeek) worth emulating conceptually,
but no code is reused or imported — the new project gets its own minimal provider abstraction
scoped to what an analyst agent needs (tool/function calling).

## 3. Missing components (everything, by design)

- Backend: FastAPI app, Pydantic schemas, dataset storage/validation, analysis tool library,
  LLM agent + tool router, chat/report endpoints, logging, tests.
- Frontend: React + TypeScript dashboard (upload, overview, chat, charts, insights, anomalies,
  report export).
- Demo dataset with intentional anomalies.
- Docs: README, architecture.md, agent-tools.md, implementation-report.md.

## 4. Technical risks

| Risk | Mitigation |
|---|---|
| LLM hallucinating numbers | Hard separation: LLM never computes — it only selects tools and narrates results returned by deterministic Python functions. Enforced by an agent loop that only reads numbers from tool-call outputs and a test that asserts this. |
| Malicious/oversized uploads | Extension allow-list (.csv/.xlsx/.xls), size cap, sanitized filenames (uuid-based, no user-controlled path components), pandas/openpyxl read only (no macro execution, no `pickle`), try/except around parsing with typed errors. |
| No LLM API key available in this environment | Provider abstraction with an OpenAI-compatible HTTP client (works with OpenAI, OpenRouter, Groq, DeepSeek — all OpenAI-compatible `/chat/completions` APIs) configured purely via `.env`; a `MockProvider` is used in automated tests so the suite runs with zero network/API dependency. |
| Scope creep (enterprise features) | No database, no auth, no multi-tenancy for MVP. In-memory + local-disk dataset store keyed by UUID, process-lifetime only, as instructed. |
| Windows path/dev environment quirks | Use `pathlib`, avoid shell-specific path assumptions, verify with `pytest` on Windows directly. |

## 5. Recommended architecture

As specified in the task brief — React/TS frontend, FastAPI backend, Pandas/NumPy analysis
tools, an LLM agent with function/tool calling routed through a `ToolRouter`, no database for
v1 (local temp storage of uploaded datasets, addressable by dataset id).

LLM provider: OpenAI-compatible chat-completions client (`backend/app/agent/providers.py`),
configurable base URL + model + key, so swapping OpenAI ↔ OpenRouter ↔ Groq ↔ DeepSeek is a
`.env` change, not a code change.

## 6. Implementation plan

1. Backend scaffold: FastAPI app, config, logging, Pydantic schemas.
2. Dataset storage + validation (upload, sanitize, profile).
3. Analysis tool library (profiler, stats, filter, group/aggregate, compare periods,
   correlation, anomaly detection, chart-data generation, insights, report).
4. Agent + tool router + provider abstraction (OpenAI-compatible + Mock for tests).
5. API routes wiring it together (`/api/datasets/upload`, `/api/datasets/{id}`,
   `/api/analysis`, `/api/chat`, `/api/reports`, `/api/health`).
6. Backend tests (pytest) — upload, profiling, stats, filtering, aggregation, anomalies,
   chart generation, validation, malicious files, agent tool-execution/no-hallucination check.
7. Demo dataset generator (`sales_data.xlsx`, realistic + seeded anomalies).
8. Frontend scaffold (Vite + React + TS + Tailwind), API client, upload/dashboard/chat/report UI.
9. End-to-end manual verification in browser preview.
10. Docs: README, architecture.md, agent-tools.md, implementation-report.md.

## 7. Estimated complexity by module

| Module | Complexity | Notes |
|---|---|---|
| Dataset storage/validation | Low | uuid-keyed temp files, extension/size checks |
| Tool library | Medium | 10 tools, mostly pandas one-liners wrapped with validation |
| Agent + tool router | Medium-High | tool-calling loop, JSON schema per tool, provider abstraction |
| API layer | Low | thin FastAPI routes over services |
| Frontend dashboard | Medium-High | most time-consuming piece for a "polished SaaS" look |
| Tests | Medium | breadth over depth, one focused test per tool |
| Docs | Low | mechanical once code exists |

No blocking issues found. Proceeding directly to implementation.
