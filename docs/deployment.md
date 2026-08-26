# Deployment

Two ways to run this project:

- **Docker (this doc)** — one command, no local Python/Node setup, closest to
  how you'd run it on a server.
- **Local dev without Docker** — see the [README's "How to run
  locally"](../README.md#how-to-run-locally) section; that path is unchanged
  by anything here and is still the faster loop for active development
  (hot reload, `--reload` uvicorn, `npm run dev`).

This doc covers the Docker path only, so it doesn't duplicate/contradict the
README's local-dev instructions.

## What gets built

- `backend/Dockerfile` — FastAPI app served by Uvicorn, port 8000 inside the
  container.
- `frontend/Dockerfile` — Vite production build served as static files by
  nginx, port 80 inside the container. nginx proxies `/api/*` to the backend
  service so the browser only ever talks to one origin (no CORS to configure).
- `docker-compose.yml` at the repo root wires the two together, plus a named
  volume (`backend_storage`) so uploaded datasets survive container restarts.

## Deploy from a clean machine

Prerequisites: [Docker](https://docs.docker.com/get-docker/) with the Compose
plugin (`docker compose version` should print something; this was verified
against Docker 29.7.2).

1. **Clone the repo**

   ```bash
   git clone <repo-url>
   cd ai-data-analyst
   ```

2. **Configure environment variables**

   ```bash
   cp .env.example .env       # macOS/Linux
   copy .env.example .env     # Windows
   ```

   Open `.env` and set `LLM_API_KEY` (a free key works —
   [console.groq.com/keys](https://console.groq.com/keys), same as the
   README's local-dev instructions). Every field in `.env.example` has a
   one-line comment explaining it; all of it maps directly to
   `backend/app/config.py`. Without `LLM_API_KEY` the app still starts and
   every tab except **AI Analyst** works — that tab shows a clear "AI
   provider not configured" message instead of failing silently.

   `.env` is gitignored — never commit it. `.env.example` only ever holds
   placeholder text, never a real key.

3. **Build and start both services**

   ```bash
   docker compose up --build
   ```

   First run builds two images (backend, frontend) and starts them. The
   frontend container waits for the backend's `HEALTHCHECK` to report
   healthy before starting (see `depends_on: condition: service_healthy` in
   `docker-compose.yml`).

4. **Verify it's up**

   - Backend health check: open `http://localhost:8000/api/health` — expect
     `{"status": "ok"}`. Interactive API docs (FastAPI's default, unprefixed
     `/docs` route) are at `http://localhost:8000/docs` if you want to browse
     the API.
   - Frontend: open `http://localhost` — the dashboard should load. Upload
     [`data/demo/sales_data.xlsx`](../data/demo/sales_data.xlsx) to try it
     end-to-end (same demo dataset the README's walkthrough uses).

5. **Stop it**

   ```bash
   docker compose down          # stop containers, keep the storage volume
   docker compose down -v       # also delete the storage volume (uploaded datasets)
   ```

## Notes

- **Ports**: backend on host `8000`, frontend on host `80`. Change the
  left-hand side of the `ports:` mappings in `docker-compose.yml` if either
  port is already taken on your machine (e.g. `"8080:80"` for the frontend).
- **Storage**: uploaded datasets live in the `backend_storage` named Docker
  volume, mounted at `/app/storage` inside the backend container (matches
  `STORAGE_DIR` in `.env.example`, which resolves relative to `backend/` —
  i.e. `/app` in the container). They persist across `docker compose
  restart`/`up` but are removed by `docker compose down -v`.
- **Rebuilding after a code change**: `docker compose up --build` again — add
  `--force-recreate` if you also changed environment variables but Compose
  doesn't pick it up.
- **CI**: `.github/workflows/ci.yml` runs the backend pytest suite and the
  frontend typecheck/build/lint on every push/PR to `main`. It does not build
  or push these Docker images (out of scope for the "zero CI exists" gap this
  fills) — that would be a natural next step if this project moves to an
  automated deployment pipeline.
