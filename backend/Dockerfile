# syntax=docker/dockerfile:1
#
# Multi-stage build for the FastAPI backend.
#
#   builder  — installs Python deps into a venv (needs a compiler toolchain for
#              scientific packages like scipy/statsmodels/scikit-learn/duckdb
#              in case a prebuilt wheel isn't available for the target platform).
#   runtime  — copies only the venv + app code into a clean slim image, runs as
#              a non-root user. No secrets, no .env file, no test/doc files.
#
# Configuration is entirely via environment variables at `docker run`/compose
# time (see backend/app/config.py / ../.env.example) — nothing is baked in.

ARG PYTHON_VERSION=3.13

FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /build

# build-essential is only needed here, in the builder stage, in case pip has to
# fall back to a source build for one of the scientific deps on this platform.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


FROM python:${PYTHON_VERSION}-slim AS runtime

# curl is needed only for the HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Only the application package is copied in — no tests/, docs/, scripts/, or
# .env* files end up in the image (see .dockerignore for the full exclusion list).
COPY app ./app

# Pre-create the storage dir (this is where docker-compose mounts the named
# volume for uploaded datasets) and hand ownership to the non-root user before
# switching to it, so the volume inherits correct permissions on first mount.
RUN mkdir -p /app/storage && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Hits the real /api/health route registered in app/main.py (routes_health.py).
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fs "http://localhost:${PORT}/api/health" || exit 1

# Shell form so ${PORT} is expanded; defaults to 8000 if not supplied.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
