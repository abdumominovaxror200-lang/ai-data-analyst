from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import routes_analysis, routes_chat, routes_datasets, routes_health, routes_reasoning, routes_reports
from app.config import get_settings
from app.datasets.validation import ValidationError
from app.datasets.storage import get_dataset_store
from app.logging_config import configure_logging, new_request_id, request_id_var
from app.tools.errors import ToolExecutionError

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


def validate_deployment_policy(config) -> None:
    if config.deployment == "public" and config.llm_egress_mode == "local_only":
        raise RuntimeError(
            "DEPLOYMENT=public cannot use LLM_EGRESS_MODE=local_only; "
            "a public backend must use external_redacted or llm_disabled."
        )
    if config.deployment == "public" and config.llm_egress_mode == "external_redacted":
        logger.info("public deployment privacy policy active: raw-value LLM egress is disabled")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    validate_deployment_policy(settings)
    get_dataset_store().sweep_expired()
    yield

app = FastAPI(title="AI Data Analyst API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    token = request_id_var.set(new_request_id())
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s status=%s elapsed_ms=%.1f", request.method, request.url.path, response.status_code, elapsed_ms)
    request_id_var.reset(token)
    return response


@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ToolExecutionError)
async def tool_error_handler(request: Request, exc: ToolExecutionError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Last-resort safety net: any exception type we didn't anticipate (a bug, an
    # unexpected third-party error, ...) must still produce a clean JSON response, never
    # a raw traceback or Python exception text leaking to the client. Full detail goes
    # to the server log only. More specific handlers above (and FastAPI's own
    # HTTPException handling) take precedence over this one.
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "An unexpected error occurred. Please try again."})


app.include_router(routes_health.router, prefix="/api", tags=["health"])
app.include_router(routes_datasets.router, prefix="/api", tags=["datasets"])
app.include_router(routes_analysis.router, prefix="/api", tags=["analysis"])
app.include_router(routes_chat.router, prefix="/api", tags=["chat"])
app.include_router(routes_reasoning.router, prefix="/api", tags=["reasoning"])
app.include_router(routes_reports.router, prefix="/api", tags=["reports"])
