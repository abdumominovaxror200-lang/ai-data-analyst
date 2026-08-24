from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import routes_analysis, routes_chat, routes_datasets, routes_health, routes_reports
from app.config import get_settings
from app.datasets.validation import ValidationError
from app.logging_config import configure_logging, new_request_id, request_id_var
from app.tools.errors import ToolExecutionError

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Data Analyst API", version="0.1.0")

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


app.include_router(routes_health.router, prefix="/api", tags=["health"])
app.include_router(routes_datasets.router, prefix="/api", tags=["datasets"])
app.include_router(routes_analysis.router, prefix="/api", tags=["analysis"])
app.include_router(routes_chat.router, prefix="/api", tags=["chat"])
app.include_router(routes_reports.router, prefix="/api", tags=["reports"])
