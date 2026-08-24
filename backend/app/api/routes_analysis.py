from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException

from app.agent.tool_router import ToolRouter
from app.datasets.storage import DatasetNotFoundError, get_dataset_store
from app.schemas import AnalysisRequest, AnalysisResponse
from app.tools.errors import ToolExecutionError

logger = logging.getLogger(__name__)
router = APIRouter()
_tool_router = ToolRouter()


@router.post("/analysis", response_model=AnalysisResponse)
def run_analysis(request: AnalysisRequest) -> AnalysisResponse:
    store = get_dataset_store()
    try:
        record = store.get(request.dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    start = time.perf_counter()
    try:
        result = _tool_router.execute(request.tool, record, request.params)
    except ToolExecutionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info("analysis tool=%s dataset_id=%s elapsed_ms=%.1f", request.tool, request.dataset_id, elapsed_ms)
    return AnalysisResponse(tool=request.tool, result=result, elapsed_ms=round(elapsed_ms, 2))
