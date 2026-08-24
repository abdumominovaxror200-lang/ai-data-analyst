from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.datasets.storage import DatasetNotFoundError, get_dataset_store
from app.schemas import ReportRequest, ReportResponse
from app.tools.errors import ToolExecutionError
from app.tools.report import generate_report

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/reports", response_model=ReportResponse)
def create_report(request: ReportRequest) -> ReportResponse:
    store = get_dataset_store()
    try:
        record = store.get(request.dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        report = generate_report(record.df, record.id, record.original_filename)
    except ToolExecutionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info("report generated dataset_id=%s", request.dataset_id)
    return ReportResponse(dataset_id=request.dataset_id, generated_at=datetime.now(timezone.utc), report=report)
