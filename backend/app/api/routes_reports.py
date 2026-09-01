from __future__ import annotations

import io
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.datasets.storage import DatasetNotFoundError, get_dataset_store
from app.reports.excel_export import build_excel_report, excel_report_filename
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


@router.get("/reports/{dataset_id}/excel")
def download_excel_report(dataset_id: str) -> StreamingResponse:
    store = get_dataset_store()
    try:
        record = store.get(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    content = build_excel_report(record)
    filename = excel_report_filename(record)
    logger.info("excel report generated dataset_id=%s", dataset_id)
    return StreamingResponse(
        io.BytesIO(content),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
