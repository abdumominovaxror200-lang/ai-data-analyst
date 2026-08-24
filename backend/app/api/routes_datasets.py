from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.datasets.storage import DatasetNotFoundError, get_dataset_store
from app.datasets.validation import ValidationError
from app.schemas import ColumnInfo, DatasetProfile, UploadResponse
from app.tools.profiler import profile_dataset

logger = logging.getLogger(__name__)
router = APIRouter()


def _to_profile(dataset_id: str, filename: str, uploaded_at: datetime, df: pd.DataFrame) -> DatasetProfile:
    raw = profile_dataset(df)
    return DatasetProfile(
        dataset_id=dataset_id,
        filename=filename,
        uploaded_at=uploaded_at,
        rows=raw["rows"],
        columns=raw["columns"],
        column_info=[ColumnInfo(**c) for c in raw["column_info"]],
        numeric_columns=raw["numeric_columns"],
        categorical_columns=raw["categorical_columns"],
        date_columns=raw["date_columns"],
        boolean_columns=raw["boolean_columns"],
        missing_total=raw["missing_total"],
        duplicate_rows=raw["duplicate_rows"],
        date_ranges=raw["date_ranges"],
    )


@router.post("/datasets/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)) -> UploadResponse:
    content = await file.read()
    store = get_dataset_store()
    try:
        record = store.save(file.filename or "dataset", content)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    profile = _to_profile(record.id, record.original_filename, record.uploaded_at, record.df)
    logger.info("upload complete dataset_id=%s filename=%s", record.id, record.original_filename)
    return UploadResponse(dataset_id=record.id, profile=profile)


@router.get("/datasets/{dataset_id}", response_model=DatasetProfile)
def get_dataset(dataset_id: str) -> DatasetProfile:
    store = get_dataset_store()
    try:
        record = store.get(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _to_profile(record.id, record.original_filename, record.uploaded_at, record.df)
