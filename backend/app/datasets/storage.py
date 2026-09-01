from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from app.config import get_settings
from app.datasets.ingestion import IngestionNotice, parse_dataset
from app.datasets.metric_registry import MetricRegistry
from app.datasets.validation import (
    ValidationError,
    sanitize_display_name,
    validate_extension,
    validate_size,
)

if TYPE_CHECKING:
    from app.reasoning.contracts import AnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class DatasetRecord:
    id: str
    original_filename: str
    extension: str
    uploaded_at: datetime
    df: pd.DataFrame
    stored_path: str
    metrics: MetricRegistry | None = None
    ingestion_notices: list[IngestionNotice] | None = None
    latest_analysis: "AnalysisResult | None" = None

    def __post_init__(self) -> None:
        if self.metrics is None:
            self.metrics = MetricRegistry.from_dataframe(self.df)
        if self.ingestion_notices is None:
            self.ingestion_notices = []


class DatasetNotFoundError(Exception):
    pass


class DatasetStore:
    """In-memory, process-lifetime dataset store keyed by uuid.

    No database for the MVP (see reports/initial-audit.md) — datasets live for the
    lifetime of the backend process, addressable only by their generated id.
    """

    def __init__(self) -> None:
        self._records: dict[str, DatasetRecord] = {}
        self._lock = threading.Lock()

    def save(self, filename: str, content: bytes) -> DatasetRecord:
        self.sweep_expired()
        settings = get_settings()
        validate_size(len(content), settings.max_upload_bytes)
        ext = validate_extension(filename)
        safe_name = sanitize_display_name(filename)

        parsed = parse_dataset(content, ext)
        df = parsed.dataframe
        if len(df) > settings.max_rows:
            raise ValidationError(
                f"Dataset has {len(df):,} rows, which exceeds the {settings.max_rows:,} row limit."
            )
        if df.shape[1] == 0:
            raise ValidationError("Dataset has no columns.")

        dataset_id = uuid.uuid4().hex
        # Storage path is built entirely from our own generated id + validated
        # extension — the user-supplied filename never reaches the filesystem path,
        # which is what makes this safe against path traversal.
        stored_path = settings.storage_path / "uploads" / f"{dataset_id}{ext}"
        stored_path.parent.mkdir(parents=True, exist_ok=True)
        stored_path.write_bytes(content)

        record = DatasetRecord(
            id=dataset_id,
            original_filename=safe_name,
            extension=ext,
            uploaded_at=datetime.now(timezone.utc),
            df=df,
            stored_path=str(stored_path),
            ingestion_notices=parsed.notices,
        )
        with self._lock:
            self._records[dataset_id] = record
        logger.info("dataset stored id=%s rows=%s cols=%s", dataset_id, df.shape[0], df.shape[1])
        return record

    @staticmethod
    def _parse(content: bytes, ext: str) -> pd.DataFrame:
        """Backward-compatible parser entry point used by existing offline fixtures."""
        return parse_dataset(content, ext).dataframe

    def get(self, dataset_id: str) -> DatasetRecord:
        self.sweep_expired()
        with self._lock:
            record = self._records.get(dataset_id)
        if record is None:
            raise DatasetNotFoundError(f"Dataset '{dataset_id}' not found.")
        return record

    def list(self) -> list[DatasetRecord]:
        self.sweep_expired()
        with self._lock:
            return list(self._records.values())

    def sweep_expired(self, now: datetime | None = None) -> int:
        """Remove expired records and their server-owned upload files."""
        settings = get_settings()
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        cutoff = current - timedelta(minutes=settings.dataset_ttl_minutes)
        with self._lock:
            expired_ids = [
                dataset_id
                for dataset_id, record in self._records.items()
                if _as_utc(record.uploaded_at) <= cutoff
            ]
            expired = [self._records.pop(dataset_id) for dataset_id in expired_ids]
        for record in expired:
            _delete_upload_file(record)
            logger.info("dataset expired id=%s", record.id)
        if expired:
            logger.info("dataset retention sweep removed=%s", len(expired))
        return len(expired)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _delete_upload_file(record: DatasetRecord) -> None:
    settings = get_settings()
    uploads_root = (settings.storage_path / "uploads").resolve()
    candidate = Path(record.stored_path).resolve()
    if candidate.parent != uploads_root or candidate.stem != record.id:
        logger.warning("expired dataset file deletion skipped: path validation failed id=%s", record.id)
        return
    try:
        candidate.unlink(missing_ok=True)
    except OSError:
        logger.warning("expired dataset file deletion failed id=%s", record.id)


_store: DatasetStore | None = None


def get_dataset_store() -> DatasetStore:
    global _store
    if _store is None:
        _store = DatasetStore()
    return _store
