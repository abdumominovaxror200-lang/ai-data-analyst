from __future__ import annotations

import io
import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from app.config import get_settings
from app.datasets.validation import (
    ValidationError,
    sanitize_display_name,
    validate_extension,
    validate_size,
)
from app.datasets.metric_registry import MetricRegistry

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

    def __post_init__(self) -> None:
        if self.metrics is None:
            self.metrics = MetricRegistry.from_dataframe(self.df)


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

        df = self._parse(content, ext)
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
        )
        with self._lock:
            self._records[dataset_id] = record
        logger.info("dataset stored id=%s rows=%s cols=%s", dataset_id, df.shape[0], df.shape[1])
        return record

    @staticmethod
    def _parse(content: bytes, ext: str) -> pd.DataFrame:
        try:
            if ext == ".csv":
                df = pd.read_csv(io.BytesIO(content))
            else:
                df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a clean 400
            raise ValidationError(f"Could not parse file: {exc}") from exc

        return _infer_datetime_columns(df)

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


def _infer_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(20)
            if sample.empty:
                continue
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.9:
                df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
    return df


_store: DatasetStore | None = None


def get_dataset_store() -> DatasetStore:
    global _store
    if _store is None:
        _store = DatasetStore()
    return _store
