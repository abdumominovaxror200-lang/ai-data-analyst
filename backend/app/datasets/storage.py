from __future__ import annotations

import hashlib
import hmac
import logging
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from app.config import get_settings
from app.datasets.catalog import DatasetCatalog, DatasetMetadata
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


class DatasetUnavailableError(DatasetNotFoundError):
    """Persistent metadata exists but its upload cannot be restored safely."""


class DatasetStore:
    """In-memory dataframe cache backed by SQLite metadata and UUID-named files."""

    def __init__(self) -> None:
        self._records: dict[str, DatasetRecord] = {}
        self._lock = threading.RLock()
        settings = get_settings()
        self._storage_root = settings.storage_path.resolve()
        self._uploads_root = (self._storage_root / "uploads").resolve()
        self._uploads_root.mkdir(parents=True, exist_ok=True)
        self._persistence_enabled = bool(getattr(settings, "dataset_persistence_enabled", True))
        self._catalog = (
            DatasetCatalog(self._storage_root / "datasets.sqlite3")
            if self._persistence_enabled
            else None
        )

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
        stored_path = self._uploads_root / f"{dataset_id}{ext}"
        temporary_path = self._uploads_root / f".{dataset_id}.tmp"

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
            try:
                temporary_path.write_bytes(content)
                os.replace(temporary_path, stored_path)
                if self._catalog is not None:
                    self._catalog.put(
                        DatasetMetadata(
                            id=record.id,
                            original_filename=record.original_filename,
                            extension=record.extension,
                            uploaded_at=_as_utc(record.uploaded_at),
                            byte_size=len(content),
                            sha256=hashlib.sha256(content).hexdigest(),
                            ingestion_notices=list(record.ingestion_notices or []),
                        )
                    )
                self._records[dataset_id] = record
            except Exception:
                temporary_path.unlink(missing_ok=True)
                stored_path.unlink(missing_ok=True)
                raise
        logger.info(
            "dataset stored id=%s rows=%s cols=%s persistent=%s",
            dataset_id,
            df.shape[0],
            df.shape[1],
            self._persistence_enabled,
        )
        return record

    @staticmethod
    def _parse(content: bytes, ext: str) -> pd.DataFrame:
        """Backward-compatible parser entry point used by existing offline fixtures."""
        return parse_dataset(content, ext).dataframe

    def get(self, dataset_id: str) -> DatasetRecord:
        self.sweep_expired()
        with self._lock:
            record = self._records.get(dataset_id)
            if record is None and self._catalog is not None:
                record = self._restore(dataset_id)
                if record is not None:
                    self._records[dataset_id] = record
                    logger.info(
                        "dataset restored id=%s rows=%s cols=%s",
                        dataset_id,
                        record.df.shape[0],
                        record.df.shape[1],
                    )
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
            expired_ids = {
                dataset_id
                for dataset_id, record in self._records.items()
                if _as_utc(record.uploaded_at) <= cutoff
            }
            metadata_by_id: dict[str, DatasetMetadata] = {}
            if self._catalog is not None:
                expired_ids.update(self._catalog.expired_ids(cutoff))
                for dataset_id in expired_ids:
                    try:
                        metadata = self._catalog.get(dataset_id)
                    except (ValueError, TypeError):
                        metadata = None
                    if metadata is not None:
                        metadata_by_id[dataset_id] = metadata
                self._catalog.delete_many(expired_ids)
            expired_records = {
                dataset_id: self._records.pop(dataset_id, None) for dataset_id in expired_ids
            }
        for dataset_id in expired_ids:
            record = expired_records[dataset_id]
            metadata = metadata_by_id.get(dataset_id)
            extension = record.extension if record is not None else metadata.extension if metadata else None
            if extension:
                try:
                    _safe_upload_path(self._uploads_root, dataset_id, extension).unlink(missing_ok=True)
                except (OSError, DatasetUnavailableError):
                    logger.warning("expired dataset file deletion failed id=%s", dataset_id)
            logger.info("dataset expired id=%s", dataset_id)
        if expired_ids:
            logger.info("dataset retention sweep removed=%s", len(expired_ids))
        return len(expired_ids)

    def persisted_count(self) -> int:
        return self._catalog.count() if self._catalog is not None else 0

    def _restore(self, dataset_id: str) -> DatasetRecord | None:
        if not _valid_dataset_id(dataset_id):
            return None
        try:
            metadata = self._catalog.get(dataset_id) if self._catalog is not None else None
        except (ValueError, TypeError) as exc:
            logger.warning("persistent dataset metadata invalid id=%s", dataset_id)
            raise DatasetUnavailableError(
                f"Dataset '{dataset_id}' cannot be restored safely. Please upload it again."
            ) from exc
        if metadata is None:
            return None
        path = _safe_upload_path(self._uploads_root, dataset_id, metadata.extension)
        try:
            content = path.read_bytes()
        except OSError as exc:
            logger.warning("persistent dataset file unavailable id=%s", dataset_id)
            raise DatasetUnavailableError(f"Dataset '{dataset_id}' is no longer available. Please upload it again.") from exc
        if len(content) != metadata.byte_size or not hmac.compare_digest(
            hashlib.sha256(content).hexdigest(), metadata.sha256
        ):
            logger.warning("persistent dataset integrity check failed id=%s", dataset_id)
            raise DatasetUnavailableError(
                f"Dataset '{dataset_id}' failed an integrity check. Please upload it again."
            )
        try:
            parsed = parse_dataset(content, metadata.extension)
        except ValidationError as exc:
            logger.warning("persistent dataset parse failed id=%s", dataset_id)
            raise DatasetUnavailableError(
                f"Dataset '{dataset_id}' cannot be restored safely. Please upload it again."
            ) from exc
        if len(parsed.dataframe) > get_settings().max_rows:
            raise DatasetUnavailableError(f"Dataset '{dataset_id}' exceeds the current row limit. Please upload a smaller file.")
        return DatasetRecord(
            id=dataset_id,
            original_filename=metadata.original_filename,
            extension=metadata.extension,
            uploaded_at=metadata.uploaded_at,
            df=parsed.dataframe,
            stored_path=str(path),
            ingestion_notices=metadata.ingestion_notices or parsed.notices,
        )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _valid_dataset_id(dataset_id: str) -> bool:
    try:
        return uuid.UUID(hex=dataset_id).hex == dataset_id
    except (ValueError, AttributeError):
        return False


def _safe_upload_path(uploads_root: Path, dataset_id: str, extension: str) -> Path:
    if not _valid_dataset_id(dataset_id) or extension not in {".csv", ".xlsx"}:
        raise DatasetUnavailableError(f"Dataset '{dataset_id}' cannot be restored safely.")
    candidate = (uploads_root / f"{dataset_id}{extension}").resolve()
    if candidate.parent != uploads_root or candidate.stem != dataset_id:
        raise DatasetUnavailableError(f"Dataset '{dataset_id}' cannot be restored safely.")
    return candidate


_store: DatasetStore | None = None


def get_dataset_store() -> DatasetStore:
    global _store
    if _store is None:
        _store = DatasetStore()
    return _store
