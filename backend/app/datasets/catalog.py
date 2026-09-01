from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from app.datasets.ingestion import IngestionNotice

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DatasetMetadata:
    id: str
    original_filename: str
    extension: str
    uploaded_at: datetime
    byte_size: int
    sha256: str
    ingestion_notices: list[IngestionNotice]


class DatasetCatalog:
    """Small transactional SQLite catalog; raw dataset values never enter it."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    original_filename TEXT NOT NULL,
                    extension TEXT NOT NULL CHECK(extension IN ('.csv', '.xlsx')),
                    uploaded_at TEXT NOT NULL,
                    byte_size INTEGER NOT NULL CHECK(byte_size >= 0),
                    sha256 TEXT NOT NULL,
                    ingestion_notices TEXT NOT NULL DEFAULT '[]',
                    schema_version INTEGER NOT NULL
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_datasets_uploaded_at ON datasets(uploaded_at)"
            )

    def put(self, metadata: DatasetMetadata) -> None:
        notices = json.dumps(
            [asdict(item) for item in metadata.ingestion_notices],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    metadata.id,
                    metadata.original_filename,
                    metadata.extension,
                    metadata.uploaded_at.isoformat(),
                    metadata.byte_size,
                    metadata.sha256,
                    notices,
                    SCHEMA_VERSION,
                ),
            )

    def get(self, dataset_id: str) -> DatasetMetadata | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT original_filename, extension, uploaded_at, byte_size, sha256, ingestion_notices, schema_version FROM datasets WHERE id = ?",
                (dataset_id,),
            ).fetchone()
        if row is None:
            return None
        filename, extension, uploaded_at, byte_size, digest, notices, version = row
        if version != SCHEMA_VERSION:
            raise ValueError("Unsupported dataset metadata schema")
        decoded = json.loads(notices)
        if not isinstance(decoded, list):
            raise ValueError("Invalid ingestion notice metadata")
        return DatasetMetadata(
            id=dataset_id,
            original_filename=filename,
            extension=extension,
            uploaded_at=datetime.fromisoformat(uploaded_at),
            byte_size=byte_size,
            sha256=digest,
            ingestion_notices=[IngestionNotice(**item) for item in decoded],
        )

    def expired_ids(self, cutoff: datetime) -> set[str]:
        with self._connect() as connection:
            return {
                row[0]
                for row in connection.execute(
                    "SELECT id FROM datasets WHERE uploaded_at <= ?", (cutoff.isoformat(),)
                )
            }

    def delete_many(self, dataset_ids: set[str]) -> None:
        if not dataset_ids:
            return
        with self._connect() as connection:
            connection.executemany("DELETE FROM datasets WHERE id = ?", ((item,) for item in dataset_ids))

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0])

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
