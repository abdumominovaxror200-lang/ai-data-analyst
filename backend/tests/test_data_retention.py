from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.datasets import storage as storage_module
from app.datasets.storage import DatasetNotFoundError, DatasetStore
from app.main import validate_deployment_policy


def _settings(tmp_path: Path, *, ttl: int = 60):
    return SimpleNamespace(
        max_upload_bytes=1024 * 1024,
        max_rows=10_000,
        storage_path=tmp_path,
        dataset_ttl_minutes=ttl,
    )


def test_expired_dataset_and_upload_file_are_removed(tmp_path, monkeypatch, caplog) -> None:
    secret_cell = "private-cell-value"
    monkeypatch.setattr(storage_module, "get_settings", lambda: _settings(tmp_path))
    store = DatasetStore()
    record = store.save("sample.csv", f"metric,note\n1,{secret_cell}\n".encode())
    upload = Path(record.stored_path)
    record.uploaded_at = datetime.now(timezone.utc) - timedelta(minutes=61)

    with caplog.at_level(logging.INFO):
        with pytest.raises(DatasetNotFoundError):
            store.get(record.id)

    assert not upload.exists()
    assert record.id not in {item.id for item in store.list()}
    assert secret_cell not in caplog.text


def test_fresh_dataset_survives_retention_sweep(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(storage_module, "get_settings", lambda: _settings(tmp_path))
    store = DatasetStore()
    record = store.save("fresh.csv", b"metric\n1\n")

    removed = store.sweep_expired(now=record.uploaded_at + timedelta(minutes=59))

    assert removed == 0
    assert store.get(record.id).id == record.id
    assert Path(record.stored_path).exists()


def test_public_local_only_configuration_fails_fast() -> None:
    with pytest.raises(RuntimeError, match="DEPLOYMENT=public"):
        validate_deployment_policy(SimpleNamespace(deployment="public", llm_egress_mode="local_only"))


def test_public_local_only_app_refuses_to_start(monkeypatch) -> None:
    monkeypatch.setattr(
        main_module,
        "settings",
        SimpleNamespace(deployment="public", llm_egress_mode="local_only"),
    )
    with pytest.raises(RuntimeError, match="DEPLOYMENT=public"):
        with TestClient(main_module.app):
            pass


def test_public_redacted_configuration_logs_privacy_notice(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="app.main"):
        validate_deployment_policy(
            SimpleNamespace(deployment="public", llm_egress_mode="external_redacted")
        )
    assert "raw-value LLM egress is disabled" in caplog.text
