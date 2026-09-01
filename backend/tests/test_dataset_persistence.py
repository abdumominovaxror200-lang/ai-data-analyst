from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from app.datasets import storage as storage_module
from app.datasets.storage import DatasetNotFoundError, DatasetStore, DatasetUnavailableError


def _settings(root: Path, *, enabled: bool = True, ttl: int = 240):
    return SimpleNamespace(
        max_upload_bytes=2 * 1024 * 1024,
        max_rows=10_000,
        storage_path=root,
        dataset_ttl_minutes=ttl,
        dataset_persistence_enabled=enabled,
    )


def test_dataset_is_restored_lazily_after_store_restart(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(storage_module, "get_settings", lambda: _settings(tmp_path))
    first = DatasetStore()
    saved = first.save("sales.csv", b"date,revenue\n2025-01-01,10\n2025-01-02,20\n")

    restarted = DatasetStore()
    assert restarted._records == {}
    restored = restarted.get(saved.id)

    assert restored.id == saved.id
    assert restored.original_filename == "sales.csv"
    pd.testing.assert_frame_equal(restored.df, saved.df)
    assert restarted.persisted_count() == 1


def test_ingestion_notices_survive_restart(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(storage_module, "get_settings", lambda: _settings(tmp_path))
    saved = DatasetStore().save("legacy.csv", "city,value\nMontréal,10\n".encode("cp1252"))

    restored = DatasetStore().get(saved.id)

    assert [notice.code for notice in restored.ingestion_notices] == ["encoding_detected"]
    assert restored.df.iloc[0]["city"] == "Montréal"


def test_persisted_dataset_remains_available_through_real_api(tmp_path, monkeypatch, client) -> None:
    monkeypatch.setattr(storage_module, "get_settings", lambda: _settings(tmp_path))
    first = DatasetStore()
    saved = first.save("sales.csv", b"region,revenue\nNorth,10\nSouth,20\n")
    storage_module._store = DatasetStore()

    response = client.get(f"/api/datasets/{saved.id}")

    assert response.status_code == 200
    assert response.json()["dataset_id"] == saved.id
    assert response.json()["rows"] == 2
    rerun = client.post(
        "/api/analysis",
        json={"dataset_id": saved.id, "tool": "profile_dataset", "params": {}},
    )
    assert rerun.status_code == 200
    assert rerun.json()["result"]["rows"] == 2


def test_tampered_persisted_file_fails_closed_without_raw_values(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setattr(storage_module, "get_settings", lambda: _settings(tmp_path))
    first = DatasetStore()
    saved = first.save("private.csv", b"email,value\nalice@example.test,10\n")
    Path(saved.stored_path).write_bytes(b"email,value\nattacker@example.test,99\n")

    with pytest.raises(DatasetUnavailableError, match="integrity check"):
        DatasetStore().get(saved.id)
    assert "alice@example.test" not in caplog.text
    assert "attacker@example.test" not in caplog.text


def test_missing_persisted_file_returns_typed_unavailable_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(storage_module, "get_settings", lambda: _settings(tmp_path))
    saved = DatasetStore().save("sales.csv", b"value\n1\n")
    Path(saved.stored_path).unlink()

    with pytest.raises(DatasetUnavailableError, match="upload it again"):
        DatasetStore().get(saved.id)


def test_expired_persisted_metadata_and_file_are_removed_after_restart(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(storage_module, "get_settings", lambda: _settings(tmp_path, ttl=60))
    saved = DatasetStore().save("sales.csv", b"value\n1\n")
    with sqlite3.connect(tmp_path / "datasets.sqlite3") as connection:
        connection.execute("UPDATE datasets SET uploaded_at = '2000-01-01T00:00:00+00:00' WHERE id = ?", (saved.id,))

    restarted = DatasetStore()
    assert restarted.sweep_expired() == 1
    assert not Path(saved.stored_path).exists()
    assert restarted.persisted_count() == 0
    with pytest.raises(DatasetNotFoundError):
        restarted.get(saved.id)


def test_catalog_failure_rolls_back_server_owned_upload(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(storage_module, "get_settings", lambda: _settings(tmp_path))
    store = DatasetStore()
    monkeypatch.setattr(store._catalog, "put", lambda _metadata: (_ for _ in ()).throw(sqlite3.OperationalError("disk full")))

    with pytest.raises(sqlite3.OperationalError):
        store.save("sales.csv", b"value\n1\n")
    assert list((tmp_path / "uploads").iterdir()) == []
    assert store.persisted_count() == 0


def test_persistence_can_be_explicitly_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(storage_module, "get_settings", lambda: _settings(tmp_path, enabled=False))
    saved = DatasetStore().save("sales.csv", b"value\n1\n")

    assert not (tmp_path / "datasets.sqlite3").exists()
    with pytest.raises(DatasetNotFoundError):
        DatasetStore().get(saved.id)


def test_invalid_identifier_never_resolves_a_path_or_catalog_row(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(storage_module, "get_settings", lambda: _settings(tmp_path))
    store = DatasetStore()
    with pytest.raises(DatasetNotFoundError):
        store.get("../../another-session")


def test_catalog_connections_do_not_hold_database_open(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(storage_module, "get_settings", lambda: _settings(tmp_path))
    store = DatasetStore()
    store.save("sales.csv", b"value\n1\n")

    renamed = tmp_path / "catalog-renamed.sqlite3"
    (tmp_path / "datasets.sqlite3").replace(renamed)
    assert renamed.exists()
