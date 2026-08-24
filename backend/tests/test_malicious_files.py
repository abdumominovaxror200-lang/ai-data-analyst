from __future__ import annotations

from pathlib import Path

from app.datasets.storage import get_dataset_store
from app.datasets.validation import sanitize_display_name


def test_sanitize_display_name_strips_path_traversal():
    assert sanitize_display_name("../../etc/passwd.csv") == "passwd.csv"
    assert sanitize_display_name("..\\..\\windows\\system32\\evil.csv") == "evil.csv"
    assert "/" not in sanitize_display_name("a/b/c.csv")
    assert "\\" not in sanitize_display_name("a\\b\\c.csv")


def test_uploaded_file_is_never_stored_at_a_user_controlled_path(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path))
    from app.config import get_settings

    get_settings.cache_clear()
    import app.datasets.storage as storage_module

    storage_module._store = None
    store = get_dataset_store()

    record = store.save("../../evil.csv", b"a,b\n1,2\n")
    stored_path = Path(record.stored_path).resolve()

    # The stored path must live inside the configured storage dir — not escape it —
    # regardless of what path-traversal payload was in the original filename.
    assert tmp_path.resolve() in stored_path.parents
    assert ".." not in stored_path.parts

    storage_module._store = None
    get_settings.cache_clear()


def test_executable_disguised_as_csv_is_rejected_or_parsed_safely(client):
    payload = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff" * 100
    files = {"file": ("payload.csv", payload, "text/csv")}
    response = client.post("/api/datasets/upload", files=files)
    # Whatever pandas makes of binary garbage, the server must respond cleanly —
    # never crash, never execute the payload.
    assert response.status_code in (200, 400)


def test_xlsx_with_wrong_extension_content_is_rejected(client):
    files = {"file": ("fake.xlsx", b"this is not a real zip/xlsx file", "application/octet-stream")}
    response = client.post("/api/datasets/upload", files=files)
    assert response.status_code == 400


def test_filename_with_null_byte_is_sanitized():
    # Defensive: some clients can smuggle a null byte in a filename.
    name = sanitize_display_name("evil.csv\x00.exe")
    assert "\x00" not in name
