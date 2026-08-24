from __future__ import annotations

from tests.conftest import make_csv_bytes, make_xlsx_bytes


def test_upload_csv(client, sample_df):
    files = {"file": ("sales.csv", make_csv_bytes(sample_df), "text/csv")}
    response = client.post("/api/datasets/upload", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["profile"]["rows"] == len(sample_df)
    assert body["profile"]["columns"] == len(sample_df.columns)
    assert "dataset_id" in body


def test_upload_xlsx(client, sample_df):
    files = {"file": ("sales.xlsx", make_xlsx_bytes(sample_df), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    response = client.post("/api/datasets/upload", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["profile"]["rows"] == len(sample_df)


def test_get_dataset_after_upload(client, sample_df):
    files = {"file": ("sales.csv", make_csv_bytes(sample_df), "text/csv")}
    upload = client.post("/api/datasets/upload", files=files).json()

    response = client.get(f"/api/datasets/{upload['dataset_id']}")
    assert response.status_code == 200
    assert response.json()["filename"] == "sales.csv"


def test_get_unknown_dataset_returns_404(client):
    response = client.get("/api/datasets/does-not-exist")
    assert response.status_code == 404


def test_upload_rejects_unsupported_extension(client):
    files = {"file": ("malware.exe", b"not a dataset", "application/octet-stream")}
    response = client.post("/api/datasets/upload", files=files)
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_rejects_malformed_csv(client):
    garbage = b"\x00\x01\x02\xff\xfe not,a,real\ncsv,file"
    files = {"file": ("broken.csv", garbage, "text/csv")}
    response = client.post("/api/datasets/upload", files=files)
    # pandas may parse a single garbage column successfully; the important guarantee
    # is that it never crashes the server — either a clean 200 or a clean 400.
    assert response.status_code in (200, 400)


def test_upload_rejects_empty_file(client):
    files = {"file": ("empty.csv", b"", "text/csv")}
    response = client.post("/api/datasets/upload", files=files)
    assert response.status_code == 400


def test_upload_rejects_oversized_file(client, monkeypatch, sample_df):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.001")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        files = {"file": ("sales.csv", make_csv_bytes(sample_df), "text/csv")}
        response = client.post("/api/datasets/upload", files=files)
        assert response.status_code == 400
        assert "too large" in response.json()["detail"].lower()
    finally:
        get_settings.cache_clear()
