from __future__ import annotations

from tests.conftest import make_csv_bytes


def test_analysis_unknown_dataset_returns_404(client):
    response = client.post("/api/analysis", json={"dataset_id": "missing", "tool": "profile_dataset", "params": {}})
    assert response.status_code == 404


def test_analysis_unknown_tool_returns_400(client, sample_df):
    files = {"file": ("sales.csv", make_csv_bytes(sample_df), "text/csv")}
    dataset_id = client.post("/api/datasets/upload", files=files).json()["dataset_id"]

    response = client.post("/api/analysis", json={"dataset_id": dataset_id, "tool": "delete_everything", "params": {}})
    assert response.status_code == 400


def test_analysis_bad_params_returns_400(client, sample_df):
    files = {"file": ("sales.csv", make_csv_bytes(sample_df), "text/csv")}
    dataset_id = client.post("/api/datasets/upload", files=files).json()["dataset_id"]

    response = client.post(
        "/api/analysis",
        json={"dataset_id": dataset_id, "tool": "group_and_aggregate", "params": {"group_by": "nope", "agg_column": "revenue"}},
    )
    assert response.status_code == 400


def test_analysis_missing_required_field_returns_422(client):
    response = client.post("/api/analysis", json={"tool": "profile_dataset"})
    assert response.status_code == 422


def test_analysis_profile_dataset_end_to_end(client, sample_df):
    files = {"file": ("sales.csv", make_csv_bytes(sample_df), "text/csv")}
    dataset_id = client.post("/api/datasets/upload", files=files).json()["dataset_id"]

    response = client.post("/api/analysis", json={"dataset_id": dataset_id, "tool": "profile_dataset", "params": {}})
    assert response.status_code == 200
    body = response.json()
    assert body["result"]["rows"] == len(sample_df)


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_without_llm_key_returns_503(client, sample_df, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("LLM_API_KEY", "")
    get_settings.cache_clear()
    try:
        files = {"file": ("sales.csv", make_csv_bytes(sample_df), "text/csv")}
        dataset_id = client.post("/api/datasets/upload", files=files).json()["dataset_id"]

        response = client.post("/api/chat", json={"dataset_id": dataset_id, "message": "Analyze this dataset."})
        assert response.status_code == 503
    finally:
        get_settings.cache_clear()
