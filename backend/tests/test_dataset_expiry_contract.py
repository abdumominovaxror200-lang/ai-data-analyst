from __future__ import annotations

import pytest


MISSING_DATASET_ID = "expired-dataset-id"
EXPECTED_DETAIL = f"Dataset '{MISSING_DATASET_ID}' not found."


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("get", f"/api/datasets/{MISSING_DATASET_ID}", None),
        (
            "post",
            "/api/analysis",
            {"dataset_id": MISSING_DATASET_ID, "tool": "profile_dataset", "params": {}},
        ),
        (
            "post",
            "/api/chat",
            {"dataset_id": MISSING_DATASET_ID, "message": "hello", "history": []},
        ),
        ("post", "/api/reports", {"dataset_id": MISSING_DATASET_ID}),
        ("get", f"/api/reports/{MISSING_DATASET_ID}/excel", None),
        ("get", f"/api/reports/{MISSING_DATASET_ID}/word", None),
        ("get", f"/api/reports/{MISSING_DATASET_ID}/powerpoint", None),
        ("post", "/api/reason", {"dataset_id": MISSING_DATASET_ID, "message": "summarize"}),
    ],
)
def test_all_dataset_frontend_flows_share_expired_dataset_contract(client, method, path, payload):
    response = getattr(client, method)(path, json=payload) if payload is not None else client.get(path)

    assert response.status_code == 404
    assert response.json() == {"detail": EXPECTED_DETAIL}
