from __future__ import annotations

import pandas as pd

from app.datasets.bundle import DatasetBundle, join_bundle
from app.datasets.storage import get_dataset_store
from app.datasets.validation import ValidationError


def _upload(client, name: str, frame: pd.DataFrame) -> str:
    response = client.post(
        "/api/datasets/upload",
        files={"file": (name, frame.to_csv(index=False).encode("utf-8"), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response.json()["dataset_id"]


def test_bundle_join_creates_analyzable_dataset_with_diagnostics(client):
    import app.datasets.storage as storage_module

    customers = _upload(
        client,
        "customers.csv",
        pd.DataFrame({"customer_id": [1, 2, 3], "region": ["N", "S", "W"]}),
    )
    orders = _upload(
        client,
        "orders.csv",
        pd.DataFrame({"order_id": [10, 11, 12], "customer_id": [1, 1, 4], "amount": [5, 7, 9]}),
    )

    response = client.post(
        "/api/datasets/bundles/join",
        json={
            "name": "customer_orders.csv",
            "members": [
                {"dataset_id": customers, "alias": "customers"},
                {"dataset_id": orders, "alias": "orders"},
            ],
            "joins": [
                {
                    "left_alias": "customers",
                    "right_alias": "orders",
                    "join_type": "left",
                    "keys": [{"left": "customer_id", "right": "customer_id"}],
                }
            ],
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["profile"]["rows"] == 4
    assert body["profile"]["filename"] == "customer_orders.csv"
    assert {column["name"] for column in body["profile"]["column_info"]} == {
        "customers__customer_id",
        "customers__region",
        "orders__order_id",
        "orders__customer_id",
        "orders__amount",
    }
    assert body["joins"] == [
        {
            "left_alias": "customers",
            "right_alias": "orders",
            "join_type": "left",
            "cardinality": "one_to_many",
            "left_rows": 3,
            "right_rows": 3,
            "left_null_key_rows": 0,
            "right_null_key_rows": 0,
            "left_duplicate_key_rows": 0,
            "right_duplicate_key_rows": 2,
            "unmatched_left_rows": 2,
            "unmatched_right_rows": 1,
        }
    ]
    storage_module._store = None
    restored = client.get(f'/api/datasets/{body["dataset_id"]}')
    assert restored.status_code == 200
    assert restored.json()["rows"] == 4


def test_many_to_many_join_fails_closed_without_explicit_opt_in(client):
    left = _upload(client, "left.csv", pd.DataFrame({"key": [1, 1], "value": ["a", "b"]}))
    right = _upload(client, "right.csv", pd.DataFrame({"key": [1, 1], "score": [3, 4]}))
    payload = {
        "members": [
            {"dataset_id": left, "alias": "left_table"},
            {"dataset_id": right, "alias": "right_table"},
        ],
        "joins": [
            {
                "left_alias": "left_table",
                "right_alias": "right_table",
                "keys": [{"left": "key", "right": "key"}],
            }
        ],
    }
    response = client.post("/api/datasets/bundles/join", json=payload)
    assert response.status_code == 400
    assert "many-to-many" in response.json()["detail"]

    payload["joins"][0]["allow_many_to_many"] = True
    allowed = client.post("/api/datasets/bundles/join", json=payload)
    assert allowed.status_code == 200
    assert allowed.json()["output_rows"] == 4
    assert allowed.json()["joins"][0]["cardinality"] == "many_to_many"


def test_bundle_validates_connected_order_and_join_columns(client):
    first = _upload(client, "first.csv", pd.DataFrame({"id": [1]}))
    second = _upload(client, "second.csv", pd.DataFrame({"id": [1]}))
    response = client.post(
        "/api/datasets/bundles/join",
        json={
            "members": [
                {"dataset_id": first, "alias": "first"},
                {"dataset_id": second, "alias": "second"},
            ],
            "joins": [
                {
                    "left_alias": "first",
                    "right_alias": "second",
                    "keys": [{"left": "missing", "right": "id"}],
                }
            ],
        },
    )
    assert response.status_code == 400
    assert "Column 'missing'" in response.json()["detail"]


def test_bundle_reports_null_keys_and_composite_key_unmatched_rows(client):
    left = _upload(
        client,
        "left.csv",
        pd.DataFrame({"account": [1, 1, 2, None], "month": [1, 2, 1, 1]}),
    )
    right = _upload(
        client,
        "right.csv",
        pd.DataFrame({"acct": [1, 2, 3, None], "period": [1, 2, 1, 1]}),
    )
    response = client.post(
        "/api/datasets/bundles/join",
        json={
            "members": [
                {"dataset_id": left, "alias": "activity"},
                {"dataset_id": right, "alias": "targets"},
            ],
            "joins": [
                {
                    "left_alias": "activity",
                    "right_alias": "targets",
                    "join_type": "full",
                    "keys": [
                        {"left": "account", "right": "acct"},
                        {"left": "month", "right": "period"},
                    ],
                }
            ],
        },
    )
    assert response.status_code == 200, response.text
    report = response.json()["joins"][0]
    assert report["left_null_key_rows"] == 1
    assert report["right_null_key_rows"] == 1
    assert report["unmatched_left_rows"] == 2
    assert report["unmatched_right_rows"] == 2
    assert response.json()["output_rows"] == 7


def test_bundle_rejects_missing_dataset_without_creating_output(client):
    real = _upload(client, "real.csv", pd.DataFrame({"id": [1]}))
    response = client.post(
        "/api/datasets/bundles/join",
        json={
            "members": [
                {"dataset_id": real, "alias": "real"},
                {"dataset_id": "0" * 32, "alias": "missing"},
            ],
            "joins": [
                {
                    "left_alias": "real",
                    "right_alias": "missing",
                    "keys": [{"left": "id", "right": "id"}],
                }
            ],
        },
    )
    assert response.status_code == 404


def test_join_row_limit_is_checked_before_output_dataset_is_saved(client):
    left = _upload(client, "left.csv", pd.DataFrame({"key": [1, 2, 3]}))
    right = _upload(client, "right.csv", pd.DataFrame({"key": [1, 2, 3]}))
    bundle = DatasetBundle.model_validate(
        {
            "members": [
                {"dataset_id": left, "alias": "left_table"},
                {"dataset_id": right, "alias": "right_table"},
            ],
            "joins": [
                {
                    "left_alias": "left_table",
                    "right_alias": "right_table",
                    "keys": [{"left": "key", "right": "key"}],
                }
            ],
        }
    )
    store = get_dataset_store()
    before = len(store.list())
    try:
        join_bundle(bundle, store, max_output_rows=2)
    except ValidationError as exc:
        assert "safe 2-row limit" in str(exc)
    else:
        raise AssertionError("Expected an oversized join to fail")
    assert len(store.list()) == before
