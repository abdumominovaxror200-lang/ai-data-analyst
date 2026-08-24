from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings


@pytest.fixture(autouse=True)
def _isolated_dataset_store(tmp_path, monkeypatch):
    """Fresh dataset store + storage dir per test, so tests never see each other's data."""
    import app.datasets.storage as storage_module

    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))
    get_settings.cache_clear()
    storage_module._store = None
    yield
    storage_module._store = None
    get_settings.cache_clear()


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    return TestClient(app)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 100
    revenue = rng.normal(1000, 100, n)
    revenue[0] = 50_000  # intentional outlier
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "region": rng.choice(["North", "South", "East", "West"], n),
            "product": rng.choice(["Widget", "Gadget", "Gizmo"], n),
            "quantity": rng.integers(1, 50, n),
            "revenue": revenue,
            "cost": revenue * rng.uniform(0.4, 0.6, n),
        }
    )


def make_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def make_xlsx_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    return buffer.getvalue()
