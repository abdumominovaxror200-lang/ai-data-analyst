from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.tools.clustering import kmeans_cluster, pca_reduce
from app.tools.errors import ToolExecutionError

DEMO_DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "demo" / "sales_data.xlsx"


def _three_blobs_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    blob_a = rng.normal(loc=[0, 0], scale=0.5, size=(40, 2))
    blob_b = rng.normal(loc=[25, 25], scale=0.5, size=(40, 2))
    blob_c = rng.normal(loc=[50, 0], scale=0.5, size=(40, 2))
    data = np.vstack([blob_a, blob_b, blob_c])
    return pd.DataFrame(data, columns=["x", "y"])


def test_kmeans_cluster_recovers_three_obvious_blobs():
    df = _three_blobs_df()
    result = kmeans_cluster(df, columns=["x", "y"])

    assert result["n_clusters"] == 3
    assert result["auto_selected_k"] is True
    sizes = sorted(c["size"] for c in result["clusters"])
    assert sizes == [40, 40, 40]
    assert result["silhouette_score"] > 0.8


def test_kmeans_cluster_explicit_n_clusters_is_honored():
    df = _three_blobs_df()
    result = kmeans_cluster(df, columns=["x", "y"], n_clusters=3)
    assert result["n_clusters"] == 3
    assert result["auto_selected_k"] is False


def test_kmeans_cluster_centroids_in_original_units():
    df = _three_blobs_df()
    result = kmeans_cluster(df, columns=["x", "y"], n_clusters=3)
    centroids_x = sorted(c["centroid"]["x"] for c in result["clusters"])
    # blobs centered at x=0, x=25, x=50 -> recovered centroids should be close to those,
    # not standardized (roughly zero-mean/unit-variance) values.
    assert centroids_x[0] == pytest.approx(0, abs=1.5)
    assert centroids_x[1] == pytest.approx(25, abs=1.5)
    assert centroids_x[2] == pytest.approx(50, abs=1.5)


def test_kmeans_cluster_requires_two_columns():
    df = pd.DataFrame({"x": [1, 2, 3, 4]})
    with pytest.raises(ToolExecutionError):
        kmeans_cluster(df, columns=["x"])


def test_kmeans_cluster_rejects_zero_variance_column():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "y": [1, 1, 1, 1, 1, 1]})
    with pytest.raises(ToolExecutionError):
        kmeans_cluster(df, columns=["x", "y"])


def test_kmeans_cluster_rejects_non_numeric_column():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": ["a", "b", "c", "d"]})
    with pytest.raises(ToolExecutionError):
        kmeans_cluster(df, columns=["x", "y"])


def test_kmeans_cluster_rejects_too_few_rows_for_requested_k():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
    with pytest.raises(ToolExecutionError):
        kmeans_cluster(df, columns=["x", "y"], n_clusters=5)


def test_kmeans_cluster_applies_filters():
    df = _three_blobs_df()
    df["region"] = ["keep"] * 80 + ["drop"] * 40
    result = kmeans_cluster(
        df, columns=["x", "y"], n_clusters=2, filters=[{"column": "region", "op": "==", "value": "keep"}]
    )
    assert result["n_rows_used"] == 80


def test_pca_reduce_explained_variance_sums_to_one_for_full_rank():
    df = _three_blobs_df()
    result = pca_reduce(df, columns=["x", "y"], n_components=2)
    assert result["total_explained_variance_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert len(result["components"]) == 2
    assert result["components"][0]["explained_variance_ratio"] >= result["components"][1]["explained_variance_ratio"]


def test_pca_reduce_loadings_present_for_every_column():
    df = _three_blobs_df()
    result = pca_reduce(df, columns=["x", "y"], n_components=1)
    loadings = result["components"][0]["loadings"]
    assert set(loadings.keys()) == {"x", "y"}


def test_pca_reduce_rejects_n_components_over_max():
    df = _three_blobs_df()
    with pytest.raises(ToolExecutionError):
        pca_reduce(df, columns=["x", "y"], n_components=3)


def test_pca_reduce_rejects_single_column():
    df = pd.DataFrame({"x": [1, 2, 3, 4]})
    with pytest.raises(ToolExecutionError):
        pca_reduce(df, columns=["x"])


def test_kmeans_and_pca_reject_duplicate_columns():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [4, 3, 2, 1]})
    with pytest.raises(ToolExecutionError):
        kmeans_cluster(df, columns=["x", "x"])
    with pytest.raises(ToolExecutionError):
        pca_reduce(df, columns=["x", "x"])


# --- Real demo dataset --------------------------------------------------------


def test_kmeans_cluster_on_demo_sales_data():
    df = pd.read_excel(DEMO_DATASET_PATH)
    result = kmeans_cluster(df, columns=["quantity", "revenue", "profit"])
    assert result["n_clusters"] >= 2
    assert sum(c["size"] for c in result["clusters"]) == result["n_rows_used"]
    assert result["n_rows_used"] == len(df)
    assert -1.0 <= result["silhouette_score"] <= 1.0
