from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from app.tools.errors import ToolExecutionError
from app.tools.filtering import apply_filters

_MIN_ROWS_PER_CLUSTER = 2
_MAX_AUTO_K = 8
_RANDOM_STATE = 42


def _prepare_numeric_subset(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns or len(columns) < 2:
        raise ToolExecutionError("At least two columns are required.")
    if len(set(columns)) != len(columns):
        raise ToolExecutionError("columns contains duplicates.")

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ToolExecutionError(f"Unknown column(s): {', '.join(missing)}")

    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ToolExecutionError(f"Column(s) must be numeric: {', '.join(non_numeric)}")

    subset = df[columns].dropna()
    if subset.empty:
        raise ToolExecutionError("No complete rows (no missing values) available across the given columns.")

    zero_variance = [c for c in columns if subset[c].std() == 0]
    if zero_variance:
        raise ToolExecutionError(
            f"Column(s) have zero variance and cannot be standardized: {', '.join(zero_variance)}"
        )
    return subset


def kmeans_cluster(
    df: pd.DataFrame,
    columns: list[str],
    n_clusters: int | None = None,
    filters: list[dict] | None = None,
) -> dict:
    """K-means clustering over `columns`.

    Why standardize first: K-means groups points by Euclidean distance, so a
    column with a naturally large numeric range (e.g. revenue in the
    thousands) would dominate the distance calculation over a column with a
    small range (e.g. a 1-5 rating), regardless of which one actually
    separates the groups better. `StandardScaler` (zero mean, unit variance)
    puts every feature on equal footing before clustering; centroids are
    reported back in the original (unscaled) units for interpretability.

    Why auto-k via silhouette score: if `n_clusters` isn't given, this fits
    k = 2..8 (capped by how many rows can support that many clusters) and
    picks the k with the highest silhouette score — a single number in
    [-1, 1] that rewards points being close to their own cluster's center and
    far from other clusters' centers, balancing tightness against
    separation. It's a standard, defensible default for "pick a reasonable
    k without a human eyeballing an elbow plot," not a claim that the data
    has a single objectively "true" number of clusters — a dataset with no
    real cluster structure will still get a k back, just with a low score.

    Validates: at least 2 numeric columns, at least `n_clusters * 2` complete
    rows (fewer than 2 rows per cluster makes a centroid meaningless), and no
    zero-variance column (would divide by zero when standardizing).
    """
    working = apply_filters(df, filters) if filters else df
    subset = _prepare_numeric_subset(working, columns)
    n_rows = len(subset)
    auto_selected = n_clusters is None

    if n_clusters is not None:
        if n_clusters < 2:
            raise ToolExecutionError("n_clusters must be at least 2.")
        if n_rows < n_clusters * _MIN_ROWS_PER_CLUSTER:
            raise ToolExecutionError(
                f"At least {n_clusters * _MIN_ROWS_PER_CLUSTER} complete rows are required for "
                f"{n_clusters} clusters (found {n_rows})."
            )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(subset.to_numpy(dtype=float))

    if auto_selected:
        max_k = min(_MAX_AUTO_K, n_rows // _MIN_ROWS_PER_CLUSTER)
        if max_k < 2:
            raise ToolExecutionError(
                f"Not enough complete rows ({n_rows}) to cluster automatically; need at least "
                f"{2 * _MIN_ROWS_PER_CLUSTER} for the minimum of 2 clusters."
            )
        best_k, best_score, best_model = None, -2.0, None
        for k in range(2, max_k + 1):
            model = KMeans(n_clusters=k, random_state=_RANDOM_STATE, n_init=10).fit(scaled)
            score = float(silhouette_score(scaled, model.labels_))
            if score > best_score:
                best_k, best_score, best_model = k, score, model
        n_clusters = best_k
        silhouette = best_score
        final_model = best_model
    else:
        final_model = KMeans(n_clusters=n_clusters, random_state=_RANDOM_STATE, n_init=10).fit(scaled)
        silhouette = float(silhouette_score(scaled, final_model.labels_))

    labels = final_model.labels_
    centroids_original = scaler.inverse_transform(final_model.cluster_centers_)

    clusters = []
    for cluster_id in range(n_clusters):
        size = int((labels == cluster_id).sum())
        centroid = {col: round(float(val), 4) for col, val in zip(columns, centroids_original[cluster_id])}
        clusters.append(
            {
                "cluster_id": int(cluster_id),
                "size": size,
                "pct_of_rows": round(size / n_rows * 100, 2),
                "centroid": centroid,
            }
        )
    clusters.sort(key=lambda c: c["cluster_id"])

    return {
        "columns": columns,
        "n_clusters": int(n_clusters),
        "n_rows_used": int(n_rows),
        "silhouette_score": round(silhouette, 4),
        "auto_selected_k": auto_selected,
        "clusters": clusters,
    }


def pca_reduce(
    df: pd.DataFrame,
    columns: list[str],
    n_components: int = 2,
    filters: list[dict] | None = None,
) -> dict:
    """PCA dimensionality reduction over `columns`, for visualization prep or as
    an input to downstream clustering.

    Standardized first for the same reason as `kmeans_cluster`: PCA finds the
    directions of maximum variance in the data, so an unscaled large-range
    column would mechanically dominate the first component regardless of
    whether it's actually the most informative variable.

    Returns, per component, its explained-variance ratio *and* its loadings
    (the weight/direction of each original column within that component).
    The loadings are what make the output interpretable rather than a set of
    abstract transformed coordinates — e.g. "PC1 is mostly revenue and
    profit moving together" is something a caller (or the narrating LLM) can
    say something useful about; a bare list of transformed values is not.
    """
    working = apply_filters(df, filters) if filters else df
    subset = _prepare_numeric_subset(working, columns)

    if n_components < 1:
        raise ToolExecutionError("n_components must be at least 1.")
    max_components = min(len(columns), len(subset))
    if n_components > max_components:
        raise ToolExecutionError(
            f"n_components ({n_components}) cannot exceed min(number of columns, number of complete rows) "
            f"= {max_components}."
        )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(subset.to_numpy(dtype=float))

    pca = PCA(n_components=n_components, random_state=_RANDOM_STATE)
    pca.fit(scaled)

    components = []
    for i in range(n_components):
        loadings = {col: round(float(val), 4) for col, val in zip(columns, pca.components_[i])}
        components.append(
            {
                "component": f"PC{i + 1}",
                "explained_variance_ratio": round(float(pca.explained_variance_ratio_[i]), 4),
                "loadings": loadings,
            }
        )

    return {
        "columns": columns,
        "n_components": int(n_components),
        "n_rows_used": int(len(subset)),
        "total_explained_variance_ratio": round(float(sum(pca.explained_variance_ratio_)), 4),
        "components": components,
    }
