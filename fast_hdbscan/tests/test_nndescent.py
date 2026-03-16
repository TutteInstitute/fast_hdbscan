"""
Tests for pynndescent-based approximate nearest neighbor support.

These tests verify that arbitrary metrics (e.g. cosine, manhattan) can be
used via the pynndescent backend, producing results comparable to the
euclidean KD-tree path where applicable.

Requires pynndescent to be installed.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import adjusted_rand_score

# Skip entire module if pynndescent is not installed
pynndescent = pytest.importorskip("pynndescent")

from fast_hdbscan import HDBSCAN, fast_hdbscan
from fast_hdbscan.nndescent import (
    _check_pynndescent_available,
    build_knn_graph,
    try_connect_knn_graph,
    _knn_graph_to_sparse,
    compute_mst_from_knn_graph,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_CLUSTERS = 3
N_SAMPLES = 200
RANDOM_STATE = 42


@pytest.fixture(scope="module")
def blob_data():
    """Well-separated blobs for smoke tests."""
    X, y = make_blobs(
        n_samples=N_SAMPLES,
        centers=N_CLUSTERS,
        random_state=RANDOM_STATE,
        cluster_std=0.5,
    )
    X = StandardScaler().fit_transform(X)
    return X, y


@pytest.fixture(scope="module")
def cosine_data(blob_data):
    """Blobs normalised to the unit sphere (good for cosine metric)."""
    X, y = blob_data
    return normalize(X, norm="l2"), y


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestCheckPyNNDescentAvailable:
    def test_returns_true(self):
        assert _check_pynndescent_available() is True


class TestBuildKnnGraph:
    def test_basic_shapes(self, blob_data):
        X, _ = blob_data
        k = 15
        index, knn_indices, knn_distances = build_knn_graph(
            X,
            n_neighbors=k,
            metric="euclidean",
            random_state=RANDOM_STATE,
        )
        assert knn_indices.shape == (X.shape[0], k)
        assert knn_distances.shape == (X.shape[0], k)
        assert knn_indices.dtype == np.int32
        assert knn_distances.dtype == np.float64

    def test_no_self_loops(self, blob_data):
        X, _ = blob_data
        _, knn_indices, _ = build_knn_graph(
            X,
            n_neighbors=10,
            metric="euclidean",
            random_state=RANDOM_STATE,
        )
        self_loops = np.any(knn_indices == np.arange(X.shape[0])[:, None])
        assert not self_loops, "KNN indices should not contain self-loops"

    def test_distances_nonnegative(self, blob_data):
        X, _ = blob_data
        _, _, knn_distances = build_knn_graph(
            X,
            n_neighbors=10,
            metric="euclidean",
            random_state=RANDOM_STATE,
        )
        assert np.all(knn_distances >= 0)

    def test_cosine_metric(self, cosine_data):
        X, _ = cosine_data
        index, knn_indices, knn_distances = build_knn_graph(
            X,
            n_neighbors=10,
            metric="cosine",
            random_state=RANDOM_STATE,
        )
        assert knn_indices.shape[1] == 10
        # Cosine distances should be in [0, 2]
        assert np.all(knn_distances >= -1e-6)
        assert np.all(knn_distances <= 2.0 + 1e-6)


class TestTryConnectKnnGraph:
    # NOTE: try_connect_knn_graph tests are skipped for now because
    # pynndescent's connect_graph can hang on well-separated clusters.
    # Re-enable once the upstream issue is debugged.

    @pytest.mark.skip(
        reason="pynndescent connect_graph can hang — disabled pending upstream fix"
    )
    def test_well_connected_graph(self, blob_data):
        """For overlapping blobs, the KNN graph should be connected already."""
        # Use larger cluster_std for overlap so KNN graph is connected
        X, _ = make_blobs(
            n_samples=N_SAMPLES,
            centers=N_CLUSTERS,
            random_state=RANDOM_STATE,
            cluster_std=2.0,
        )
        X = StandardScaler().fit_transform(X)
        index, knn_indices, knn_distances = build_knn_graph(
            X,
            n_neighbors=15,
            metric="euclidean",
            random_state=RANDOM_STATE,
        )
        connected, adj = try_connect_knn_graph(knn_indices, knn_distances, index)
        assert connected is True
        assert adj is not None
        assert adj.shape == (X.shape[0], X.shape[0])

    @pytest.mark.skip(
        reason="pynndescent connect_graph can hang — disabled pending upstream fix"
    )
    def test_disconnected_graph_returns_gracefully(self, blob_data):
        """For well-separated blobs, try_connect may fail — should return cleanly."""
        X, _ = blob_data  # cluster_std=0.5, blobs are well-separated
        index, knn_indices, knn_distances = build_knn_graph(
            X,
            n_neighbors=10,
            metric="euclidean",
            random_state=RANDOM_STATE,
        )
        connected, adj = try_connect_knn_graph(knn_indices, knn_distances, index)
        # May or may not connect; just ensure no crash and valid output types
        assert isinstance(connected, bool)
        if connected:
            assert adj is not None


class TestKnnGraphToSparse:
    def test_basic_properties(self, blob_data):
        X, _ = blob_data
        n = X.shape[0]
        _, knn_indices, knn_distances = build_knn_graph(
            X,
            n_neighbors=10,
            metric="euclidean",
            random_state=RANDOM_STATE,
        )
        sparse_mat = _knn_graph_to_sparse(knn_indices, knn_distances, n)
        assert sparse_mat.shape == (n, n)
        # Should be symmetric
        diff = sparse_mat - sparse_mat.T
        assert diff.nnz == 0 or np.abs(diff.data).max() < 1e-10

    def test_nonnegative_entries(self, blob_data):
        X, _ = blob_data
        _, knn_indices, knn_distances = build_knn_graph(
            X,
            n_neighbors=10,
            metric="euclidean",
            random_state=RANDOM_STATE,
        )
        sparse_mat = _knn_graph_to_sparse(knn_indices, knn_distances, X.shape[0])
        assert np.all(sparse_mat.data >= 0)


# ---------------------------------------------------------------------------
# Integration tests: compute_mst_from_knn_graph
# ---------------------------------------------------------------------------


class TestComputeMstFromKnnGraph:
    def test_mst_shape(self, blob_data):
        X, _ = blob_data
        mst, neighbors, core_dists = compute_mst_from_knn_graph(
            X,
            min_samples=5,
            metric="euclidean",
            random_state=RANDOM_STATE,
        )
        assert mst.shape == (X.shape[0] - 1, 3)
        assert core_dists.shape == (X.shape[0],)
        assert np.all(np.isfinite(core_dists))

    def test_mst_weights_nonnegative(self, blob_data):
        X, _ = blob_data
        mst, _, _ = compute_mst_from_knn_graph(
            X,
            min_samples=5,
            metric="euclidean",
            random_state=RANDOM_STATE,
        )
        # Weights might be +inf for disconnected bridges, otherwise >= 0
        finite_mask = np.isfinite(mst[:, 2])
        assert np.all(mst[finite_mask, 2] >= 0)


# ---------------------------------------------------------------------------
# Integration tests: HDBSCAN class and fast_hdbscan function
# ---------------------------------------------------------------------------


class TestHDBSCANArbitraryMetric:
    def test_cosine_smoke(self, cosine_data):
        """HDBSCAN with cosine metric should produce valid labels."""
        X, y_true = cosine_data
        model = HDBSCAN(
            min_cluster_size=10,
            min_samples=5,
            metric="cosine",
        ).fit(X)
        assert hasattr(model, "labels_")
        assert model.labels_.shape == (X.shape[0],)
        # Should find at least some clusters (not all noise)
        assert len(set(model.labels_)) > 1

    def test_manhattan_smoke(self, blob_data):
        """HDBSCAN with manhattan metric should produce valid labels."""
        X, y_true = blob_data
        model = HDBSCAN(
            min_cluster_size=10,
            min_samples=5,
            metric="manhattan",
        ).fit(X)
        assert hasattr(model, "labels_")
        assert model.labels_.shape == (X.shape[0],)
        assert len(set(model.labels_)) > 1

    def test_metric_kwds_minkowski(self, blob_data):
        """metric_kwds should be correctly forwarded to pynndescent."""
        X, _ = blob_data
        model = HDBSCAN(
            min_cluster_size=10,
            min_samples=5,
            metric="minkowski",
            metric_kwds={"p": 1.5},
        ).fit(X)
        assert model.labels_.shape == (X.shape[0],)

    def test_stores_raw_data(self, blob_data):
        """Fitted model should store _raw_data for arbitrary metrics."""
        X, _ = blob_data
        model = HDBSCAN(
            min_cluster_size=10,
            min_samples=5,
            metric="manhattan",
        ).fit(X)
        assert hasattr(model, "_raw_data")
        assert model._raw_data is not None

    def test_algorithm_ignored_for_arbitrary_metric(self, blob_data):
        """PyNNDescent path always uses Kruskal regardless of algorithm param."""
        X, _ = blob_data
        for algo in ("kruskal", "boruvka"):
            model = HDBSCAN(
                min_cluster_size=10,
                min_samples=5,
                metric="manhattan",
                algorithm=algo,
            ).fit(X)
            assert model.labels_.shape == (X.shape[0],)
            assert len(set(model.labels_)) > 1


class TestFastHdbscanArbitraryMetric:
    def test_cosine_functional_api(self, cosine_data):
        """fast_hdbscan() should work with cosine metric."""
        X, _ = cosine_data
        labels, probs = fast_hdbscan(
            X,
            min_cluster_size=10,
            min_samples=5,
            metric="cosine",
        )
        assert labels.shape == (X.shape[0],)
        assert probs.shape == (X.shape[0],)

    def test_return_trees(self, blob_data):
        """fast_hdbscan() with return_trees should return full output."""
        X, _ = blob_data
        result = fast_hdbscan(
            X,
            min_cluster_size=10,
            min_samples=5,
            metric="manhattan",
            return_trees=True,
        )
        # labels, probs, linkage, condensed, mst, neighbors, core_dists
        assert len(result) == 7


class TestParityWithEuclidean:
    """
    Check that the pynndescent euclidean path produces results comparable
    to the native KD-tree euclidean path (ARI >= threshold).
    """

    def test_euclidean_ari_parity(self, blob_data):
        X, y_true = blob_data
        # Native euclidean path
        model_native = HDBSCAN(
            min_cluster_size=10,
            min_samples=5,
            metric="euclidean",
        ).fit(X)

        # PyNNDescent euclidean path (forces nndescent by using 'l2' alias
        # or by directly calling compute_mst_from_knn_graph)
        from fast_hdbscan.hdbscan import (
            compute_minimum_spanning_tree,
            clusters_from_spanning_tree,
        )

        mst, neighbors, core_dists = compute_mst_from_knn_graph(
            X,
            min_samples=5,
            metric="euclidean",
            random_state=RANDOM_STATE,
        )
        labels_pynn, _, _, _, _ = clusters_from_spanning_tree(
            mst,
            min_cluster_size=10,
        )

        ari = adjusted_rand_score(model_native.labels_, labels_pynn)
        assert (
            ari > 0.8
        ), f"Expected ARI > 0.8 between native and pynndescent euclidean, got {ari:.3f}"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_invalid_algorithm(self, blob_data):
        X, _ = blob_data
        with pytest.raises(ValueError, match="algorithm must be"):
            fast_hdbscan(
                X,
                min_cluster_size=10,
                metric="manhattan",
                algorithm="invalid",
            )

    def test_metric_kwds_stored_on_model(self):
        model = HDBSCAN(metric="minkowski", metric_kwds={"p": 3})
        assert model.metric_kwds == {"p": 3}

    def test_get_params_includes_metric_kwds(self):
        model = HDBSCAN(metric="cosine", metric_kwds={"some_key": 42})
        params = model.get_params()
        assert "metric_kwds" in params
        assert params["metric_kwds"] == {"some_key": 42}
