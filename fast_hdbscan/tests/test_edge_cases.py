import numpy as np
import pytest
from sklearn.datasets import make_blobs
from fast_hdbscan import HDBSCAN


class TestEdgeCaseInputs:
    def test_single_point(self):
        """A single point should be labeled as noise."""
        X = np.array([[1.0, 2.0]])
        labels = HDBSCAN(min_cluster_size=2).fit_predict(X)
        assert labels[0] == -1

    def test_two_points(self):
        """Two points with min_cluster_size=2 should form one cluster."""
        X = np.array([[0.0, 0.0], [0.1, 0.0]])
        labels = HDBSCAN(min_cluster_size=2).fit_predict(X)
        assert len(set(labels) - {-1}) <= 1

    def test_duplicate_points(self):
        """All identical points should not crash."""
        X = np.full((50, 3), 1.0)
        labels = HDBSCAN(min_cluster_size=5).fit_predict(X)
        assert len(labels) == 50

    def test_min_cluster_size_larger_than_n_samples(self):
        """When min_cluster_size > n_samples, all points should be noise."""
        np.random.seed(42)
        X = np.random.rand(10, 2)
        labels = HDBSCAN(min_cluster_size=100).fit_predict(X)
        assert np.all(labels == -1)

    def test_one_dimensional_data(self):
        """1D data should cluster without error."""
        np.random.seed(42)
        X = np.concatenate(
            [
                np.random.normal(0, 0.1, 50),
                np.random.normal(5, 0.1, 50),
            ]
        ).reshape(-1, 1)
        labels = HDBSCAN(min_cluster_size=10).fit_predict(X)
        n_clusters = len(set(labels) - {-1})
        assert n_clusters >= 1

    def test_high_min_samples(self):
        """min_samples close to n_samples should not crash."""
        np.random.seed(42)
        X = np.random.rand(20, 2)
        labels = HDBSCAN(min_cluster_size=5, min_samples=18).fit_predict(X)
        assert len(labels) == 20

    def test_epsilon_zero(self):
        """cluster_selection_epsilon=0 should behave normally."""
        X, _ = make_blobs(n_samples=200, centers=3, random_state=42)
        labels = HDBSCAN(
            min_cluster_size=10, cluster_selection_epsilon=0.0
        ).fit_predict(X)
        assert len(set(labels) - {-1}) >= 1

    def test_epsilon_very_large(self):
        """Very large epsilon should merge everything into one cluster."""
        X, _ = make_blobs(n_samples=200, centers=3, random_state=42)
        labels = HDBSCAN(
            min_cluster_size=10,
            cluster_selection_epsilon=1000.0,
            allow_single_cluster=True,
        ).fit_predict(X)
        n_clusters = len(set(labels) - {-1})
        assert n_clusters <= 1
