"""
Tests for PLSCAN clustering algorithm
Based on test_hdbscan.py
"""

import numpy as np
from sklearn.utils._testing import (
    assert_array_equal,
    assert_array_almost_equal,
)
from fast_hdbscan import PLSCAN
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import pytest

n_clusters = 3
X, y = make_blobs(n_samples=300, random_state=42, centers=n_clusters, cluster_std=1.0)
# X, y = shuffle(X, y, random_state=7)
# X = StandardScaler().fit_transform(X)


def test_plscan_feature_vector():
    labels = PLSCAN().fit(X).labels_
    print(labels)
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_1 == n_clusters


def test_plscan_input_lists():
    X_ = [[1.0, 2.0], [3.0, 4.0]]
    PLSCAN().fit(X_)  # must not raise exception


def test_plscan_badargs():
    with pytest.raises(ValueError):
        PLSCAN().fit("fail")
    with pytest.raises(ValueError):
        PLSCAN().fit(None)
    with pytest.raises(ValueError):
        PLSCAN(min_samples="fail").fit(X)
    with pytest.raises(ValueError):
        PLSCAN(min_samples=-1).fit(X)


def test_plscan_max_layers():
    model = PLSCAN(max_layers=1).fit(X)
    assert len(model.cluster_layers_) == 1


def test_plscan_layer_similarity_threshold():
    # Lower threshold should allow more layers
    model_low = PLSCAN(layer_similarity_threshold=0.01, max_layers=5).fit(X)
    model_high = PLSCAN(layer_similarity_threshold=0.9, max_layers=5).fit(X)
    assert len(model_low.cluster_layers_) >= len(model_high.cluster_layers_)


def test_plscan_reproducibility():
    model1 = PLSCAN(reproducible=True).fit(X)
    model2 = PLSCAN(reproducible=True).fit(X)
    assert_array_equal(model1.labels_, model2.labels_)


def test_plscan_base_n_clusters():
    # Should get close to requested n_clusters in first layer
    requested = 4
    model = PLSCAN(base_n_clusters=requested).fit(X)
    first_layer = model.cluster_layers_[0]
    n_found = len(set(first_layer)) - int(-1 in first_layer)
    assert abs(n_found - requested) <= 1


def test_plscan_multiple_layers():
    model = PLSCAN(max_layers=3, layer_similarity_threshold=0.1).fit(X)
    assert len(model.cluster_layers_) > 1
    # Each layer should have different clusterings
    unique_labels = [set(layer) for layer in model.cluster_layers_]
    assert len(set(map(tuple, unique_labels))) == len(model.cluster_layers_)


def test_plscan_tree_generation():
    model = PLSCAN(max_layers=3).fit(X)
    tree = model.cluster_tree_
    # Tree should be a dict with tuple keys and list values
    assert isinstance(tree, dict)
    for k, v in tree.items():
        assert isinstance(k, tuple)
        assert isinstance(v, list)
