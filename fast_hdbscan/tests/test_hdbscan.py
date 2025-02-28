"""
Tests for HDBSCAN clustering algorithm
Shamelessly based on (i.e. ripped off from) the DBSCAN test code
"""
import numpy as np
from scipy.spatial import distance
from scipy import sparse
from scipy import stats
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils._testing import (
    assert_array_equal,
    assert_array_almost_equal,
)
from fast_hdbscan import (
    HDBSCAN,
    fast_hdbscan,
)
from fast_hdbscan.hdbscan import clusters_from_spanning_tree

# from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode

from tempfile import mkdtemp
from functools import wraps
import pytest

from sklearn import datasets

import warnings

n_clusters = 3
# X = generate_clustered_data(n_clusters=n_clusters, n_samples_per_cluster=50)
X, y = make_blobs(n_samples=200, random_state=10)
X, y = shuffle(X, y, random_state=7)
X = StandardScaler().fit_transform(X)

X_missing_data = X.copy()
X_missing_data[0] = [np.nan, 1]
X_missing_data[5] = [np.nan, np.nan]


def test_missing_data():
    """Tests if nan data are treated as infinite distance from all other points and assigned to -1 cluster"""
    model = HDBSCAN().fit(X_missing_data)
    assert model.labels_[0] == -1
    assert model.labels_[5] == -1
    assert model.probabilities_[0] == 0
    assert model.probabilities_[5] == 0
    assert model.probabilities_[5] == 0
    clean_indices = list(range(1, 5)) + list(range(6, 200))
    clean_model = HDBSCAN().fit(X_missing_data[clean_indices])
    assert np.allclose(clean_model.labels_, model.labels_[clean_indices])


def generate_noisy_data():
    blobs, _ = datasets.make_blobs(
        n_samples=200, centers=[(-0.75, 2.25), (1.0, 2.0)], cluster_std=0.25
    )
    moons, _ = datasets.make_moons(n_samples=200, noise=0.05)
    noise = np.random.uniform(-1.0, 3.0, (50, 2))
    return np.vstack([blobs, moons, noise])


def homogeneity(labels1, labels2):
    num_missed = 0.0
    for label in set(labels1):
        matches = labels2[labels1 == label]
        match_mode = mode(matches)[0][0]
        num_missed += np.sum(matches != match_mode)

    for label in set(labels2):
        matches = labels1[labels2 == label]
        match_mode = mode(matches)[0][0]
        num_missed += np.sum(matches != match_mode)

    return num_missed / 2.0


def test_hdbscan_feature_vector():
    labels, p, ltree, ctree, mtree, neighbors, cdists = fast_hdbscan(X, return_trees=True)
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_1 == n_clusters

    labels = HDBSCAN().fit(X).labels_
    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_2 == n_clusters


def test_hdbscan_dbscan_clustering():
    clusterer = HDBSCAN().fit(X)
    labels = clusterer.dbscan_clustering(0.3)
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert(n_clusters == n_clusters_1)


def test_hdbscan_no_clusters():
    labels, p= fast_hdbscan(X, min_cluster_size=len(X) + 1)
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_1 == 0

    labels = HDBSCAN(min_cluster_size=len(X) + 1).fit(X).labels_
    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_2 == 0


def test_hdbscan_sample_weight():
    sample_weight = y.astype(np.float32)
    labels, p = fast_hdbscan(X, sample_weights=sample_weight)
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_1 == n_clusters - 1


def test_hdbscan_min_cluster_size():
    for min_cluster_size in range(2, len(X) + 1, 1):
        labels, p = fast_hdbscan(
            X, min_cluster_size=min_cluster_size
        )
        true_labels = [label for label in labels if label != -1]
        if len(true_labels) != 0:
            assert np.min(np.bincount(true_labels)) >= min_cluster_size

        labels = HDBSCAN(min_cluster_size=min_cluster_size).fit(X).labels_
        true_labels = [label for label in labels if label != -1]
        if len(true_labels) != 0:
            assert np.min(np.bincount(true_labels)) >= min_cluster_size


def test_hdbscan_input_lists():
    X = [[1.0, 2.0], [3.0, 4.0]]
    HDBSCAN().fit(X)  # must not raise exception


def test_hdbscan_badargs():
    with pytest.raises(ValueError): 
        fast_hdbscan("fail")
    with pytest.raises(ValueError): 
        fast_hdbscan(None)
    with pytest.raises(ValueError): 
        fast_hdbscan(X, min_cluster_size="fail")
    with pytest.raises(ValueError): 
        fast_hdbscan(X, min_samples="fail")
    with pytest.raises(ValueError): 
        fast_hdbscan(X, min_samples=-1)
    with pytest.raises(ValueError): 
        fast_hdbscan(X, cluster_selection_epsilon="fail")
    with pytest.raises(ValueError): 
        fast_hdbscan(X, cluster_selection_epsilon=-1)
    with pytest.raises(ValueError): 
        fast_hdbscan(X, cluster_selection_epsilon=-0.1)
    with pytest.raises(ValueError): 
        fast_hdbscan(X, cluster_selection_persistence="fail")
    with pytest.raises(ValueError): 
        fast_hdbscan(X, cluster_selection_persistence=1)
    with pytest.raises(ValueError): 
        fast_hdbscan(X, cluster_selection_persistence=-0.1)
    with pytest.raises(ValueError): 
        fast_hdbscan(X, cluster_selection_method="fail")
    with pytest.raises(ValueError): 
        fast_hdbscan(X, semi_supervised=True, ss_algorithm="fail")
    with pytest.raises(ValueError): 
        fast_hdbscan(X, semi_supervised=True, data_labels=None)


def test_hdbscan_allow_single_cluster_with_epsilon():
    np.random.seed(0)
    no_structure = np.random.rand(150, 2)
    # without epsilon we should see 68 noise points and 8 labels
    c = HDBSCAN(
        min_cluster_size=5,
        cluster_selection_epsilon=0.0,
    ).fit(no_structure)
    unique_labels, counts = np.unique(c.labels_, return_counts=True)
    assert len(unique_labels) == 8
    assert counts[unique_labels == -1] == 68

    # An epsilon of 0.2 will produce 2 noise points and 2 labels
    # Allow single cluster does not prevent applying the epsilon threshold.
    c = HDBSCAN(
        min_cluster_size=5,
        cluster_selection_epsilon=0.2
    ).fit(no_structure)
    unique_labels, counts = np.unique(c.labels_, return_counts=True)
    assert len(unique_labels) == 2
    assert counts[unique_labels == -1] == 2

def test_hdbscan_max_cluster_size():
    model = HDBSCAN(max_cluster_size=30).fit(X)
    assert len(set(model.labels_)) >= 3
    for label in set(model.labels_):
        if label != -1:
            assert np.sum(model.labels_ == label) <= 30


def test_hdbscan_persistence_threshold():
    model = HDBSCAN(
        min_cluster_size=5,
        cluster_selection_method="leaf",
        cluster_selection_persistence=20.0,
    ).fit(X)
    assert np.all(model.labels_ == -1)


def test_mst_entry():
    # Assumes default keyword arguments match between class and function
    model = HDBSCAN(min_cluster_size=5).fit(X)
    (
        labels, 
        probabilities, 
        linkage_tree, 
        condensed_tree, 
        sorted_mst
    ) = clusters_from_spanning_tree(model._min_spanning_tree, min_cluster_size=5)
    assert np.all(model.labels_ == labels)
    assert np.allclose(model.probabilities_, probabilities)
    assert np.allclose(model._min_spanning_tree, sorted_mst)
    assert np.allclose(model._single_linkage_tree, linkage_tree)
    assert np.allclose(model._condensed_tree['parent'], condensed_tree.parent)
    assert np.allclose(model._condensed_tree['child'], condensed_tree.child)
    assert np.allclose(model._condensed_tree['lambda_val'], condensed_tree.lambda_val)
    assert np.allclose(model._condensed_tree['child_size'], condensed_tree.child_size)


# Disable for now -- need to refactor to meet newer standards
@pytest.mark.skip(reason="need to refactor to meet newer standards")
def test_hdbscan_is_sklearn_estimator():
    check_estimator(HDBSCAN())


# Probably not applicable now #
# def test_dbscan_sparse():
# def test_dbscan_balltree():
# def test_pickle():
# def test_dbscan_core_samples_toy():
# def test_boundaries():
