"""sample_weights must behave like duplicated rows.

Each test pins one of the defects fixed in the sample_weights code path:
core distances from too few neighbours, plain and squared distances mixed,
a non-minimum edge in the Boruvka initialisation, point fall-outs stored with
size 1 in the condensed tree, and points with weight >= min_cluster_size
dropped to noise.
"""

import numpy as np
import pytest
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score

from fast_hdbscan import HDBSCAN
from fast_hdbscan.boruvka import initialize_boruvka_from_knn
from fast_hdbscan.disjoint_set import ds_rank_create
from fast_hdbscan.hdbscan import compute_minimum_spanning_tree

MIN_SAMPLES = 10


def three_blobs(seed, n_per_blob=400, dim=2, separation=6.0):
    rng = np.random.default_rng(seed)
    centers = np.zeros((3, dim))
    centers[1, 0] = separation
    centers[2, 1] = separation
    return np.vstack([c + rng.normal(size=(n_per_blob, dim)) for c in centers]), rng


def labels_weighted_and_expanded(X, weights, min_cluster_size, **kwargs):
    counts = weights.astype(int)
    expanded = np.repeat(X, counts, axis=0)
    inverse = np.repeat(np.arange(len(X)), counts)
    params = dict(min_samples=MIN_SAMPLES, min_cluster_size=min_cluster_size, **kwargs)
    labels_expanded = HDBSCAN(**params).fit(expanded).labels_
    labels_weighted = HDBSCAN(**params).fit(X, sample_weight=weights).labels_
    return labels_expanded, labels_weighted[inverse], labels_weighted


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("dim", [2, 5])
def test_weights_match_duplicated_rows(seed, dim):
    X, rng = three_blobs(seed, dim=dim, separation=6.0 if dim == 2 else 8.0)
    weights = rng.integers(1, 5, size=len(X)).astype(np.float32)
    expanded, weighted, _ = labels_weighted_and_expanded(X, weights, min_cluster_size=50)
    assert expanded.max() + 1 == 3
    assert adjusted_rand_score(expanded, weighted) > 0.99


def test_weights_of_one_equal_no_weights():
    X, _ = three_blobs(0)
    no_weights = HDBSCAN(min_samples=MIN_SAMPLES, min_cluster_size=50).fit(X).labels_
    ones = (
        HDBSCAN(min_samples=MIN_SAMPLES, min_cluster_size=50)
        .fit(X, sample_weight=np.ones(len(X), dtype=np.float32))
        .labels_
    )
    np.testing.assert_array_equal(no_weights, ones)


def test_weighted_core_distances_ignore_far_heavy_points():
    """Points with weight 1 whose neighbours have weight 1 keep a positive core distance."""
    X = np.random.default_rng(0).normal(size=(1000, 3))
    weights = np.ones(len(X), dtype=np.float32)
    weights[:10] = 1000.0
    core = compute_minimum_spanning_tree(
        X, min_samples=MIN_SAMPLES, sample_weights=weights
    )[2]
    assert (core[10:] > 0).all()


def test_weighted_core_distances_scale_with_data():
    X = np.random.default_rng(0).normal(size=(500, 3))
    weights = np.ones(len(X), dtype=np.float32)
    core_1 = compute_minimum_spanning_tree(
        X, min_samples=MIN_SAMPLES, sample_weights=weights
    )[2]
    core_4 = compute_minimum_spanning_tree(
        4 * X, min_samples=MIN_SAMPLES, sample_weights=weights
    )[2]
    np.testing.assert_allclose(core_4, 4 * core_1, rtol=1e-5)


def test_weighted_mst_is_the_exact_mutual_reachability_mst():
    X, rng = three_blobs(0, n_per_blob=100)
    weights = rng.integers(1, 5, size=len(X)).astype(np.float32)
    mst, _, core = compute_minimum_spanning_tree(
        X, min_samples=MIN_SAMPLES, sample_weights=weights
    )
    distances = cdist(X, X)
    order = np.argsort(distances, axis=1, kind="stable")
    cumulative = np.cumsum(weights[order], axis=1)
    reached = np.argmax(cumulative >= MIN_SAMPLES + 1, axis=1)
    exact_core = distances[np.arange(len(X)), order[np.arange(len(X)), reached]]
    np.testing.assert_allclose(core, exact_core, rtol=1e-5)
    reachability = np.maximum(np.maximum(distances, exact_core[:, None]), exact_core[None, :])
    np.fill_diagonal(reachability, 0.0)
    exact_total = minimum_spanning_tree(reachability).sum()
    np.testing.assert_allclose(mst[:, 2].sum(), exact_total, rtol=1e-5)


def test_boruvka_initialisation_takes_minimum_edge():
    # point 0: neighbour 1 at distance 1.5 with core 2.0, neighbour 2 at distance 3.0
    # with core 1.0; core of point 0 is 1.0. Mutual reachability: (0, 1) = 2.0, (0, 2) = 3.0.
    knn_indices = np.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]], dtype=np.int32)
    knn_distances = np.array(
        [[0.0, 1.5, 3.0], [0.0, 1.5, 1.5], [0.0, 1.5, 3.0]], dtype=np.float32
    )
    core = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    edges = initialize_boruvka_from_knn(knn_indices, knn_distances, core, ds_rank_create(3))
    edge_from_0 = edges[edges[:, 0] == 0]
    assert len(edge_from_0) == 0 or edge_from_0[0, 2] <= 2.0


@pytest.mark.parametrize("method", ["eom", "leaf"])
def test_heavy_points_match_duplicated_rows(method):
    """A point with weight >= min_cluster_size is its own cluster, like w identical rows."""
    X, rng = three_blobs(0, n_per_blob=300, dim=5, separation=8.0)
    weights = np.ones(len(X), dtype=np.float32)
    heavy = rng.choice(len(X), 12, replace=False)
    weights[heavy] = rng.integers(100, 400, size=len(heavy))
    expanded, weighted, per_point = labels_weighted_and_expanded(
        X, weights, min_cluster_size=100, cluster_selection_method=method
    )
    assert (per_point[heavy] >= 0).all()
    assert adjusted_rand_score(expanded, weighted) > 0.99


def test_point_entries_carry_their_weight():
    from fast_hdbscan.cluster_trees import (
        condense_tree,
        mst_to_linkage_tree_w_sample_weights,
    )

    X, rng = three_blobs(0, n_per_blob=200)
    weights = rng.integers(1, 5, size=len(X)).astype(np.float32)
    mst = compute_minimum_spanning_tree(
        X, min_samples=MIN_SAMPLES, sample_weights=weights
    )[0]
    mst = mst[np.lexsort((mst.T[1], mst.T[0], mst.T[2]))]
    tree = condense_tree(
        mst_to_linkage_tree_w_sample_weights(mst, weights),
        min_cluster_size=50,
        sample_weights=weights,
    )
    points = tree.child < len(X)
    np.testing.assert_array_equal(tree.child_size[points], weights[tree.child[points]])
    assert sorted(tree.child[points].tolist()) == list(range(len(X)))
