import inspect

import numpy as np
import pytest
from scipy import sparse
from sklearn.datasets import make_blobs
from sklearn.exceptions import NotFittedError
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph

from fast_hdbscan import HDBSCAN, fast_hdbscan
from fast_hdbscan.hdbscan import clusters_from_spanning_tree


def sorted_rows(points):
    return points[np.lexsort(points.T[::-1])]


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("min_cluster_size", [5, 15, 40])
def test_exemplars_match_hdbscan_on_same_tree(seed, min_cluster_size):
    pytest.importorskip("hdbscan")
    from hdbscan.plots import CondensedTree
    from hdbscan.prediction import PredictionData

    X, _ = make_blobs(n_samples=2000, centers=6, cluster_std=1.8, random_state=seed)
    model = HDBSCAN(min_cluster_size=min_cluster_size).fit(X)
    condensed_tree = CondensedTree(model._condensed_tree, labels=model.labels_)

    # hdbscan picks its exemplars from its own selection on the tree; compare
    # only where both libraries selected the same clusters.
    assert sorted(model._selected_clusters) == sorted(condensed_tree._select_clusters())
    expected = PredictionData(X, condensed_tree, min_cluster_size).exemplars
    assert len(model.exemplars_) == len(expected)
    for got, want in zip(model.exemplars_, expected):
        np.testing.assert_array_equal(sorted_rows(got), sorted_rows(want))


def test_exemplars_use_the_selected_cluster_when_epsilon_drops_its_points():
    # The root is selected but epsilon leaves most of it as noise; exemplars
    # still come from the leaves below the root and keep only label-0 points.
    X, _ = make_blobs(n_samples=100, centers=4, cluster_std=1.5, random_state=0)
    model = HDBSCAN(
        min_cluster_size=5, allow_single_cluster=True, cluster_selection_epsilon=0.7
    ).fit(X)

    np.testing.assert_array_equal(model._selected_clusters, [len(X)])
    (exemplars,) = model.exemplars_
    assert len(exemplars) > 0
    members = X[model.labels_ == 0]
    assert all((members == point).all(axis=1).any() for point in exemplars)


def test_exemplars_empty_when_epsilon_leaves_only_noise():
    X, _ = make_blobs(n_samples=100, centers=4, cluster_std=1.5, random_state=0)
    model = HDBSCAN(
        min_cluster_size=5, allow_single_cluster=True, cluster_selection_epsilon=0.01
    ).fit(X)

    assert model.labels_.max() == -1
    assert model.exemplars_ == []


def test_medoid_with_metric_unknown_to_sklearn():
    pytest.importorskip("pynndescent")
    X = np.abs(make_blobs(n_samples=120, centers=3, random_state=9)[0])
    model = HDBSCAN(metric="hellinger", min_cluster_size=5, knn_k=15).fit(X)
    assert model.labels_.max() >= 0
    with pytest.raises(NotImplementedError):
        model.weighted_cluster_medoid(0)
    assert model.weighted_cluster_centroid(0).shape == (2,)


def test_return_selected_clusters_keeps_default_outputs():
    X, _ = make_blobs(n_samples=300, centers=3, random_state=8)
    assert len(fast_hdbscan(X)) == 2
    assert len(fast_hdbscan(X, return_trees=True)) == 7
    *trees, selected = fast_hdbscan(X, return_trees=True, return_selected_clusters=True)
    # Keyword-only, so positional callers keep their old meaning.
    for function in (fast_hdbscan, clusters_from_spanning_tree):
        parameter = inspect.signature(function).parameters["return_selected_clusters"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert len(trees) == 7
    np.testing.assert_array_equal(trees[0], fast_hdbscan(X)[0])
    assert len(selected) == trees[0].max() + 1

    mst = trees[4]
    assert len(clusters_from_spanning_tree(mst, min_cluster_size=10)) == 5
    *_, selected_again = clusters_from_spanning_tree(
        mst, min_cluster_size=10, return_selected_clusters=True
    )
    np.testing.assert_array_equal(np.sort(selected_again), np.sort(selected))


@pytest.mark.parametrize(
    "params",
    [
        dict(cluster_selection_method="leaf"),
        dict(cluster_selection_epsilon=1.0),
        dict(allow_single_cluster=True),
        dict(max_cluster_size=150),
    ],
)
def test_exemplars_belong_to_their_label(params):
    X, _ = make_blobs(n_samples=800, centers=4, cluster_std=1.2, random_state=3)
    model = HDBSCAN(min_cluster_size=10, **params).fit(X)

    exemplars = model.exemplars_
    assert len(exemplars) == model.labels_.max() + 1
    for label, points in enumerate(exemplars):
        assert len(points) > 0
        members = X[model.labels_ == label]
        assert all((members == point).all(axis=1).any() for point in points)


def test_exemplars_with_non_finite_rows():
    X, _ = make_blobs(n_samples=300, centers=3, random_state=4)
    finite = HDBSCAN(min_cluster_size=10).fit(X)
    X_with_nan = np.vstack([X, np.full((2, 2), np.nan)])
    model = HDBSCAN(min_cluster_size=10).fit(X_with_nan)

    np.testing.assert_array_equal(model._selected_clusters, finite._selected_clusters + 2)
    for got, want in zip(model.exemplars_, finite.exemplars_):
        np.testing.assert_array_equal(sorted_rows(got), sorted_rows(want))


def test_exemplars_all_noise():
    X = np.random.RandomState(0).uniform(size=(20, 2))
    model = HDBSCAN(min_cluster_size=30).fit(X)
    assert model.exemplars_ == []


def test_weighted_centroid_and_medoid_match_hdbscan_formula():
    X, _ = make_blobs(n_samples=600, centers=3, cluster_std=1.0, random_state=5)
    model = HDBSCAN(min_cluster_size=15).fit(X)

    for label in range(model.labels_.max() + 1):
        mask = model.labels_ == label
        members, strengths = X[mask], model.probabilities_[mask]
        np.testing.assert_allclose(
            model.weighted_cluster_centroid(label),
            np.average(members, weights=strengths, axis=0),
        )
        weighted = pairwise_distances(members) * strengths
        np.testing.assert_array_equal(
            model.weighted_cluster_medoid(label),
            members[np.argmin(weighted.sum(axis=1))],
        )


@pytest.mark.parametrize("cluster_id", [-1, 99])
def test_cluster_representatives_reject_noise_and_unknown_ids(cluster_id):
    X, _ = make_blobs(n_samples=200, centers=2, random_state=6)
    model = HDBSCAN(min_cluster_size=10).fit(X)
    with pytest.raises(ValueError):
        model.weighted_cluster_centroid(cluster_id)
    with pytest.raises(ValueError):
        model.weighted_cluster_medoid(cluster_id)


def test_precomputed_has_no_exemplars():
    X, _ = make_blobs(n_samples=300, centers=3, random_state=7)
    graph = sparse.csr_matrix(kneighbors_graph(X, 15, mode="distance"))
    model = HDBSCAN(metric="precomputed", min_cluster_size=10).fit(graph)
    with pytest.raises(AttributeError):
        model.exemplars_
    with pytest.raises(AttributeError):
        model.weighted_cluster_centroid(0)


def test_exemplars_before_fit_raises():
    with pytest.raises(NotFittedError):
        HDBSCAN().exemplars_
    with pytest.raises(NotFittedError):
        HDBSCAN().weighted_cluster_centroid(0)
