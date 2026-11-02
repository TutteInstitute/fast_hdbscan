import numpy as np
import pytest
from scipy import sparse
from sklearn.datasets import make_blobs
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import kneighbors_graph

from fast_hdbscan import HDBSCAN
from fast_hdbscan.hdbscan import relative_validity


def reference_relative_validity(mst, labels):
    clusters = sorted(set(labels) - {-1})
    sparseness = {c: 0.0 for c in clusters}
    separation = {c: np.inf for c in clusters}
    outlier_separation = np.inf
    for a, b, d in mst:
        la, lb = labels[int(a)], labels[int(b)]
        if np.isinf(d):
            if la == lb != -1:
                sparseness[la] = np.inf
            continue
        if la == -1 and lb == -1:
            continue
        if la == -1 or lb == -1:
            outlier_separation = min(outlier_separation, d)
        elif la == lb:
            sparseness[la] = max(sparseness[la], d)
        else:
            separation[la] = min(separation[la], d)
            separation[lb] = min(separation[lb], d)
    finite = mst[np.isfinite(mst[:, 2]), 2]
    max_distance = finite.max() if len(finite) else 0.0
    if len(clusters) > 1 or outlier_separation == np.inf:
        isolated = 2 * max_distance
    else:
        isolated = 2 * outlier_separation
    score = 0.0
    for c in clusters:
        sep = isolated if separation[c] == np.inf else separation[c]
        size = np.sum(labels == c)
        if np.isinf(sparseness[c]):
            score -= size
        else:
            score += size * (sep - sparseness[c]) / max(sep, sparseness[c])
    return score / len(labels)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("min_cluster_size", [5, 15])
def test_relative_validity_matches_reference(seed, min_cluster_size):
    X, _ = make_blobs(400, centers=4, cluster_std=1.0 + 0.4 * seed, random_state=seed)
    model = HDBSCAN(min_cluster_size=min_cluster_size).fit(X)
    expected = reference_relative_validity(model._min_spanning_tree, model.labels_)
    assert model.relative_validity_ == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_relative_validity_matches_hdbscan():
    hdbscan = pytest.importorskip("hdbscan")
    X, _ = make_blobs(800, centers=5, random_state=0)
    ours = HDBSCAN(min_cluster_size=15).fit(X)
    theirs = hdbscan.HDBSCAN(
        min_cluster_size=15, gen_min_span_tree=True, approx_min_span_tree=False
    ).fit(X)
    assert ours.relative_validity_ == pytest.approx(theirs.relative_validity_, rel=1e-6)


def test_relative_validity_counts_non_finite_rows_as_noise():
    X, _ = make_blobs(300, centers=3, random_state=0)
    X[[3, 50, 200]] = np.nan
    finite = np.isfinite(X).all(axis=1)
    with_nan = HDBSCAN(min_cluster_size=10).fit(X)
    without_nan = HDBSCAN(min_cluster_size=10).fit(X[finite])
    assert with_nan.relative_validity_ == pytest.approx(
        without_nan.relative_validity_ * finite.sum() / len(X)
    )


def test_relative_validity_ignores_edges_between_disconnected_components():
    near, _ = make_blobs(60, centers=1, random_state=1)
    X = np.vstack([near, near + 1000])
    graph = kneighbors_graph(X, n_neighbors=10, mode="distance")
    model = HDBSCAN(min_cluster_size=10, metric="precomputed").fit(graph)
    mst = model._min_spanning_tree
    assert np.isinf(mst[:, 2]).any()
    assert model.relative_validity_ == pytest.approx(
        reference_relative_validity(mst, model.labels_)
    )


def test_relative_validity_cluster_split_across_components_is_invalid():
    clique = sparse.csr_matrix(np.ones((4, 4)) - np.eye(4))
    graph = sparse.block_diag([clique] * 3, format="csr")
    model = HDBSCAN(
        metric="precomputed",
        min_cluster_size=2,
        min_samples=1,
        semi_supervised=True,
        allow_single_cluster=True,
    ).fit(graph, np.zeros(12, dtype=int))
    assert (model.labels_ == 0).all()
    assert model.relative_validity_ == -1.0


def test_relative_validity_all_noise_is_zero():
    X, _ = make_blobs(50, centers=2, random_state=0)
    model = HDBSCAN(min_cluster_size=len(X) + 1).fit(X)
    assert (model.labels_ == -1).all()
    assert model.relative_validity_ == 0.0


def test_relative_validity_single_point():
    assert relative_validity(np.zeros((0, 3)), np.array([-1])) == 0.0


def test_relative_validity_requires_fit():
    with pytest.raises(NotFittedError):
        HDBSCAN().relative_validity_
