import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.exceptions import NotFittedError

from fast_hdbscan import HDBSCAN


def reference_outlier_scores(tree, num_points):
    """GLOSH written the direct way: the largest lambda in each cluster's subtree."""

    def subtree_max_lambda(cluster):
        rows = tree[tree["parent"] == cluster]
        best = rows["lambda_val"].max() if len(rows) else 0.0
        for child in rows["child"][rows["child"] >= num_points]:
            best = max(best, subtree_max_lambda(child))
        return best

    scores = np.zeros(num_points)
    for row in tree[tree["child"] < num_points]:
        lambda_max = subtree_max_lambda(row["parent"])
        if lambda_max == 0.0 or not np.isfinite(row["lambda_val"]):
            continue
        with np.errstate(invalid="ignore"):  # inf / inf is the expected NaN
            scores[row["child"]] = (lambda_max - row["lambda_val"]) / lambda_max
    return scores


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("min_cluster_size", [5, 15])
def test_outlier_scores_match_reference(seed, min_cluster_size):
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.5, random_state=seed)
    model = HDBSCAN(min_cluster_size=min_cluster_size).fit(X)

    expected = reference_outlier_scores(model._condensed_tree, len(X))
    np.testing.assert_allclose(model.outlier_scores_, expected)



@pytest.mark.parametrize("seed", [1, 2, 3])
def test_outlier_scores_after_persistence_simplification(seed):
    # simplify_hierarchy renumbers the clusters the kernel walks by id.
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.5, random_state=seed)
    model = HDBSCAN(min_cluster_size=5, cluster_selection_persistence=0.1).fit(X)

    expected = reference_outlier_scores(model._condensed_tree, len(X))
    np.testing.assert_allclose(model.outlier_scores_, expected)

def test_outlier_scores_match_hdbscan_on_same_tree():
    hdbscan_tree = pytest.importorskip("hdbscan._hdbscan_tree")
    X, _ = make_blobs(n_samples=2000, centers=6, cluster_std=2.0, random_state=0)
    model = HDBSCAN(min_cluster_size=10).fit(X)

    expected = hdbscan_tree.outlier_scores(model._condensed_tree)
    np.testing.assert_allclose(model.outlier_scores_, expected)


def test_outlier_scores_are_highest_for_far_away_points():
    X, _ = make_blobs(n_samples=500, centers=3, cluster_std=0.5, random_state=1)
    far_points = np.array([[40.0, 40.0], [-40.0, 40.0]])
    model = HDBSCAN(min_cluster_size=10).fit(np.vstack([X, far_points]))

    scores = model.outlier_scores_
    assert scores.shape == (len(X) + 2,)
    assert np.all((scores >= 0) & (scores <= 1))
    assert scores[-2:].min() > np.quantile(scores[:-2], 0.99)


def test_outlier_scores_with_non_finite_rows():
    X, _ = make_blobs(n_samples=200, centers=3, random_state=2)
    X[[5, 50]] = np.nan
    model = HDBSCAN(min_cluster_size=10).fit(X)

    scores = model.outlier_scores_
    assert scores.shape == (len(X),)
    expected = reference_outlier_scores(model._condensed_tree, len(X))
    np.testing.assert_allclose(scores, expected)


def test_outlier_scores_before_fit_raises():
    with pytest.raises(NotFittedError):
        HDBSCAN().outlier_scores_


def duplicate_points_model():
    X = np.vstack([np.zeros((8, 2)), [[1.0, 0.0], [2.0, 0.0]], np.full((8, 2), 10.0)])
    return HDBSCAN(min_samples=2, min_cluster_size=3).fit(X)


def test_outlier_scores_with_duplicate_points():
    # Duplicates give infinite lambdas, which turn some scores into NaN, as upstream.
    model = duplicate_points_model()
    scores = model.outlier_scores_

    assert np.isnan(scores).any()
    expected = reference_outlier_scores(model._condensed_tree, len(scores))
    np.testing.assert_allclose(scores, expected, equal_nan=True)


def test_outlier_scores_match_hdbscan_with_duplicate_points():
    hdbscan_tree = pytest.importorskip("hdbscan._hdbscan_tree")
    model = duplicate_points_model()

    expected = hdbscan_tree.outlier_scores(model._condensed_tree)
    np.testing.assert_allclose(model.outlier_scores_, expected, equal_nan=True)


def test_outlier_scores_single_point():
    model = HDBSCAN(min_cluster_size=2).fit(np.array([[0.0, 0.0]]))
    np.testing.assert_array_equal(model.outlier_scores_, [0.0])
