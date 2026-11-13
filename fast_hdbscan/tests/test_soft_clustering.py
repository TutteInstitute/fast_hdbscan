from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from fast_hdbscan import HDBSCAN, all_points_membership_vectors

blobs, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.2, random_state=3)


@pytest.mark.parametrize("min_cluster_size", [5, 15, 40])
@pytest.mark.parametrize("cluster_selection_method", ["eom", "leaf"])
def test_matches_hdbscan_on_same_tree(min_cluster_size, cluster_selection_method):
    hdbscan = pytest.importorskip("hdbscan")
    from hdbscan.plots import CondensedTree
    from hdbscan.prediction import PredictionData

    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        cluster_selection_method=cluster_selection_method,
        prediction_data=True,
    ).fit(blobs)
    # hdbscan's soft clustering on our tree, so tree differences don't mix in
    condensed_tree = CondensedTree(model._condensed_tree, labels=model.labels_)
    reference = SimpleNamespace(
        condensed_tree_=condensed_tree,
        prediction_data_=PredictionData(blobs, condensed_tree, min_cluster_size),
    )

    np.testing.assert_allclose(
        all_points_membership_vectors(model),
        hdbscan.all_points_membership_vectors(reference),
    )


def test_core_points_belong_to_their_cluster():
    model = HDBSCAN(min_cluster_size=15, prediction_data=True).fit(blobs)
    vectors = all_points_membership_vectors(model)

    assert vectors.shape == (blobs.shape[0], model.labels_.max() + 1)
    assert (vectors >= 0).all()
    assert (vectors.sum(axis=1) <= 1 + 1e-12).all()
    core_points = model.probabilities_ == 1.0
    assert core_points.sum() > 30
    np.testing.assert_array_equal(
        vectors[core_points].argmax(axis=1), model.labels_[core_points]
    )


def test_non_finite_rows_get_zero_membership():
    finite_blobs = blobs[:298]
    with_nan = np.insert(finite_blobs, [0, 150], np.nan, axis=0)
    model = HDBSCAN(min_cluster_size=15, prediction_data=True).fit(with_nan)
    clean = HDBSCAN(min_cluster_size=15, prediction_data=True).fit(finite_blobs)

    vectors = all_points_membership_vectors(model)
    finite = np.isfinite(with_nan).all(axis=1)
    assert (vectors[~finite] == 0).all()
    np.testing.assert_allclose(vectors[finite], all_points_membership_vectors(clean))


def test_no_clusters_and_missing_prediction_data():
    single_cluster = HDBSCAN(min_cluster_size=200, prediction_data=True).fit(blobs[:150])
    assert single_cluster.labels_.max() == -1
    assert all_points_membership_vectors(single_cluster).shape == (150, 0)

    # epsilon selects the root but leaves every point as noise
    two_blobs, _ = make_blobs(n_samples=200, centers=2, random_state=0)
    all_noise = HDBSCAN(
        min_cluster_size=10,
        allow_single_cluster=True,
        cluster_selection_epsilon=0.01,
        prediction_data=True,
    ).fit(two_blobs)
    assert len(all_noise._selected_clusters) == 1
    assert all_noise.labels_.max() == -1
    assert all_points_membership_vectors(all_noise).shape == (200, 0)

    with pytest.raises(ValueError, match="prediction data"):
        all_points_membership_vectors(HDBSCAN(min_cluster_size=15).fit(blobs))


def test_points_missing_from_the_tree_raise():
    small_blobs, _ = make_blobs(n_samples=60, centers=3, random_state=0)
    model = HDBSCAN(min_cluster_size=1, prediction_data=True).fit(small_blobs)
    with pytest.raises(ValueError, match="missing from the condensed tree"):
        all_points_membership_vectors(model)
