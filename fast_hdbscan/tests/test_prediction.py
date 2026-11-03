import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from sklearn.datasets import make_blobs

from fast_hdbscan import HDBSCAN
from fast_hdbscan.prediction import (
    PredictionData,
    approximate_predict,
    approximate_predict_scores,
    nearest_mutual_reachability_neighbor,
    predict_clusters,
    predict_outlier_scores,
)

blobs, _, centers = make_blobs(
    n_samples=300, centers=3, cluster_std=0.6, random_state=0, return_centers=True
)


@pytest.fixture(scope="module")
def model():
    return HDBSCAN(min_cluster_size=15, prediction_data=True).fit(blobs)


def test_training_points_predict_their_own_labels(model):
    labels, probabilities = approximate_predict(model, blobs)

    core_points = model.probabilities_ == 1.0
    assert core_points.sum() > 30
    np.testing.assert_array_equal(labels[core_points], model.labels_[core_points])
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_new_point_core_distance_follows_fit_convention():
    # fit excludes the point itself, so a new point's core distance is its
    # min_samples-th nearest training point: index 1 for min_samples=2
    neighbor_distances = np.array([1.0, 2.0, 3.0, 4.0])
    neighbor_indices = np.array([0, 1, 2, 3])
    core_distances = np.array([0.5, 0.5, 0.5, 0.5])
    nearest, lambda_ = nearest_mutual_reachability_neighbor(
        neighbor_distances, neighbor_indices, core_distances, 2
    )
    assert nearest == 0
    assert lambda_ == pytest.approx(1 / 2.0)


def test_blob_centres_and_far_points(model):
    centre_labels, centre_probabilities = approximate_predict(model, centers)
    nearest_blob = [model.labels_[np.argmin(((blobs - c) ** 2).sum(axis=1))] for c in centers]
    np.testing.assert_array_equal(centre_labels, nearest_blob)
    assert (centre_probabilities > 0).all()

    far_labels, far_probabilities = approximate_predict(model, centers + 100.0)
    assert (far_labels == -1).all()
    assert (far_probabilities == 0).all()


def test_connecting_points_are_training_rows(model):
    new_points = centers + 0.1
    labels, _, connecting = approximate_predict(model, new_points, return_connecting_points=True)
    np.testing.assert_array_equal(model.labels_[connecting], labels)


def test_matches_hdbscan_on_held_out_points(model):
    hdbscan = pytest.importorskip("hdbscan")
    reference = hdbscan.HDBSCAN(min_cluster_size=15, prediction_data=True).fit(blobs)
    held_out, _ = make_blobs(n_samples=200, centers=centers, cluster_std=0.6, random_state=1)

    labels, _ = approximate_predict(model, held_out)
    reference_labels, _ = hdbscan.approximate_predict(reference, held_out)
    # align label ids through the training labels, which both fits agree on
    label_map = {
        ours: np.bincount(
            reference.labels_[(model.labels_ == ours) & (reference.labels_ >= 0)]
        ).argmax()
        for ours in set(model.labels_) - {-1}
    }
    label_map[-1] = -1
    mapped = np.array([label_map[label] for label in labels])
    assert (mapped == reference_labels).mean() >= 0.95


def test_non_finite_training_rows_keep_raw_ids(model):
    with_nan = np.insert(blobs, [0, 150], np.nan, axis=0)
    nan_model = HDBSCAN(min_cluster_size=15, prediction_data=True).fit(with_nan)
    new_points = centers + 0.1

    labels, probabilities, connecting = approximate_predict(
        nan_model, new_points, return_connecting_points=True
    )
    clean_labels, clean_probabilities = approximate_predict(model, new_points)
    np.testing.assert_array_equal(nan_model.labels_[connecting], labels)
    assert np.isfinite(with_nan[connecting]).all()
    np.testing.assert_allclose(probabilities, clean_probabilities)
    assert len(set(labels)) == len(set(clean_labels))


def test_scores_track_training_outlier_scores(model):
    scores = approximate_predict_scores(model, blobs)
    assert spearmanr(scores, model.outlier_scores_).statistic > 0.9
    far_score = approximate_predict_scores(model, centers[:1] + 100.0)
    assert far_score[0] > 0.99


def test_errors():
    plain = HDBSCAN(min_cluster_size=15).fit(blobs)
    with pytest.raises(ValueError, match="prediction data"):
        approximate_predict(plain, centers)
    plain.generate_prediction_data()
    with pytest.raises(ValueError, match="dimension"):
        approximate_predict(plain, centers[:, :1])

    with pytest.raises(NotImplementedError, match="euclidean"):
        # precomputed rather than another vector metric: it needs no pynndescent
        distances = csr_matrix(np.sqrt(((blobs[:, None] - blobs[None]) ** 2).sum(axis=-1)))
        HDBSCAN(min_cluster_size=15, metric="precomputed", prediction_data=True).fit(distances)
    with pytest.raises(NotImplementedError, match="sample_weight"):
        HDBSCAN(min_cluster_size=15, prediction_data=True).fit(
            blobs, sample_weight=np.ones(len(blobs))
        )


def test_no_clusters_predicts_noise():
    lonely = HDBSCAN(min_cluster_size=500, prediction_data=True).fit(blobs)
    with pytest.warns(UserWarning, match="any defined clusters"):
        labels, probabilities = approximate_predict(lonely, centers)
    assert (labels == -1).all()
    assert (probabilities == 0).all()


def test_scores_without_selected_clusters():
    tiny_clusters = HDBSCAN(min_cluster_size=15, max_cluster_size=2, prediction_data=True).fit(blobs)
    assert (tiny_clusters.labels_ == -1).all()
    scores = approximate_predict_scores(tiny_clusters, blobs)
    assert spearmanr(scores, tiny_clusters.outlier_scores_).statistic > 0.9


def test_single_root_cluster_keeps_fit_noise_threshold():
    gaussian = np.random.RandomState(0).normal(size=(100, 2))
    single = HDBSCAN(
        min_cluster_size=5, min_samples=3, allow_single_cluster=True, prediction_data=True
    ).fit(gaussian)
    assert set(single.labels_) == {-1, 0}
    labels, probabilities = approximate_predict(single, [[1e4, 1e4]])
    assert labels[0] == -1
    assert probabilities[0] == 0


def test_non_finite_new_points_are_rejected(model):
    with pytest.raises(ValueError, match="finite"):
        approximate_predict(model, [[np.nan, 0.0]])
    with pytest.raises(ValueError, match="finite"):
        approximate_predict_scores(model, [[np.inf, 0.0]])


def test_single_point_fit():
    single_point = HDBSCAN(min_cluster_size=2, prediction_data=True).fit([[0.0, 0.0]])
    with pytest.warns(UserWarning, match="any defined clusters"):
        labels, _ = approximate_predict(single_point, [[1.0, 1.0]])
    assert labels[0] == -1
    with pytest.warns(UserWarning, match="condensed tree"):
        scores = approximate_predict_scores(single_point, [[1.0, 1.0]])
    assert scores[0] == 1.0


def test_fortran_ordered_points(model):
    new_points = centers + 0.1
    expected = approximate_predict(model, new_points)
    fortran = approximate_predict(model, np.asfortranarray(new_points))
    np.testing.assert_array_equal(fortran[0], expected[0])
    np.testing.assert_allclose(fortran[1], expected[1])


@pytest.fixture(scope="module")
def hand_built(model):
    #            7 (root)
    #          /   \         lambda 1.0
    #         8     9        8 and 9 selected: labels 0 and 1
    #       / | \   | \
    #     10  11 6  4  5     point 6 at 1.5, point 4 at inf, point 5 at 1.0
    #    / \   |\             lambda 2.0
    #   0   1  2 3           point 0 at 4.0, point 1 at 2.0, points 2, 3 at 4.0
    rows = [
        (7, 8, 1.0, 5), (7, 9, 1.0, 2), (8, 10, 2.0, 2), (8, 11, 2.0, 2), (8, 6, 1.5, 1),
        (10, 0, 4.0, 1), (10, 1, 2.0, 1), (11, 2, 4.0, 1), (11, 3, 4.0, 1),
        (9, 4, np.inf, 1), (9, 5, 1.0, 1),
    ]
    tree = np.array(rows, dtype=model._condensed_tree.dtype)
    points = np.random.RandomState(0).normal(size=(7, 2))
    # min_samples=1 and zero core distances: a new point's lambda is 1 / distance
    data = PredictionData(points, np.arange(7), tree, [8, 9], np.zeros(7), 1, 0.0)
    return data.placement


def place(placement, neighbor, distance, kernel=predict_clusters):
    return kernel(np.array([[distance]]), np.array([[neighbor]]), placement)


def test_placement_walks_up_the_hand_built_tree(hand_built):
    # lambda 4 >= point 0's own lambda: stays in leaf 10, labelled by its selected parent 8
    labels, probabilities, _ = place(hand_built, 0, 0.25)
    assert (labels[0], probabilities[0]) == (0, 1.0)
    # lambda 1.2: walks up past leaf 10 (born at 2.0) into 8 (born at 1.0)
    labels, probabilities, _ = place(hand_built, 0, 1 / 1.2)
    assert labels[0] == 0
    assert probabilities[0] == pytest.approx(1.2 / 1.5)
    # lambda 1.0 equals the birth of 9, so the point leaves 9 for the root: noise
    labels, _, _ = place(hand_built, 4, 1.0)
    assert labels[0] == -1
    # lambda 1.0 equals point 5's own lambda: no walk up, it stays in 9
    labels, _, _ = place(hand_built, 5, 1.0)
    assert labels[0] == 1
    # an exact duplicate (infinite lambda) gets probability 1, as in fit, even
    # in a cluster whose max lambda is infinite (min_samples duplicates of point 4)
    labels, probabilities, _ = place(hand_built, 4, 0.0)
    assert (labels[0], probabilities[0]) == (1, 1.0)


def test_scores_on_the_hand_built_tree(hand_built):
    # 8's subtree reaches lambda 4, though points directly under 8 stop at 1.5
    assert place(hand_built, 6, 1.0, predict_outlier_scores)[0] == pytest.approx(0.75)
    # an exact duplicate of point 1 scores like point 1: (4 - 2) / 4
    assert place(hand_built, 1, 0.0, predict_outlier_scores)[0] == pytest.approx(0.5)
