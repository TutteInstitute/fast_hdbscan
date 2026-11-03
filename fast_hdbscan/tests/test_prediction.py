import numpy as np
from sklearn.datasets import make_blobs

from fast_hdbscan import HDBSCAN
from fast_hdbscan.prediction import approximate_predict

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=0)


def test_training_points_predict_their_own_labels():
    model = HDBSCAN(min_cluster_size=15, prediction_data=True).fit(X)
    labels, probabilities = approximate_predict(model, X)

    core_points = model.probabilities_ == 1.0
    assert core_points.sum() > 30
    np.testing.assert_array_equal(labels[core_points], model.labels_[core_points])
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
