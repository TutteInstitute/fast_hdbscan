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
    LayerClustering,
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

def test_layer_clustering():
    model = LayerClustering().fit(X)
    assert np.max(model.labels_) == 2