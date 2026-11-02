import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from fast_hdbscan import HDBSCAN
from fast_hdbscan.hdbscan import clusters_from_spanning_tree

# Two pairs of nearby blobs and one far blob: without a limit EOM merges each
# pair, so splitting them shows the limit at work.
CENTERS = [[0, 0], [3, 0], [20, 0], [23, 0], [0, 20]]


def paired_blobs():
    X, _ = make_blobs(n_samples=2000, centers=CENTERS, cluster_std=0.8, random_state=0)
    return X


def same_partition(a, b):
    return np.array_equal(a == -1, b == -1) and adjusted_rand_score(a, b) == 1.0


@pytest.mark.parametrize("epsilon_max", [np.inf, 5.0, 1.0, 0.5])
@pytest.mark.parametrize("allow_single_cluster", [False, True])
def test_epsilon_max_matches_hdbscan_on_same_tree(epsilon_max, allow_single_cluster):
    hdbscan_tree = pytest.importorskip("hdbscan._hdbscan_tree")
    model = HDBSCAN(
        min_cluster_size=20,
        cluster_selection_epsilon_max=epsilon_max,
        allow_single_cluster=allow_single_cluster,
    ).fit(paired_blobs())

    tree = model._condensed_tree.astype(
        [("parent", np.intp), ("child", np.intp), ("lambda_val", float), ("child_size", np.intp)]
    )
    expected, _, _ = hdbscan_tree.get_clusters(
        tree,
        hdbscan_tree.compute_stability(tree),
        "eom",
        allow_single_cluster,
        False,
        0.0,
        0,
        epsilon_max,
    )
    assert same_partition(model.labels_, expected)


def test_epsilon_max_splits_clusters_born_above_it():
    X = paired_blobs()
    unlimited = HDBSCAN(min_cluster_size=20).fit(X)
    limited = HDBSCAN(min_cluster_size=20, cluster_selection_epsilon_max=2.0).fit(X)

    assert unlimited.labels_.max() + 1 == 3
    assert limited.labels_.max() + 1 == 4


def test_epsilon_max_with_disconnected_components():
    # The infinite edge joining the two components gives each a lambda of 0,
    # that is, an infinite birth distance. As in hdbscan, any finite limit then
    # rejects both components, and neither can split further.
    mst = np.array(
        [[0, 1, 0.1], [1, 2, 0.1], [3, 4, 0.1], [4, 5, 0.1], [2, 3, np.inf]]
    )
    unlimited, *_ = clusters_from_spanning_tree(mst, min_cluster_size=2)
    limited, *_ = clusters_from_spanning_tree(
        mst, min_cluster_size=2, cluster_selection_epsilon_max=1.0
    )
    assert same_partition(unlimited, np.array([0, 0, 0, 1, 1, 1]))
    np.testing.assert_array_equal(limited, np.full(6, -1))


@pytest.mark.parametrize("ss_algorithm", ["bc", "bc_simple"])
def test_epsilon_max_has_no_effect_on_semi_supervised_selection(ss_algorithm):
    X = paired_blobs()
    y = np.full(len(X), -1)
    y[:40] = 0
    params = dict(min_cluster_size=20, semi_supervised=True, ss_algorithm=ss_algorithm)
    unlimited = HDBSCAN(**params).fit(X, y)
    limited = HDBSCAN(cluster_selection_epsilon_max=0.5, **params).fit(X, y)
    np.testing.assert_array_equal(unlimited.labels_, limited.labels_)


def test_integer_epsilon_max_is_accepted():
    X = paired_blobs()
    integer = HDBSCAN(min_cluster_size=20, cluster_selection_epsilon_max=2).fit(X)
    floating = HDBSCAN(min_cluster_size=20, cluster_selection_epsilon_max=2.0).fit(X)
    np.testing.assert_array_equal(integer.labels_, floating.labels_)


def test_epsilon_max_has_no_effect_on_leaf_selection():
    X = paired_blobs()
    leaf = HDBSCAN(min_cluster_size=20, cluster_selection_method="leaf").fit(X)
    limited = HDBSCAN(
        min_cluster_size=20, cluster_selection_method="leaf", cluster_selection_epsilon_max=0.5
    ).fit(X)
    np.testing.assert_array_equal(leaf.labels_, limited.labels_)


@pytest.mark.parametrize("epsilon_max", [0.5, -1.0, np.nan, True, "2"])
def test_invalid_epsilon_max_raises(epsilon_max):
    with pytest.raises(ValueError):
        HDBSCAN(cluster_selection_epsilon=1.0, cluster_selection_epsilon_max=epsilon_max).fit(
            paired_blobs()
        )


@pytest.mark.parametrize("epsilon_max, expected_clusters", [(3.0, 1), (1.0, 2)])
def test_epsilon_max_limits_the_root_when_single_cluster_is_allowed(
    epsilon_max, expected_clusters
):
    # Two overlapping blobs: EOM picks the root, born at about 2.1, unless the
    # limit is below that.
    X, _ = make_blobs(
        n_samples=2000, centers=[[0, 0], [1.5, 0]], cluster_std=0.8, random_state=0
    )
    model = HDBSCAN(
        min_cluster_size=20,
        allow_single_cluster=True,
        cluster_selection_epsilon_max=epsilon_max,
    ).fit(X)
    assert model.labels_.max() + 1 == expected_clusters
