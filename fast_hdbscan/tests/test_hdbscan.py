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
import sys

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
    labels, p, ltree, ctree, mtree, neighbors, cdists = fast_hdbscan(
        X, return_trees=True
    )
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_1 == n_clusters

    labels = HDBSCAN().fit(X).labels_
    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_2 == n_clusters


def test_hdbscan_dbscan_clustering():
    clusterer = HDBSCAN().fit(X)
    labels = clusterer.dbscan_clustering(0.3)
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert n_clusters == n_clusters_1


def test_hdbscan_no_clusters():
    labels, p = fast_hdbscan(X, min_cluster_size=len(X) + 1)
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
        labels, p = fast_hdbscan(X, min_cluster_size=min_cluster_size)
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
    assert counts[unique_labels == -1] == 69

    # An epsilon of 0.2 will produce 2 noise points and 2 labels
    # Allow single cluster does not prevent applying the epsilon threshold.
    c = HDBSCAN(min_cluster_size=5, cluster_selection_epsilon=0.2).fit(no_structure)
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
    (labels, probabilities, linkage_tree, condensed_tree, sorted_mst) = (
        clusters_from_spanning_tree(model._min_spanning_tree, min_cluster_size=5)
    )
    assert np.all(model.labels_ == labels)
    assert np.allclose(model.probabilities_, probabilities)
    assert np.allclose(model._min_spanning_tree, sorted_mst)
    assert np.allclose(model._single_linkage_tree, linkage_tree)
    assert np.allclose(model._condensed_tree["parent"], condensed_tree.parent)
    assert np.allclose(model._condensed_tree["child"], condensed_tree.child)
    assert np.allclose(model._condensed_tree["lambda_val"], condensed_tree.lambda_val)
    assert np.allclose(model._condensed_tree["child_size"], condensed_tree.child_size)


# Disable for now -- need to refactor to meet newer standards
@pytest.mark.skip(reason="We accept NaNs in the input data")
def test_hdbscan_is_sklearn_estimator():
    check_estimator(HDBSCAN())


# Probably not applicable now #
# def test_dbscan_sparse():
# def test_dbscan_balltree():
# def test_pickle():
# def test_dbscan_core_samples_toy():
# def test_boundaries():


# -----------------------------------------------------------------------
# Precomputed sparse metric tests
# -----------------------------------------------------------------------


def _make_simple_sparse_graph(n=8, seed=42):
    """Return a connected sparse distance graph on n nodes."""
    rng = np.random.RandomState(seed)
    # Chain graph with some extra edges so it's well-connected
    rows, cols, data = [], [], []
    for i in range(n - 1):
        w = rng.uniform(0.1, 1.0)
        rows += [i, i + 1]
        cols += [i + 1, i]
        data += [w, w]
    # Extra edges
    for _ in range(n):
        i, j = rng.choice(n, 2, replace=False)
        w = rng.uniform(0.1, 2.0)
        rows += [i, j]
        cols += [j, i]
        data += [w, w]
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)


def test_hdbscan_precomputed_sparse_basic_fit():
    """Fit succeeds on a connected sparse precomputed graph."""
    G = _make_simple_sparse_graph(n=20)
    model = HDBSCAN(min_cluster_size=3, metric="precomputed").fit(G)
    assert hasattr(model, "labels_")
    assert len(model.labels_) == 20
    assert hasattr(model, "probabilities_")
    assert len(model.probabilities_) == 20

    labels, p = fast_hdbscan(G, min_cluster_size=3, metric="precomputed")
    assert len(labels) == 20


def test_precomputed_explicit_zero_edge_preserved():
    """Explicit 0.0 edges must be preserved and can appear in MST."""
    # 4 nodes: 0-1 at distance 0.0, 1-2 at 1.0, 2-3 at 1.0
    rows = [0, 1, 1, 2, 2, 3]
    cols = [1, 0, 2, 1, 3, 2]
    data = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    G = sparse.csr_matrix((data, (rows, cols)), shape=(4, 4), dtype=np.float64)

    from fast_hdbscan.hdbscan import compute_minimum_spanning_tree

    # min_samples=1 -> core_distances=0 -> MRD == base weight
    mst, neighbors, core_dists = compute_minimum_spanning_tree(
        G, min_samples=1, metric="precomputed"
    )
    assert mst.shape == (3, 3), "MST must have n-1=3 edges"
    # The 0.0 edge (0,1) should be in the MST
    weights = mst[:, 2]
    assert np.any(weights == 0.0), "MST should contain the 0.0-weight edge"


def test_precomputed_asymmetric_min_symmetrization():
    """Asymmetric weights must be symmetrized by taking the minimum."""
    # 3 nodes, asymmetric: w(0->1)=5.0, w(1->0)=1.0, chain 1-2=2.0
    rows = [0, 1, 1, 2]
    cols = [1, 0, 2, 1]
    data = [5.0, 1.0, 2.0, 2.0]
    G = sparse.csr_matrix((data, (rows, cols)), shape=(3, 3), dtype=np.float64)

    from fast_hdbscan.hdbscan import compute_minimum_spanning_tree

    mst, _, _ = compute_minimum_spanning_tree(G, min_samples=1, metric="precomputed")
    assert mst.shape == (2, 3)
    # Edge (0,1) should have weight min(5.0, 1.0) = 1.0 (possibly inflated by core dist)
    edge_01_mask = ((mst[:, 0] == 0) & (mst[:, 1] == 1)) | (
        (mst[:, 0] == 1) & (mst[:, 1] == 0)
    )
    assert edge_01_mask.any(), "Edge (0,1) must be in MST"
    assert mst[edge_01_mask, 2][0] == pytest.approx(
        1.0
    ), "Edge (0,1) weight should be min(5.0, 1.0) = 1.0"


def test_precomputed_disconnected_autobridge_inf():
    """Disconnected graph: MST has n-1 edges with exactly one +inf bridge."""
    # Component A: nodes 0,1,2 chain; Component B: nodes 3,4 chain
    rows = [0, 1, 1, 2, 3, 4]
    cols = [1, 0, 2, 1, 4, 3]
    data = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    G = sparse.csr_matrix((data, (rows, cols)), shape=(5, 5), dtype=np.float64)

    from fast_hdbscan.hdbscan import compute_minimum_spanning_tree

    mst, _, _ = compute_minimum_spanning_tree(G, min_samples=1, metric="precomputed")
    assert mst.shape == (4, 3), "MST must have n-1=4 edges"
    inf_edges = mst[np.isinf(mst[:, 2])]
    assert len(inf_edges) == 1, "Exactly one +inf bridge edge expected"


def test_precomputed_isolated_nodes_autobridge_inf_chain():
    """Graph with no edges: all n-1 MST edges should be +inf bridges."""
    n = 5
    G = sparse.csr_matrix((n, n), dtype=np.float64)

    from fast_hdbscan.hdbscan import compute_minimum_spanning_tree

    mst, _, _ = compute_minimum_spanning_tree(G, min_samples=1, metric="precomputed")
    assert mst.shape == (n - 1, 3), "MST must have n-1 edges"
    assert np.all(np.isinf(mst[:, 2])), "All bridge edges should be +inf"


def test_precomputed_reject_sample_weight():
    """sample_weight with metric='precomputed' must raise NotImplementedError."""
    G = _make_simple_sparse_graph(n=8)
    with pytest.raises(NotImplementedError):
        HDBSCAN(metric="precomputed").fit(G, sample_weight=np.ones(8))
    with pytest.raises(NotImplementedError):
        fast_hdbscan(G, metric="precomputed", sample_weights=np.ones(8))


def test_precomputed_semi_supervised_requires_labels():
    """Precomputed + semi-supervised must raise a clear error when y is missing."""
    G = _make_simple_sparse_graph(n=8)
    with pytest.raises(ValueError, match="must not be None"):
        HDBSCAN(metric="precomputed", semi_supervised=True).fit(G)


def test_precomputed_semi_supervised_label_length_validation():
    """Precomputed + semi-supervised must validate label count against node count."""
    G = _make_simple_sparse_graph(n=8)
    with pytest.raises(ValueError, match="one label per node"):
        HDBSCAN(metric="precomputed", semi_supervised=True).fit(G, np.array([0, 1, 2]))


def test_precomputed_semi_supervised_matches_euclidean_behavior():
    """
    Semi-supervised clustering should behave the same for euclidean and precomputed paths.

    We use a full pairwise graph built from X and verify exact label parity after
    canonical label alignment, matching the repository's metric parity strategy.
    """
    if sys.version_info < (3, 12):
        pytest.skip(
            "Numba typing instability in legacy bcubed extraction on Python < 3.12"
        )

    X_local, _ = make_blobs(
        n_samples=40,
        centers=[[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]],
        cluster_std=0.2,
        random_state=11,
    )
    X_local = StandardScaler().fit_transform(X_local)
    G = _make_full_pairwise_sparse(X_local)

    # Labels include supervised examples and unlabeled points (-1).
    y_local = np.full(X_local.shape[0], -1, dtype=np.int64)
    y_local[0] = 0
    y_local[10] = 1
    y_local[20] = 2

    # Use bc_simple in this parity test to avoid numba instability in the
    # optional virtual-node path on some platforms while still exercising
    # semi-supervised parity between euclidean and precomputed metrics.
    eu = HDBSCAN(
        min_cluster_size=5,
        semi_supervised=True,
        ss_algorithm="bc_simple",
    ).fit(X_local, y_local)
    pre = HDBSCAN(
        min_cluster_size=5,
        semi_supervised=True,
        ss_algorithm="bc_simple",
        metric="precomputed",
    ).fit(G, y_local)

    aligned = _align_labels(eu.labels_, pre.labels_)
    assert np.array_equal(eu.labels_, aligned)
    assert np.allclose(eu.probabilities_, pre.probabilities_)


def test_precomputed_validation_non_square():
    """Non-square matrix must raise ValueError."""
    G = sparse.csr_matrix(np.ones((4, 5)))
    with pytest.raises(ValueError, match="square"):
        HDBSCAN(metric="precomputed").fit(G)


def test_precomputed_validation_negative_weight():
    """Negative stored weight must raise ValueError."""
    rows, cols, data = [0, 1], [1, 0], [-1.0, -1.0]
    G = sparse.csr_matrix((data, (rows, cols)), shape=(3, 3))
    with pytest.raises(ValueError, match="non-negative"):
        HDBSCAN(metric="precomputed").fit(G)


def test_precomputed_validation_nonfinite_weight():
    """NaN/Inf stored weight must raise ValueError."""
    rows, cols, data = [0, 1], [1, 0], [np.nan, np.nan]
    G = sparse.csr_matrix((data, (rows, cols)), shape=(3, 3))
    with pytest.raises(ValueError, match="finite"):
        HDBSCAN(metric="precomputed").fit(G)


def test_precomputed_validation_dense_matrix():
    """Dense matrix with metric='precomputed' must raise ValueError."""
    G = np.ones((4, 4))
    with pytest.raises(ValueError, match="sparse"):
        HDBSCAN(metric="precomputed").fit(G)


def test_subcluster_rejects_precomputed_clusterer():
    """SubClusterDetector must raise NotImplementedError on a precomputed-fitted model."""
    from fast_hdbscan.sub_clusters import find_sub_clusters

    G = _make_simple_sparse_graph(n=20)
    model = HDBSCAN(min_cluster_size=3, metric="precomputed").fit(G)
    with pytest.raises(NotImplementedError):
        find_sub_clusters(model)


def test_plscan_rejects_sparse_input():
    """PLSCAN must raise ValueError for sparse input when metric is not precomputed."""
    from fast_hdbscan import PLSCAN

    G = _make_simple_sparse_graph(n=20)
    plscan = PLSCAN()
    with pytest.raises(ValueError, match="dense"):
        plscan.fit_predict(G)

    # But sparse should be accepted with metric='precomputed'
    plscan_precomputed = PLSCAN(metric="precomputed")
    plscan_precomputed.fit_predict(G)
    assert hasattr(plscan_precomputed, "labels_")
    assert len(plscan_precomputed.labels_) == 20


# -----------------------------------------------------------------------
# Parity tests: metric='precomputed' vs 'euclidean' must produce equivalent
# results when given the full pairwise distance matrix.
#
# Precision note: the Euclidean path uses float32 data internally (build_kdtree
# casts to float32; boruvka.py uses rdist = squared-Euclidean in float32 and
# converts via np.sqrt at line 611-612).  The precomputed path receives float64
# cdist distances.  Therefore intermediate values (core distances, MST edge
# weights) may differ by up to ~float32 epsilon (~1e-4), but on well-separated
# cluster data the condensed-tree structure is unambiguous and final labels
# must be exactly identical.
# -----------------------------------------------------------------------


def _make_full_pairwise_sparse(X):
    """
    Build a complete pairwise Euclidean distance CSR matrix from feature matrix X.

    X is cast to float32 before computing distances, matching the internal
    precision of build_kdtree (numba_kdtree.py:463).  Off-diagonal distances
    for continuous data are always > 0, so sparse.csr_matrix drops only the
    diagonal zeros (self-loops), which the precomputed pipeline already ignores.

    Returns a symmetric CSR matrix storing all n*(n-1) directed distances.
    """
    X32 = X.astype(np.float32)
    D = distance.cdist(X32, X32)  # float64 from float32 inputs
    np.fill_diagonal(D, 0.0)  # self-loops → 0 → eliminated by CSR
    return sparse.csr_matrix(D)


def _align_labels(ref, other):
    """
    Remap *other*'s integer labels to maximally agree with *ref*.

    Algorithm: majority-vote permutation.
    For each cluster id ``o`` in *other*, collect all ``ref`` labels seen on
    the same points (excluding noise, -1, on either side), then pick the most
    frequent ``ref`` label as the canonical mapping for ``o``.  Finally apply
    the mapping element-wise; unmapped ids (e.g. noise) are passed through.

    Parameters
    ----------
    ref : array-like of shape (n_samples,)
        Reference cluster labels to align to.
    other : array-like of shape (n_samples,)
        Cluster labels to align.

    Returns
    -------
    aligned : ndarray of shape (n_samples,)
        The aligned labels from *other*, remapped to best match *ref*.
    """
    # Build {other_id: [ref_ids seen on co-located points]} ignoring noise
    mapping = {}
    for r, o in zip(ref, other):
        if r == -1 or o == -1:
            continue
        mapping.setdefault(o, []).append(r)
    # For each other cluster, pick the plurality ref label
    remap = {o: max(set(vs), key=vs.count) for o, vs in mapping.items()}
    # Apply remapping; ids absent from remap (noise -1) pass through unchanged
    return np.array([remap.get(v, v) for v in other])


def _assert_label_parity(X, G, msg_ctx="", **hdbscan_kwargs):
    """
    Fit HDBSCAN with both metric='euclidean' on X and metric='precomputed' on G,
    then assert identical cluster structure and label assignments.

    Parameters
    ----------
    X : ndarray, feature matrix used for the euclidean fit
    G : sparse matrix, full pairwise distance graph of X
    msg_ctx : str, optional label appended to failure messages for context
    **hdbscan_kwargs : forwarded to both HDBSCAN constructors (metric excluded)

    Returns
    -------
    eu, pre : fitted HDBSCAN instances for further inspection
    """
    eu = HDBSCAN(**hdbscan_kwargs).fit(X)
    pre = HDBSCAN(**hdbscan_kwargs, metric="precomputed").fit(G)

    eu_n = len(set(eu.labels_) - {-1})
    pre_n = len(set(pre.labels_) - {-1})
    assert (
        eu_n == pre_n
    ), f"Cluster count mismatch {msg_ctx}: euclidean={eu_n}, precomputed={pre_n}"

    aligned = _align_labels(eu.labels_, pre.labels_)
    assert np.array_equal(eu.labels_, aligned), (
        f"Label assignment mismatch {msg_ctx}: "
        f"euclidean={np.unique(eu.labels_, return_counts=True)}, "
        f"precomputed={np.unique(pre.labels_, return_counts=True)}"
    )
    return eu, pre


# ── Module-level parity datasets ──────────────────────────────────────────────
# Two structurally different datasets probe different aspects of the algorithm:
#   blobs  – well-separated isotropic Gaussians; baseline correctness
#   moons  – non-convex crescents; exercises density-path connectivity
_X_blobs, _ = make_blobs(
    n_samples=60,
    centers=[[-4, -4], [0, 0], [4, 4]],
    cluster_std=0.3,
    random_state=0,
)
_X_blobs = StandardScaler().fit_transform(_X_blobs)

_X_moons, _ = datasets.make_moons(n_samples=120, noise=0.05, random_state=0)
_X_moons = StandardScaler().fit_transform(_X_moons)

_PARITY_DATASETS = {
    "blobs": (_X_blobs, _make_full_pairwise_sparse(_X_blobs)),
    "moons": (_X_moons, _make_full_pairwise_sparse(_X_moons)),
}


# ── Parametrised parity tests ─────────────────────────────────────────────────


@pytest.mark.parametrize("dataset", ["blobs", "moons"])
@pytest.mark.parametrize("min_samples", [1, 5, 10])
def test_parity_labels_min_samples(dataset, min_samples):
    """Exact label parity for multiple min_samples values on two dataset types."""
    X, G = _PARITY_DATASETS[dataset]
    _assert_label_parity(
        X,
        G,
        msg_ctx=f"dataset={dataset} min_samples={min_samples}",
        min_cluster_size=10,
        min_samples=min_samples,
    )


@pytest.mark.parametrize("dataset", ["blobs", "moons"])
@pytest.mark.parametrize("mcs", [3, 5, 10, 15])
def test_parity_min_cluster_size_sweep(dataset, mcs):
    """Exact label parity across min_cluster_size values on two dataset types."""
    X, G = _PARITY_DATASETS[dataset]
    _assert_label_parity(
        X,
        G,
        msg_ctx=f"dataset={dataset} min_cluster_size={mcs}",
        min_cluster_size=mcs,
    )


@pytest.mark.parametrize("dataset", ["blobs", "moons"])
def test_parity_no_clusters(dataset):
    """Both paths return all-noise when min_cluster_size > n_samples."""
    X, G = _PARITY_DATASETS[dataset]
    n = X.shape[0]
    eu = HDBSCAN(min_cluster_size=n + 1).fit(X)
    pre = HDBSCAN(min_cluster_size=n + 1, metric="precomputed").fit(G)
    assert np.all(eu.labels_ == -1), "euclidean should be all noise"
    assert np.all(pre.labels_ == -1), "precomputed should be all noise"


@pytest.mark.parametrize("dataset", ["blobs", "moons"])
def test_parity_allow_single_cluster(dataset):
    """allow_single_cluster=True produces identical results on both paths."""
    X, G = _PARITY_DATASETS[dataset]
    _assert_label_parity(
        X,
        G,
        msg_ctx=f"dataset={dataset}",
        min_cluster_size=5,
        allow_single_cluster=True,
    )


def test_parity_leaf_selection_method():
    """
    cluster_selection_method='leaf' parity on blobs.

    Moons is excluded: leaf selection finds many micro-clusters whose boundaries
    are defined by individual MST edge weights.  The float32 rdist (euclidean
    path) vs float64 cdist (precomputed path) precision gap can shift a handful
    of borderline points across cluster boundaries on non-convex data.  On
    well-separated blobs the gap is far below the inter-cluster distance, so
    leaf assignments are unambiguous and exact parity is expected.
    """
    X, G = _PARITY_DATASETS["blobs"]
    _assert_label_parity(
        X,
        G,
        msg_ctx="dataset=blobs",
        min_cluster_size=5,
        cluster_selection_method="leaf",
    )


def test_parity_mst_edge_count_and_approx_weights():
    """
    Both MSTs must have n-1 edges; sorted weights must be close.

    Exact bit-for-bit equality is not expected:
    - Euclidean: float32 rdist → np.sqrt → float32 edge weights.
    - Precomputed: float64 cdist → float32 CoreGraph weights.
    Tolerance rtol=1e-3 / atol=1e-4 covers the expected float32 epsilon gap.
    """
    X, G = _PARITY_DATASETS["blobs"]
    eu, pre = _assert_label_parity(X, G, min_cluster_size=5, min_samples=5)

    n = X.shape[0]
    # Cast via np.asarray so Pyright sees ndarray (not the ambiguous inferred type
    # of the private attribute, which is set from a statically un-narrowable slice).
    eu_mst = np.asarray(eu._min_spanning_tree)
    pre_mst = np.asarray(pre._min_spanning_tree)
    assert eu_mst.shape == (n - 1, 3)
    assert pre_mst.shape == (n - 1, 3)

    np.testing.assert_allclose(
        np.sort(eu_mst[:, 2]),
        np.sort(pre_mst[:, 2]),
        rtol=1e-3,
        atol=1e-4,
        err_msg="MST sorted weights differ beyond float32 epsilon",
    )


def test_parity_probabilities_match():
    """
    Membership probabilities must be approximately equal on matched non-noise points.

    Tolerance rtol=1e-2 / atol=1e-3 accommodates the float32 vs float64
    precision difference in the underlying MST edge weights that feed into
    the condensed tree lambda values.
    """
    X, G = _PARITY_DATASETS["blobs"]
    eu, pre = _assert_label_parity(X, G, min_cluster_size=10)

    aligned = _align_labels(eu.labels_, pre.labels_)
    matched = (eu.labels_ != -1) & (pre.labels_ != -1) & (eu.labels_ == aligned)
    assert matched.sum() > 0, "No matched non-noise points to compare"
    np.testing.assert_allclose(
        eu.probabilities_[matched],
        pre.probabilities_[matched],
        rtol=1e-2,
        atol=1e-3,
        err_msg="Membership probabilities differ beyond tolerance",
    )


def test_parity_function_api_matches_class_api():
    """fast_hdbscan() function and HDBSCAN class must return identical results."""
    _, G = _PARITY_DATASETS["blobs"]
    pre_class = HDBSCAN(min_cluster_size=10, metric="precomputed").fit(G)
    # fast_hdbscan returns a sliced tuple whose length Pyright cannot narrow statically;
    # index explicitly to avoid the tuple-size type error.
    result = fast_hdbscan(G, min_cluster_size=10, metric="precomputed")
    fn_labels, fn_probs = result[0], result[1]
    assert np.array_equal(pre_class.labels_, fn_labels)
    np.testing.assert_allclose(pre_class.probabilities_, fn_probs, rtol=1e-6)


@pytest.mark.parametrize("dataset", ["blobs", "moons"])
def test_parity_epsilon_selection(dataset):
    """cluster_selection_epsilon threshold is applied identically on both paths."""
    X, G = _PARITY_DATASETS[dataset]
    _assert_label_parity(
        X,
        G,
        msg_ctx=f"dataset={dataset}",
        min_cluster_size=5,
        cluster_selection_epsilon=0.5,
    )
