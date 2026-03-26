import numpy as np
import numba
import numbers

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import (
    check_is_fitted,
    check_non_negative,
    validate_data,
)
from sklearn.neighbors import KDTree

from warnings import warn
from typing import Optional

from .numba_kdtree import kdtree_to_numba, build_kdtree
from .boruvka import parallel_boruvka
from .cluster_trees import (
    mst_to_linkage_tree,
    mst_to_linkage_tree_w_sample_weights,
    condense_tree,
    simplify_hierarchy,
    extract_eom_clusters,
    cluster_tree_leaves,
    cluster_epsilon_search,
    get_cluster_labelling_at_cut,
    get_cluster_label_vector,
    get_point_membership_strength_vector,
    cluster_tree_from_condensed_tree,
    extract_clusters_bcubed,
)
from .layer_clusters import build_cluster_layers, build_layer_cluster_tree

try:
    from hdbscan.plots import CondensedTree, SingleLinkageTree, MinimumSpanningTree

    _HAVE_HDBSCAN = True
except ImportError:
    _HAVE_HDBSCAN = False


# Simplified version of sklearn's check_sample_weight
def _check_sample_weight(
    sample_weight,
    X,
    dtype,
    *,
    ensure_non_negative=False,
    copy=False,
    allow_all_zero_weights=False,
):
    n_samples = X.shape[0]

    if isinstance(sample_weight, numbers.Number):
        sample_weight = np.full(n_samples, sample_weight, dtype=dtype)
    else:
        sample_weight = check_array(
            sample_weight,
            accept_sparse=False,
            ensure_2d=False,
            dtype=dtype,
            order="C",
            copy=copy,
            input_name="sample_weight",
        )
        if sample_weight.ndim != 1:
            raise ValueError(
                f"Sample weights must be 1D array or scalar, got "
                f"{sample_weight.ndim}D array. Expected either a scalar value "
                f"or a 1D array of length {n_samples}."
            )

        if sample_weight.shape != (n_samples,):
            raise ValueError(
                "sample_weight.shape == {}, expected {}!".format(
                    sample_weight.shape, (n_samples,)
                )
            )

    if not allow_all_zero_weights:
        if np.all(sample_weight == 0):
            raise ValueError(
                "Sample weights must contain at least one non-zero number."
            )

    if ensure_non_negative:
        check_non_negative(sample_weight, "`sample_weight`")

    return sample_weight


def to_numpy_rec_array(named_tuple_tree):
    size = named_tuple_tree.parent.shape[0]
    result = np.empty(
        size,
        dtype=[
            ("parent", np.intp),
            ("child", np.intp),
            ("lambda_val", float),
            ("child_size", np.float32),
        ],
    )

    result["parent"] = named_tuple_tree.parent
    result["child"] = named_tuple_tree.child
    result["lambda_val"] = named_tuple_tree.lambda_val
    result["child_size"] = named_tuple_tree.child_size

    return result


def remap_condensed_tree(tree, internal_to_raw, outliers):
    """
    Takes an internal condensed_tree structure and adds back in a set of points
    that were initially detected as non-finite and returns that new tree.
    These points will all be split off from the maximal node at lambda zero and
    considered noise points.
    Parameters
    ----------
    tree: condensed_tree
    internal_to_raw: dict
        a mapping from internal integer index to the raw integer index
    finite_index: ndarray
        Boolean array of which entries in the raw data were finite
    """
    finite_count = len(internal_to_raw)

    outlier_count = len(outliers)
    for i, (parent, child, lambda_val, child_size) in enumerate(tree):
        if child < finite_count:
            child = internal_to_raw[child]
        else:
            child = child + outlier_count
        tree[i] = (parent + outlier_count, child, lambda_val, child_size)

    outlier_list = []
    root = tree[0][0]  # Should I check to be sure this is the minimal lambda?
    for outlier in outliers:
        outlier_list.append((root, outlier, 0, 1))

    outlier_tree = np.array(
        outlier_list,
        dtype=[
            ("parent", np.intp),
            ("child", np.intp),
            ("lambda_val", float),
            ("child_size", np.intp),
        ],
    )
    tree = np.append(outlier_tree, tree)
    return tree


def remap_single_linkage_tree(tree, internal_to_raw, outliers):
    """
    Takes an internal single_linkage_tree structure and adds back in a set of points
    that were initially detected as non-finite and returns that new tree.
    These points will all be merged into the final node at np.inf distance and
    considered noise points.
    Parameters
    ----------
    tree: single_linkage_tree
    internal_to_raw: dict
        a mapping from internal integer index to the raw integer index
    finite_index: ndarray
        Boolean array of which entries in the raw data were finite
    """
    finite_count = len(internal_to_raw)

    outlier_count = len(outliers)
    for i, (left, right, distance, size) in enumerate(tree):
        if left < finite_count:
            tree[i, 0] = internal_to_raw[left]
        else:
            tree[i, 0] = left + outlier_count
        if right < finite_count:
            tree[i, 1] = internal_to_raw[right]
        else:
            tree[i, 1] = right + outlier_count

    outlier_tree = np.zeros((len(outliers), 4))
    last_cluster_id = tree[tree.shape[0] - 1][0:2].max()
    last_cluster_size = tree[tree.shape[0] - 1][3]
    for i, outlier in enumerate(outliers):
        outlier_tree[i] = (outlier, last_cluster_id + 1, np.inf, last_cluster_size + 1)
        last_cluster_id += 1
        last_cluster_size += 1
    tree = np.vstack([tree, outlier_tree])
    return tree


def fast_hdbscan(
    data,
    data_labels=None,
    semi_supervised=False,
    ss_algorithm="bc",
    min_samples=10,
    min_cluster_size=10,
    cluster_selection_method="eom",
    max_cluster_size=np.inf,
    allow_single_cluster=False,
    cluster_selection_epsilon=0.0,
    cluster_selection_persistence=0.0,
    sample_weights=None,
    reproducible=False,
    return_trees=False,
    metric="euclidean",
    algorithm="boruvka",
    knn_k=None,
    cannot_link=None,
    validate_cannot_link=True,
    cannot_link_groups=None,
    metric_kwds=None,
):
    """
    Run the full HDBSCAN pipeline: MST construction, condensed tree
    extraction, and cluster selection.
    """
    if metric == "precomputed":
        if sample_weights is not None:
            raise NotImplementedError(
                "sample_weights is not supported with metric='precomputed'."
            )
        # Validation is deferred to compute_minimum_spanning_tree -> precomputed module
    elif metric == "euclidean":
        data = check_array(data)
        if sample_weights is not None:
            sample_weights = _check_sample_weight(
                sample_weights, data, dtype=np.float32
            )
    else:
        # Arbitrary metric — requires pynndescent
        from .nndescent import _check_pynndescent_available

        _check_pynndescent_available()
        data = check_array(data)

    # Detect parameter inconsistencies early.
    if semi_supervised:
        if data_labels is None:
            raise ValueError(
                "data_labels must not be None when semi_supervised is set to True!"
            )
        if ss_algorithm not in ["bc", "bc_simple"]:
            raise ValueError(f"Invalid ss_algorithm {ss_algorithm}")

    if (
        (not (np.issubdtype(type(min_samples), np.integer) or min_samples is None))
        or not np.issubdtype(type(min_cluster_size), np.integer)
        or (min_samples is not None and min_samples <= 0)
        or min_cluster_size <= 0
    ):
        raise ValueError("Min samples and min cluster size must be positive integers!")

    if (
        not np.issubdtype(type(cluster_selection_epsilon), np.floating)
        or cluster_selection_epsilon < 0.0
    ):
        raise ValueError(
            "Cluster selection epsilon must be a positive floating point number!"
        )
    if (
        not np.issubdtype(type(cluster_selection_persistence), np.floating)
        or cluster_selection_persistence < 0.0
    ):
        raise ValueError(
            "Cluster selection persistence must be a positive floating point number!"
        )

    minimum_spanning_tree, neighbors, core_distances = compute_minimum_spanning_tree(
        data,
        min_samples=min_cluster_size if min_samples is None else min_samples,
        sample_weights=sample_weights,
        reproducible=reproducible,
        metric=metric,
        algorithm=algorithm,
        knn_k=knn_k,
        cannot_link=cannot_link,
        validate_cannot_link=validate_cannot_link,
        cannot_link_groups=cannot_link_groups,
        metric_kwds=metric_kwds,
    )

    return (
        *clusters_from_spanning_tree(
            minimum_spanning_tree,
            data_labels=data_labels,
            semi_supervised=semi_supervised,
            ss_algorithm=ss_algorithm,
            min_cluster_size=min_cluster_size,
            cluster_selection_method=cluster_selection_method,
            max_cluster_size=max_cluster_size,
            allow_single_cluster=allow_single_cluster,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_persistence=cluster_selection_persistence,
            sample_weights=sample_weights,
        ),
        neighbors,
        core_distances,
    )[: (None if return_trees else 2)]


def compute_minimum_spanning_tree(
    data,
    min_samples=10,
    sample_weights=None,
    reproducible=False,
    metric="euclidean",
    algorithm="boruvka",
    knn_k=None,
    cannot_link=None,
    validate_cannot_link=True,
    cannot_link_groups=None,
    metric_kwds=None,
):
    """
    Compute the minimum spanning tree for HDBSCAN.

    Parameters
    ----------
    data : array-like or scipy sparse matrix
        Feature matrix (metric='euclidean') or pairwise weight graph
        (metric='precomputed'). For arbitrary metrics, a dense feature matrix.
    min_samples : int
    sample_weights : array-like or None
        Not supported when metric='precomputed'.
    reproducible : bool
    metric : str
        'euclidean' (default), 'precomputed', or any metric supported by
        pynndescent (e.g. 'cosine', 'manhattan', 'minkowski', etc.).
    algorithm : str
        MST algorithm to use: 'boruvka' (default) or 'kruskal'.
        - 'boruvka' : parallel Borůvka via KD-tree for euclidean; Borůvka on
          CoreGraph (float32) for precomputed / pynndescent.
        - 'kruskal' : Kruskal DSU on KNN graph for euclidean; Kruskal DSU on
          full CSR edge list (float64) for precomputed / pynndescent.
    knn_k : int or None
        Number of neighbors for KNN graph. Used when algorithm='kruskal' and
        metric='euclidean', or for any pynndescent-backed metric.
        - None : for euclidean/kruskal, exact MST via full pairwise distances;
          for pynndescent metrics, defaults to max(3 * min_samples, 15).
        - int  : approximate MST via KNN subgraph with this many neighbors.
    cannot_link : scipy sparse matrix or None
        Symmetric sparse matrix of cannot-link constraints.  Entry (i, j) != 0
        means samples i and j must not co-cluster.  Only supported with
        ``algorithm='kruskal'``.  Mutually exclusive with
        ``cannot_link_groups``.
    validate_cannot_link : bool
        If True (default), validate and symmetrize the cannot-link matrix
        (handles upper-triangle-only and lower-triangle-only inputs).
        Set to False to skip validation when you know the input is already
        a symmetric CSR matrix — avoids an O(nnz) symmetrization step.
    cannot_link_groups : array-like of int or None
        Group-label cannot-link constraints.  An ``int32[n_samples]`` array
        where samples sharing the same non-negative label cannot co-cluster.
        ``-1`` means unconstrained.  This is an O(n) alternative to the
        O(n * k^2) sparse ``cannot_link`` matrix for block-diagonal
        constraint structures (e.g., same-session, same-batch).  Only
        supported with ``algorithm='kruskal'``.  Mutually exclusive with
        ``cannot_link``.
    metric_kwds : dict or None
        Additional keyword arguments for the distance metric (pynndescent only).

    Returns
    -------
    mst_edges : float64[:, :], shape (n_samples - 1, 3)
        Minimum spanning tree edges, columns [src, dst, mrd_weight].
    neighbors : int32[:, :], shape (n_samples, k)
        Nearest-neighbor indices for each sample.
    core_distances : float64[:], shape (n_samples,)
        Core distance for each sample.
    """
    if algorithm not in ("boruvka", "kruskal"):
        raise ValueError(
            "algorithm must be 'boruvka' or 'kruskal'. Got: %s" % algorithm
        )

    if cannot_link is not None and cannot_link_groups is not None:
        raise ValueError(
            "cannot_link and cannot_link_groups are mutually exclusive. "
            "Provide one or the other, not both."
        )

    _has_cl = cannot_link is not None or cannot_link_groups is not None
    if _has_cl and algorithm != "kruskal":
        raise ValueError(
            "cannot_link constraints are only supported with "
            "algorithm='kruskal'. Got algorithm=%r." % algorithm
        )

    if metric == "precomputed":
        if sample_weights is not None:
            raise NotImplementedError(
                "sample_weights is not supported with metric='precomputed'."
            )
        if algorithm == "kruskal":
            from .precomputed import compute_mst_from_precomputed_sparse_kruskal

            return compute_mst_from_precomputed_sparse_kruskal(
                data,
                min_samples,
                cannot_link=cannot_link,
                validate_cannot_link=validate_cannot_link,
                cannot_link_groups=cannot_link_groups,
            )
        else:
            from .precomputed import compute_mst_from_precomputed_sparse

            return compute_mst_from_precomputed_sparse(data, min_samples)

    if metric not in ("euclidean", "precomputed") or data.shape[1] > 30:
        # Arbitrary metric — delegate to pynndescent KNN graph path
        from .nndescent import compute_mst_from_knn_graph

        return compute_mst_from_knn_graph(
            data,
            min_samples=min_samples,
            metric=metric,
            metric_kwds=metric_kwds,
            knn_k=knn_k,
        )

    # metric == "euclidean"
    numba_tree = build_kdtree(data)

    if algorithm == "kruskal":
        from .kruskal import kruskal_mst_from_feature_matrix

        return kruskal_mst_from_feature_matrix(
            numba_tree,
            min_samples,
            knn_k=knn_k,
            sample_weights=sample_weights,
            reproducible=reproducible,
            cannot_link=cannot_link,
            validate_cannot_link=validate_cannot_link,
            cannot_link_groups=cannot_link_groups,
        )
    else:
        n_threads = numba.get_num_threads()
        edges, neighbors, core_distances = parallel_boruvka(
            numba_tree,
            n_threads,
            min_samples=min_samples,
            sample_weights=(
                sample_weights
                if sample_weights is not None
                else np.empty(1, dtype=np.float32)
            ),
            reproducible=reproducible,
        )
        return edges, neighbors, core_distances


def clusters_from_spanning_tree(
    minimum_spanning_tree,
    data_labels=None,
    semi_supervised=False,
    ss_algorithm="bc",
    min_cluster_size=10,
    cluster_selection_method="eom",
    max_cluster_size=np.inf,
    allow_single_cluster=False,
    cluster_selection_epsilon=0.0,
    cluster_selection_persistence=0.0,
    sample_weights=None,
):
    n_points = minimum_spanning_tree.shape[0] + 1
    sorted_mst = minimum_spanning_tree[
        np.lexsort(
            (
                minimum_spanning_tree.T[1],
                minimum_spanning_tree.T[0],
                minimum_spanning_tree.T[2],
            )
        )
    ]

    if sample_weights is None:
        linkage_tree = mst_to_linkage_tree(sorted_mst)
    else:
        linkage_tree = mst_to_linkage_tree_w_sample_weights(sorted_mst, sample_weights)

    condensed_tree = condense_tree(
        linkage_tree, min_cluster_size=min_cluster_size, sample_weights=sample_weights
    )
    if cluster_selection_persistence > 0.0 and len(condensed_tree.parent) > 0:
        condensed_tree = simplify_hierarchy(
            condensed_tree, cluster_selection_persistence
        )

    cluster_tree = cluster_tree_from_condensed_tree(condensed_tree)
    if cluster_selection_method == "eom":
        if semi_supervised:
            # Assumes ss_algorithm is either 'bc' or 'bc_simple'
            selected_clusters = extract_clusters_bcubed(
                condensed_tree,
                cluster_tree,
                data_labels,
                allow_virtual_nodes=True if ss_algorithm == "bc" else False,
                allow_single_cluster=allow_single_cluster,
            )
        else:
            selected_clusters = extract_eom_clusters(
                condensed_tree,
                cluster_tree,
                max_cluster_size=max_cluster_size,
                allow_single_cluster=allow_single_cluster,
            )
    elif cluster_selection_method == "leaf":
        if cluster_tree.parent.shape[0] == 0:
            selected_clusters = np.empty(0, dtype=np.int64)
        else:
            selected_clusters = cluster_tree_leaves(cluster_tree, n_points)
    else:
        raise ValueError(f"Invalid cluster_selection_method {cluster_selection_method}")

    if len(selected_clusters) > 1 and cluster_selection_epsilon > 0.0:
        selected_clusters = cluster_epsilon_search(
            selected_clusters,
            cluster_tree,
            min_epsilon=cluster_selection_epsilon,
        )

    clusters = get_cluster_label_vector(
        condensed_tree, selected_clusters, cluster_selection_epsilon, n_samples=n_points
    )
    membership_strengths = get_point_membership_strength_vector(
        condensed_tree, selected_clusters, clusters
    )

    return clusters, membership_strengths, linkage_tree, condensed_tree, sorted_mst


class HDBSCAN(ClusterMixin, BaseEstimator):
    """
    Hierarchical Density-Based Spatial Clustering (HDBSCAN).

    Scikit-learn compatible estimator wrapping :func:`fast_hdbscan`.

    Cannot-link constraints
    -----------------------
    Two mutually exclusive modes are supported (both require
    ``algorithm='kruskal'``):

    * **Pairwise** (``cannot_link``): an ``(n, n)`` sparse boolean matrix
      where entry ``(i, j) != 0`` forbids samples *i* and *j* from
      co-clustering.  Supports arbitrary constraint structures.
    * **Group-label** (``cannot_link_groups``): an ``int32[n]`` array where
      samples sharing the same non-negative label cannot co-cluster
      (``-1`` = unconstrained).  O(n) memory and O(1) per-merge conflict
      check via bitmask, ideal for block-diagonal constraints
      (same-session / same-batch / same-replicate).

    Parameters
    ----------
    min_cluster_size : int
        Minimum number of samples in a cluster.
    min_samples : int or None
        Number of neighbors for core distance.  Defaults to
        ``min_cluster_size`` if None.
    cluster_selection_method : str
        ``'eom'`` (excess of mass) or ``'leaf'``.
    allow_single_cluster : bool
        If True, allow returning a single cluster.
    max_cluster_size : float
        Upper bound on cluster size (EOM only).
    cluster_selection_epsilon : float
        DBSCAN-style epsilon threshold for cluster selection.
    cluster_selection_persistence : float
        Minimum persistence for a cluster to be selected.
    semi_supervised : bool
        Enable semi-supervised clustering.
    ss_algorithm : str
        Semi-supervised algorithm: ``'bc'`` or ``'bc_simple'``.
    reproducible : bool
        If True, use deterministic algorithms (Borůvka only; Kruskal
        is always deterministic).
    metric : str
        ``'euclidean'``, ``'precomputed'``, or any pynndescent metric.
    algorithm : str
        ``'boruvka'`` or ``'kruskal'``.
    knn_k : int or None
        Number of nearest neighbors for KNN graph (Kruskal only).
    cannot_link : scipy sparse matrix or None
        Pairwise cannot-link constraint matrix.  Mutually exclusive with
        ``cannot_link_groups``.
    validate_cannot_link : bool
        If True, validate and symmetrize the CL matrix.
    cannot_link_groups : array-like of int or None
        Group-label cannot-link constraints (``int32[n]``, ``-1`` =
        unconstrained).  Mutually exclusive with ``cannot_link``.
    metric_kwds : dict or None
        Keyword arguments for the distance metric (pynndescent only).

    Attributes
    ----------
    labels_ : int32[:], shape (n_samples,)
        Cluster labels.  ``-1`` indicates noise.
    probabilities_ : float64[:], shape (n_samples,)
        Cluster membership strengths in [0, 1].
    """

    def __init__(
        self,
        *,
        min_cluster_size=5,
        min_samples=None,
        cluster_selection_method="eom",
        allow_single_cluster=False,
        max_cluster_size=np.inf,
        cluster_selection_epsilon=0.0,
        cluster_selection_persistence=0.0,
        semi_supervised=False,
        ss_algorithm="bc",
        reproducible=False,
        metric="euclidean",
        algorithm="boruvka",
        knn_k=None,
        cannot_link=None,
        validate_cannot_link=True,
        cannot_link_groups=None,
        metric_kwds=None,
        # Removed **kwargs to comply with scikit-learn's API requirements
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.max_cluster_size = max_cluster_size
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_persistence = cluster_selection_persistence
        self.semi_supervised = semi_supervised
        self.ss_algorithm = ss_algorithm
        self.reproducible = reproducible
        self.metric = metric
        self.algorithm = algorithm
        self.knn_k = knn_k
        self.cannot_link = cannot_link
        self.validate_cannot_link = validate_cannot_link
        self.cannot_link_groups = cannot_link_groups
        self.metric_kwds = metric_kwds

    def fit(self, X, y=None, sample_weight=None, **fit_params):

        if self.metric == "precomputed":
            # Precomputed sparse graph path: skip dense feature-matrix validations.
            from .precomputed import validate_precomputed_sparse_graph

            validate_precomputed_sparse_graph(X)
            if sample_weight is not None:
                raise NotImplementedError(
                    "sample_weights is not supported with metric='precomputed'."
                )
            self._raw_data = None
            clean_data = X

            if self.semi_supervised:
                # In precomputed mode, labels still correspond 1:1 with graph rows.
                # Mirror the euclidean semi-supervised behavior by validating labels
                # with scikit-learn utilities and requiring at least one supervised
                # point.
                if y is None:
                    raise ValueError(
                        "y must not be None when semi_supervised is set to True!"
                    )

                clean_data_labels = check_array(
                    y,
                    ensure_2d=False,
                ).copy()

                if clean_data_labels.ndim != 1:
                    clean_data_labels = np.ravel(clean_data_labels)
                if clean_data_labels.shape[0] != X.shape[0]:
                    raise ValueError(
                        "y must contain exactly one label per node when "
                        "metric='precomputed'. "
                        f"Got {clean_data_labels.shape[0]} labels for {X.shape[0]} nodes."
                    )

                self._raw_labels = clean_data_labels.copy()
                if ~np.any(clean_data_labels != -1):
                    raise ValueError(
                        "y must contain at least one label > -1. Currently it only "
                        "contains -1 labels!"
                    )
            else:
                clean_data_labels = None
            self._all_finite = True  # no per-row finite filtering needed
        elif self.semi_supervised:
            X, y = check_X_y(X, y, accept_sparse="csr", ensure_all_finite=False)
            self._raw_data = X
            self._raw_labels = y
            # Replace non-finite labels with -1 labels
            y[~np.isfinite(y)] = -1

            if ~np.any(y != -1):
                raise ValueError(
                    "y must contain at least one label > -1. Currently it only contains -1 and/or non-finite labels!"
                )
            if sample_weight is not None:
                sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float32)

            self._all_finite = np.all(np.isfinite(X))
            if ~self._all_finite:
                finite_index = np.where(np.isfinite(X).sum(axis=1) == X.shape[1])[0]
                clean_data = X[finite_index]
                clean_data_labels = y[finite_index]
                sample_weight = (
                    sample_weight[finite_index] if sample_weight is not None else None
                )
                internal_to_raw = {
                    x: y for x, y in zip(range(len(finite_index)), finite_index)
                }
                outliers = list(set(range(X.shape[0])) - set(finite_index))
            else:
                clean_data = X
                clean_data_labels = y
        else:
            X = validate_data(self, X, accept_sparse="csr", ensure_all_finite=False)
            self._raw_data = X
            if sample_weight is not None:
                sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float32)

            self._all_finite = np.all(np.isfinite(X))
            if ~self._all_finite:
                # Pass only the purely finite indices into hdbscan
                # We will later assign all non-finite points to the background -1 cluster
                finite_index = np.where(np.isfinite(X).sum(axis=1) == X.shape[1])[0]
                clean_data = X[finite_index]
                clean_data_labels = None
                sample_weight = (
                    sample_weight[finite_index] if sample_weight is not None else None
                )
                internal_to_raw = {
                    x: y for x, y in zip(range(len(finite_index)), finite_index)
                }
                outliers = list(set(range(X.shape[0])) - set(finite_index))
            else:
                clean_data = X
                clean_data_labels = y

        kwargs = self.get_params()

        (
            self.labels_,
            self.probabilities_,
            self._single_linkage_tree,
            self._condensed_tree,
            self._min_spanning_tree,
            self._neighbors,
            self._core_distances,
        ) = fast_hdbscan(
            clean_data,
            clean_data_labels,
            return_trees=True,
            sample_weights=sample_weight,
            **kwargs,
        )

        self._condensed_tree = to_numpy_rec_array(self._condensed_tree)

        if not self._all_finite:
            # remap indices to align with original data in the case of non-finite entries.
            self._condensed_tree = remap_condensed_tree(
                self._condensed_tree, internal_to_raw, outliers
            )
            self._single_linkage_tree = remap_single_linkage_tree(
                self._single_linkage_tree, internal_to_raw, outliers
            )
            new_labels = np.full(X.shape[0], -1)
            new_labels[finite_index] = self.labels_
            self.labels_ = new_labels

            new_probabilities = np.zeros(X.shape[0])
            new_probabilities[finite_index] = self.probabilities_
            self.probabilities_ = new_probabilities

        return self

    def dbscan_clustering(self, epsilon):
        check_is_fitted(
            self,
            "_single_linkage_tree",
            msg="You first need to fit the HDBSCAN model before picking a DBSCAN clustering",
        )
        return get_cluster_labelling_at_cut(
            self._single_linkage_tree,
            epsilon,
            self.min_samples if self.min_samples is not None else self.min_cluster_size,
        )

    @property
    def condensed_tree_(self):
        check_is_fitted(
            self,
            "_condensed_tree",
            msg="You first need to fit the HDBSCAN model before accessing the condensed tree",
        )
        if self._condensed_tree is not None:
            return CondensedTree(
                self._condensed_tree,
                self.labels_,
            )
        else:
            raise AttributeError(
                "No condensed tree was generated; try running fit first."
            )

    @property
    def single_linkage_tree_(self):
        check_is_fitted(
            self,
            "_single_linkage_tree",
            msg="You first need to fit the HDBSCAN model before accessing the single linkage tree",
        )
        if self._single_linkage_tree is not None:
            return SingleLinkageTree(self._single_linkage_tree)
        else:
            raise AttributeError(
                "No single linkage tree was generated; try running fit first."
            )

    @property
    def minimum_spanning_tree_(self):
        check_is_fitted(
            self,
            "_min_spanning_tree",
            msg="You first need to fit the HDBSCAN model before accessing the minimum spanning tree",
        )
        if self._min_spanning_tree is not None:
            if self._raw_data is not None:
                return MinimumSpanningTree(self._min_spanning_tree, self._raw_data)
            else:
                warn(
                    "No raw data is available; this may be due to using"
                    " a precomputed metric matrix. No minimum spanning"
                    " tree will be provided without raw data."
                )
                return None
        else:
            raise AttributeError(
                "No minimum spanning tree was generated; try running fit first."
            )


class PLSCAN(ClusterMixin, BaseEstimator):
    """Perform PLSCAN clustering for automated multi-resolution cluster analysis.

    PLSCAN automatically discovers meaningful clustering resolutions from data
    by analysing the persistence landscape of the HDBSCAN cluster hierarchy.
    It produces multiple *cluster layers*, each representing the data at a
    different resolution, and selects the most persistent layer as the primary
    labelling.

    See `Persistent Multiscale Density-based Clustering <https://arxiv.org/abs/2512.16558>`_ for algorithmic details.

    Parameters
    ----------
    min_samples : int, default=5
        The number of samples in a neighbourhood for a point to be considered
        a core point.  This controls the level of smoothing applied to the
        density estimate.

    max_layers : int, default=10
        Maximum number of cluster resolution layers to return (including the
        base layer).

    base_min_cluster_size : int, default=5
        The minimum cluster size used when constructing the initial condensed
        tree.  Clusters smaller than this are treated as noise.

    base_n_clusters : int or None, default=None
        If set, the base layer is constructed by binary-searching for a
        ``min_cluster_size`` that yields approximately this many clusters.
        When ``None``, ``base_min_cluster_size`` is used directly.

    layer_similarity_threshold : float, default=0.2
        Minimum dissimilarity between selected resolution layers.  Lower
        values produce more diverse layers; higher values allow more similar
        layers to coexist.

    reproducible : bool, default=False
        If ``True``, use a slower but deterministic algorithm for the
        minimum spanning tree computation so that results are reproducible
        across runs.

    metric : str, default='euclidean'
        The distance metric to use.  ``'euclidean'`` uses a fast KD-tree
        implementation.  ``'precomputed'`` expects a sparse distance graph
        as input.  Any other metric supported by ``pynndescent`` may be
        used, but requires ``pynndescent`` to be installed.

    algorithm : str, default='boruvka'
        The MST algorithm to use.  Must be ``'boruvka'`` or ``'kruskal'``.
        ``'kruskal'`` is required when using ``cannot_link`` constraints.

    knn_k : int or None, default=None
        Number of nearest neighbours to compute when building the MST.
        When ``None``, a suitable default is chosen automatically.

    cannot_link : array-like or None, default=None
        An array of pairs ``(i, j)`` of sample indices that must not be
        placed in the same cluster.  Only supported with
        ``algorithm='kruskal'``.

    validate_cannot_link : bool, default=True
        Whether to validate ``cannot_link`` constraints before clustering.

    metric_kwds : dict or None, default=None
        Additional keyword arguments to pass to the distance metric.

    verbose : bool, default=False
        If ``True``, print progress messages during fitting.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset.  Noisy samples are
        given the label -1.  Corresponds to the layer with the highest
        persistence score.

    membership_strengths_ : ndarray of shape (n_samples,)
        Cluster membership strength for each sample in ``labels_``.

    cluster_layers_ : list of ndarray
        List of label arrays, one per discovered resolution layer, sorted
        by number of clusters in descending order.

    membership_strength_layers_ : list of ndarray
        Membership strengths corresponding to each entry in
        ``cluster_layers_``.

    layer_persistence_scores_ : list of float
        Persistence score for each layer.  The layer with the highest score
        is used as the primary labelling.

    min_cluster_sizes_ : ndarray
        The ``min_cluster_size`` values explored during the persistence
        landscape analysis.

    total_persistence_ : ndarray
        Total persistence values across the ``min_cluster_size`` range.

    cluster_tree_ : dict
        Hierarchical tree relating clusters across layers.  Keys are
        ``(layer, cluster_id)`` tuples representing parent clusters; values
        are lists of ``(layer, cluster_id)`` child tuples.

    Examples
    --------
    >>> import fast_hdbscan
    >>> from sklearn.datasets import make_blobs
    >>> data, _ = make_blobs(1000, random_state=42)
    >>> clusterer = fast_hdbscan.PLSCAN()
    >>> labels = clusterer.fit_predict(data)
    >>> len(clusterer.cluster_layers_)  # number of resolution layers found
    1

    References
    ----------
    .. [1] Daniël Bot, Leland McInnes, Jan Aerts (2025). "Persistent Multiscale Density-based Clustering."
       arXiv:2512.16558.
    """

    def __init__(
        self,
        *,
        min_samples: int = 5,
        max_layers: int = 10,
        base_min_cluster_size: int = 5,
        base_n_clusters: Optional[int] = None,
        layer_similarity_threshold: float = 0.2,
        reproducible: bool = False,
        metric: str = "euclidean",
        algorithm: str = "boruvka",
        knn_k: Optional[int] = None,
        cannot_link=None,
        validate_cannot_link: bool = True,
        metric_kwds: Optional[dict] = None,
        verbose=False,
    ):
        self.min_samples = min_samples
        self.max_layers = max_layers
        self.base_min_cluster_size = base_min_cluster_size
        self.base_n_clusters = base_n_clusters
        self.layer_similarity_threshold = layer_similarity_threshold
        self.reproducible = reproducible
        self.metric = metric
        self.algorithm = algorithm
        self.knn_k = knn_k
        self.cannot_link = cannot_link
        self.validate_cannot_link = validate_cannot_link
        self.metric_kwds = metric_kwds
        self.verbose = verbose

        self._validate_params()

    def _validate_params(self):
        if (
            not np.issubdtype(type(self.min_samples), np.integer)
        ) or self.min_samples <= 0:
            raise ValueError("Min samples must be a positive integer!")

        if (
            not np.issubdtype(type(self.max_layers), np.integer)
        ) or self.max_layers <= 0:
            raise ValueError("Max layers must be a positive integer!")

        if (
            not np.issubdtype(type(self.base_min_cluster_size), np.integer)
        ) or self.base_min_cluster_size <= 0:
            raise ValueError("Base min cluster size must be a positive integer!")

        if self.base_n_clusters is not None:
            if (
                not np.issubdtype(type(self.base_n_clusters), np.integer)
            ) or self.base_n_clusters <= 0:
                raise ValueError("Base n clusters must be a positive integer!")

        if self.algorithm not in ("boruvka", "kruskal"):
            raise ValueError(
                "algorithm must be 'boruvka' or 'kruskal'. Got: %s" % self.algorithm
            )

        if self.cannot_link is not None and self.algorithm != "kruskal":
            raise ValueError(
                "cannot_link constraints are only supported with "
                "algorithm='kruskal'. Got algorithm=%r." % self.algorithm
            )

    def fit_predict(self, X, y=None, sample_weight=None, **fit_params):
        """Fit the model and return cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), or sparse graph
            Training data.  When ``metric='precomputed'``, ``X`` should be
            a sparse distance graph.

        y : ignored
            Not used, present for API consistency.

        sample_weight : array-like of shape (n_samples,) or None, default=None
            Weights for each sample.  Not supported with
            ``metric='precomputed'``.

        **fit_params : dict
            Additional fitting parameters (unused).

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels for the layer with the highest persistence
            score.  Noisy samples are labelled -1.
        """
        import scipy.sparse

        if self.metric == "precomputed":
            from .precomputed import validate_precomputed_sparse_graph

            validate_precomputed_sparse_graph(X)
            if sample_weight is not None:
                raise NotImplementedError(
                    "sample_weights is not supported with metric='precomputed'."
                )
            self._raw_data = None
        elif self.metric != "euclidean":
            from .nndescent import _check_pynndescent_available

            _check_pynndescent_available()
            if scipy.sparse.issparse(X):
                raise ValueError(
                    "PLSCAN requires a dense feature matrix for metric=%r. "
                    "Sparse matrices are only supported with "
                    "metric='precomputed'." % self.metric
                )
            X = validate_data(self, X, accept_sparse=False, ensure_all_finite=False)
            self._raw_data = X
            if sample_weight is not None:
                sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float32)
        else:
            if scipy.sparse.issparse(X):
                raise ValueError(
                    "PLSCAN requires a dense feature matrix for metric='euclidean'. "
                    "Sparse matrices are only supported with "
                    "metric='precomputed'."
                )
            X = validate_data(self, X, accept_sparse=False, ensure_all_finite=False)
            self._raw_data = X
            if sample_weight is not None:
                sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float32)

        kwargs = self.get_params()

        (
            self.cluster_layers_,
            self.membership_strength_layers_,
            self.layer_persistence_scores_,
            self.min_cluster_sizes_,
            self.total_persistence_,
        ) = build_cluster_layers(
            X,
            sample_weights=sample_weight,
            **kwargs,
        )
        self.cluster_tree_ = build_layer_cluster_tree(self.cluster_layers_)

        if len(self.cluster_layers_) == 1:
            self.labels_ = self.cluster_layers_[0]
            self.membership_strengths_ = self.membership_strength_layers_[0]
        else:
            best_layer = np.argmax(self.layer_persistence_scores_)
            self.labels_ = self.cluster_layers_[best_layer]
            self.membership_strengths_ = self.membership_strength_layers_[best_layer]

        return self.labels_

    def fit(self, X, y=None, sample_weight=None, **fit_params):
        """Fit the PLSCAN model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), or sparse graph
            Training data.  When ``metric='precomputed'``, ``X`` should be
            a sparse distance graph.

        y : ignored
            Not used, present for API consistency.

        sample_weight : array-like of shape (n_samples,) or None, default=None
            Weights for each sample.  Not supported with
            ``metric='precomputed'``.

        **fit_params : dict
            Additional fitting parameters (unused).

        Returns
        -------
        self : PLSCAN
            The fitted estimator.
        """
        _ = self.fit_predict(X, y=y, sample_weight=sample_weight, **fit_params)
        return self
