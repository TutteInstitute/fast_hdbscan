from __future__ import annotations

import numpy as np
import numba

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import (
    check_is_fitted,
    _check_sample_weight,
    validate_data,
)
from sklearn.neighbors import KDTree

from warnings import warn
from typing import Optional, Union, Literal, Any
import numpy.typing as npt

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
):
    """Perform HDBSCAN clustering on data.

    This is the low-level functional interface for HDBSCAN clustering.
    For a sklearn-compatible estimator interface, use the :class:`HDBSCAN` class.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The input data to cluster. Must contain finite values.

    data_labels : array-like of shape (n_samples,), optional
        Partial labels for semi-supervised clustering. Labels should be >= 0
        for labeled points and -1 for unlabeled points. Only used if
        `semi_supervised=True`.

    semi_supervised : bool, default=False
        Whether to perform semi-supervised clustering using provided labels
        in `data_labels`.

    ss_algorithm : {'bc', 'bc_simple'}, default='bc'
        Semi-supervised algorithm to use. Only relevant when
        `semi_supervised=True`. 'bc' allows virtual nodes, 'bc_simple' does not.

    min_samples : int, default=10
        The number of samples in a neighborhood for a point to be considered
        as a core point. If None, defaults to `min_cluster_size`.

    min_cluster_size : int, default=10
        The minimum size for clusters to be recognized as such.

    cluster_selection_method : {'eom', 'leaf'}, default='eom'
        Method for selecting clusters from the condensed tree. 'eom' uses
        Excess of Mass, 'leaf' selects all leaf nodes.

    max_cluster_size : float, default=inf
        Maximum size of clusters to extract. Only relevant for 'eom' method.

    allow_single_cluster : bool, default=False
        Whether to allow the entire dataset to be returned as a single cluster.

    cluster_selection_epsilon : float, default=0.0
        Epsilon value for cluster selection. Governs the minimal distance
        between clusters selected in the hierarchy.

    cluster_selection_persistence : float, default=0.0
        Simplify the hierarchy before cluster extraction by removing nodes
        with persistence below this value.

    sample_weights : array-like of shape (n_samples,), optional
        Weights for each sample. If specified, weighted clustering is performed.

    reproducible : bool, default=False
        Whether to use reproducible (slightly slower thread-safe) algorithms.
        Useful for ensuring exact reproducibility across runs.

    return_trees : bool, default=False
        Whether to return the single linkage tree, condensed tree, and minimum
        spanning tree in addition to labels and probabilities.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster label for each point. Noise points are labeled -1.

    probabilities : ndarray of shape (n_samples,)
        The probability that each point is a core member of its assigned cluster.

    single_linkage_tree : ndarray, optional
        Single linkage tree formed from the mutual reachability distances.
        Only returned if `return_trees=True`.

    condensed_tree : NamedTuple, optional
        Condensed representation of the single linkage tree. Only returned if
        `return_trees=True`.

    min_spanning_tree : ndarray, optional
        Minimum spanning tree constructed from the data. Only returned if
        `return_trees=True`.

    neighbors : ndarray of shape (n_samples, min_samples)
        The k-nearest neighbors of each point.

    core_distances : ndarray of shape (n_samples,)
        The core distance of each point.

    Notes
    -----
    This function is intended for advanced users. The :class:`HDBSCAN` class
    provides a more convenient sklearn-compatible interface.

    Examples
    --------
    >>> from fast_hdbscan import fast_hdbscan
    >>> import numpy as np
    >>> data = np.random.random((100, 2))
    >>> labels, probabilities, neighbors, core_distances = fast_hdbscan(
    ...     data, min_cluster_size=10)
    """
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
    data, min_samples=10, sample_weights=None, reproducible=False
):
    n_threads = numba.get_num_threads()
    numba_tree = build_kdtree(data)
    edges, neighbors, core_distances = parallel_boruvka(
        numba_tree,
        n_threads,
        min_samples=min_samples,
        sample_weights=sample_weights,
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
    """Hierarchical Density-Based Spatial Clustering of Applications with Noise.

    HDBSCAN is a clustering algorithm that uses a hierarchical approach based on
    density, making it robust to varying cluster densities and effective at
    identifying noise. This implementation is optimized for low-dimensional,
    Euclidean data (2D to ~20D) and leverages multi-core parallelization for
    improved performance.

    The algorithm works by:
    1. Computing a k-nearest neighbor graph using a KD-tree
    2. Constructing a minimum spanning tree from mutual reachability distances
    3. Building a single-linkage clustering hierarchy
    4. Extracting clusters from the hierarchy using specified criteria

    This implementation supports several research extensions including:
    - Semi-supervised clustering with partial labels
    - Sample weighting
    - Branch detection for identifying sub-structure within clusters

    Parameters
    ----------
    min_cluster_size : int, default=5
        The minimum number of samples in a group for that group to be considered
        as a cluster; groupings smaller than this size will be left as noise.

    min_samples : int, optional
        The number of samples in a neighborhood for a point to be considered as
        a core point. If None, defaults to `min_cluster_size`.

    cluster_selection_method : {'eom', 'leaf'}, default='eom'
        Method for selecting clusters from the hierarchy:
        - 'eom': Excess of Mass - removes points not essential to the clustering.
        - 'leaf': Select all leaf nodes from the cluster hierarchy.

    allow_single_cluster : bool, default=False
        By default HDBSCAN will not produce a single cluster, setting all points
        as noise. Set to True to allow a single cluster output when appropriate.

    max_cluster_size : float, default=inf
        Maximum size of clusters extracted. This only affects the 'eom' cluster
        selection method. Clusters larger than this are split and treated as
        sub-clusters. Useful for detecting hierarchical cluster structure.

    cluster_selection_epsilon : float, default=0.0
        Governs the minimal distance between clusters returned. Clusters closer
        than this distance in the hierarchy are merged. Useful for controlling
        the granularity of the clustering.

    cluster_selection_persistence : float, default=0.0
        Simplifies the hierarchy before cluster extraction by removing nodes
        with persistence (stability) below this threshold. Helps identify more
        stable cluster structure.

    semi_supervised : bool, default=False
        Enable semi-supervised clustering when partial labels are provided via
        the `y` parameter during fit.

    ss_algorithm : {'bc', 'bc_simple'}, default='bc'
        Semi-supervised algorithm variant:
        - 'bc': Allows virtual nodes for unlabeled point inference.
        - 'bc_simple': Direct label propagation without virtual nodes.

    reproducible : bool, default=False
        Use slightly slower thread-safe algorithms to ensure reproducibility across runs.
        Useful for debugging but may be slower.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in the fitted data. Noise points are
        labeled -1.

    probabilities_ : ndarray of shape (n_samples,)
        Probability that each point is a core member of its assigned cluster.
        Values range from 0 to 1.

    condensed_tree_ : CondensedTree
        The condensed cluster hierarchy from which clusters were extracted.
        Provides visualization and analysis of the clustering structure.
        See :class:`~hdbscan.plots.CondensedTree`.

    single_linkage_tree_ : SingleLinkageTree
        The single linkage tree formed by the mutual reachability distances.
        See :class:`~hdbscan.plots.SingleLinkageTree`.

    minimum_spanning_tree_ : MinimumSpanningTree
        The minimum spanning tree of the input data. Only available if raw
        data was provided during fit.
        See :class:`~hdbscan.plots.MinimumSpanningTree`.

    Examples
    --------
    >>> from fast_hdbscan import HDBSCAN
    >>> import numpy as np
    >>> X = np.random.random((100, 2))
    >>> clusterer = HDBSCAN(min_cluster_size=5)
    >>> cluster_labels = clusterer.fit_predict(X)

    With semi-supervised clustering:

    >>> y_partial = np.full(100, -1)
    >>> y_partial[:10] = 0  # Label first 10 points as cluster 0
    >>> clusterer = HDBSCAN(semi_supervised=True)
    >>> cluster_labels = clusterer.fit_predict(X, y=y_partial)

    References
    ----------
    .. [1] Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013).
           Density-based clustering based on hierarchical density estimates.
           In PAKDD (pp. 160-172).

    .. [2] McInnes, L., Healy, J., & Astels, S. (2017).
           hdbscan: Hierarchical density based clustering.
           The Journal of Open Source Software, 2(11), 205.
    """

    def __init__(
        self,
        *,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        cluster_selection_method: Literal["eom", "leaf"] = "eom",
        allow_single_cluster: bool = False,
        max_cluster_size: float = np.inf,
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_persistence: float = 0.0,
        semi_supervised: bool = False,
        ss_algorithm: Literal["bc", "bc_simple"] = "bc",
        reproducible: bool = False,
        # Removed **kwargs to comply with scikit-learn's API requirements
    ) -> None:
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

    def fit(
        self,
        X: npt.ArrayLike,
        y: Optional[npt.ArrayLike] = None,
        sample_weight: Optional[npt.ArrayLike] = None,
        **fit_params: Any,
    ) -> "HDBSCAN":

        if self.semi_supervised:
            X, y = check_X_y(X, y, accept_sparse="csr", ensure_all_finite=False)
            self._raw_labels = y
            # Replace non-finite labels with -1 labels
            y[~np.isfinite(y)] = -1

            if ~np.any(y != -1):
                raise ValueError(
                    "y must contain at least one label > -1. Currently it only contains -1 and/or non-finite labels!"
                )
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
            clean_data_labels = y[finite_index] if self.semi_supervised else None
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

    def dbscan_clustering(self, epsilon: float) -> npt.NDArray[np.int_]:
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
    """Persistence-based Layered Clustering for Automated Cluster Multiscaling.

    PLSCAN is a clustering algorithm that automatically determines the optimal
    clustering resolution by constructing a hierarchy of clusterings at different
    scales. It identifies the most persistent (stable) clustering layer across
    scales, providing principled automatic cluster count selection.

    The algorithm:
    1. Effectively iteratively applies HDBSCAN at increasing min_cluster_size values
    2. Computes persistence scores for each layer based on cluster stability
    3. Returns the clustering layer with the best persistence/stability

    It achieves this by building all possible HDBSCAN clusterings with progressively
    larger `min_cluster_size` values, without having to re-cluster the data,
    but instead extracts the relevant clusterings from a single hierarchical structure.

    This approach is particularly valuable when the optimal number of clusters
    is unknown and needs to be automatically determined from the data.

    Parameters
    ----------
    min_samples : int, default=5
        The number of samples in a neighborhood for a point to be considered
        as a core point in each effective HDBSCAN iteration.

    max_layers : int, default=10
        Maximum number of clustering layers to construct.
        Each layer uses a progressively larger `min_cluster_size`.

    base_min_cluster_size : int, default=10
        Initial `min_cluster_size` for the first HDBSCAN iteration. Each
        subsequent layer increases this value.

    base_n_clusters : int, optional
        Expected baseline number of clusters. If specified, used to guide
        the search for optimal clustering. If None, determined adaptively.

    layer_similarity_threshold : float, default=0.2
        Threshold for determining when clustering layers are sufficiently
        different. Used to identify redundant or similar layers.

    reproducible : bool, default=False
        Use slightly slower thread-safe algorithms to ensure reproducibility across runs.
        Useful for debugging but may be slower.

    verbose : bool, default=False
        Enable verbose output during layer construction.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels from the best (most persistent) layer. Noise points
        are labeled -1.

    membership_strengths_ : ndarray of shape (n_samples,)
        Probability that each point is a core member of its assigned cluster.

    cluster_layers_ : list of ndarray
        List of cluster label arrays, one for each constructed layer.
        Each element is an array of shape (n_samples,).

    membership_strength_layers_ : list of ndarray
        List of membership strength arrays, one for each layer.

    layer_persistence_scores_ : ndarray of shape (n_layers,)
        Persistence (stability) score for each layer. Higher values indicate
        more stable, reliable clusterings.

    min_cluster_sizes_ : ndarray of shape (n_layers,)
        The `min_cluster_size` value used for each HDBSCAN layer.

    total_persistence_ : float
        Overall persistence score across all layers.

    cluster_tree_ : dict
        Hierarchical relationship between clusters across layers.

    Examples
    --------
    >>> from fast_hdbscan import PLSCAN
    >>> import numpy as np
    >>> X = np.random.random((200, 2))
    >>> clusterer = PLSCAN(max_layers=5)
    >>> cluster_labels = clusterer.fit_predict(X)
    >>> n_layers = len(clusterer.cluster_layers_)
    >>> print(f"Found {n_layers} clustering layers")
    >>> best_layer_idx = np.argmax(clusterer.layer_persistence_scores_)
    >>> print(f"Best layer: {best_layer_idx}")
    """

    def __init__(
        self,
        *,
        min_samples: int = 5,
        max_layers: int = 10,
        base_min_cluster_size: int = 10,
        base_n_clusters: Optional[int] = None,
        layer_similarity_threshold: float = 0.2,
        reproducible: bool = False,
        verbose: bool = False,
    ) -> None:
        self.min_samples = min_samples
        self.max_layers = max_layers
        self.base_min_cluster_size = base_min_cluster_size
        self.base_n_clusters = base_n_clusters
        self.layer_similarity_threshold = layer_similarity_threshold
        self.reproducible = reproducible
        self.verbose = verbose

        self._validate_params()

    def _validate_params(self) -> None:
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

    def fit_predict(
        self,
        X: npt.ArrayLike,
        y: Optional[npt.ArrayLike] = None,
        sample_weight: Optional[npt.ArrayLike] = None,
        **fit_params: Any,
    ) -> npt.NDArray[np.int_]:
        X = validate_data(self, X, accept_sparse="csr", ensure_all_finite=False)
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

    def fit(
        self,
        X: npt.ArrayLike,
        y: Optional[npt.ArrayLike] = None,
        sample_weight: Optional[npt.ArrayLike] = None,
        **fit_params: Any,
    ) -> "PLSCAN":
        _ = self.fit_predict(X, y=y, sample_weight=sample_weight, **fit_params)
        return self
