import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.neighbors import KDTree

from warnings import warn

from .numba_kdtree import kdtree_to_numba
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
    ss_algorithm='bc',
    min_samples=10,
    min_cluster_size=10,
    cluster_selection_method="eom",
    max_cluster_size=np.inf,
    allow_single_cluster=False,
    cluster_selection_epsilon=0.0,
    cluster_selection_persistence=0.0,
    sample_weights=None,
    return_trees=False,
):
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


def compute_minimum_spanning_tree(data, min_samples=10, sample_weights=None):
    sklearn_tree = KDTree(data)
    numba_tree = kdtree_to_numba(sklearn_tree)
    edges, neighbors, core_distances = parallel_boruvka(
        numba_tree, min_samples=min_samples, sample_weights=sample_weights
    )
    return edges, neighbors, core_distances


def clusters_from_spanning_tree(
    minimum_spanning_tree,
    data_labels=None,
    semi_supervised=False,
    ss_algorithm='bc',
    min_cluster_size=10,
    cluster_selection_method="eom",
    max_cluster_size=np.inf,
    allow_single_cluster=False,
    cluster_selection_epsilon=0.0,
    cluster_selection_persistence=0.0,
    sample_weights=None,
):
    n_points = minimum_spanning_tree.shape[0] + 1
    sorted_mst = minimum_spanning_tree[np.argsort(minimum_spanning_tree.T[2])]
    
    if sample_weights is None:
        linkage_tree = mst_to_linkage_tree(sorted_mst)
    else:
        linkage_tree = mst_to_linkage_tree_w_sample_weights(sorted_mst, sample_weights)
    
    condensed_tree = condense_tree(
        linkage_tree, min_cluster_size=min_cluster_size, sample_weights=sample_weights
    )
    if cluster_selection_persistence > 0.0:
        condensed_tree = simplify_hierarchy(
            condensed_tree, n_points, cluster_selection_persistence
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
        condensed_tree,
        selected_clusters,
        cluster_selection_epsilon,
        n_samples=n_points
    )
    membership_strengths = get_point_membership_strength_vector(
        condensed_tree, selected_clusters, clusters
    )

    return clusters, membership_strengths, linkage_tree, condensed_tree, sorted_mst


class HDBSCAN(BaseEstimator, ClusterMixin):
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
        ss_algorithm='bc',
        **kwargs,
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

    def fit(self, X, y=None, sample_weight=None, **fit_params):

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
            X = check_array(X, accept_sparse="csr", ensure_all_finite=False)
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
            sample_weight = sample_weight[finite_index] if sample_weight is not None else None

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
            self._core_distances
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
