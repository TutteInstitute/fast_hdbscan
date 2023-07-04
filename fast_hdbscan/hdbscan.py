import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.neighbors import KDTree

from warnings import warn

from .numba_kdtree import kdtree_to_numba
from .boruvka import parallel_boruvka
from .cluster_trees import (
    mst_to_linkage_tree,
    condense_tree,
    extract_eom_clusters,
    extract_leaves,
    cluster_epsilon_search,
    get_cluster_labelling_at_cut,
    get_cluster_label_vector,
    get_point_membership_strength_vector,
    cluster_tree_from_condensed_tree,
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
            ("child_size", np.intp),
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
    min_samples=10,
    min_cluster_size=10,
    cluster_selection_method="eom",
    allow_single_cluster=False,
    cluster_selection_epsilon=0.0,
    return_trees=False,
):
    data = check_array(data)

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
        raise ValueError('Cluster selection epsilon must be a positive floating point number!')

    sklearn_tree = KDTree(data)
    numba_tree = kdtree_to_numba(sklearn_tree)
    edges = parallel_boruvka(
        numba_tree, min_samples=min_cluster_size if min_samples is None else min_samples
    )
    sorted_mst = edges[np.argsort(edges.T[2])]
    linkage_tree = mst_to_linkage_tree(sorted_mst)
    condensed_tree = condense_tree(linkage_tree, min_cluster_size=min_cluster_size)
    if cluster_selection_epsilon > 0.0 or cluster_selection_method == "eom":
        cluster_tree = cluster_tree_from_condensed_tree(condensed_tree)

    if cluster_selection_method == "eom":
        selected_clusters = extract_eom_clusters(
            condensed_tree, cluster_tree, allow_single_cluster=allow_single_cluster
        )
    elif cluster_selection_method == "leaf":
        selected_clusters = extract_leaves(
            condensed_tree, allow_single_cluster=allow_single_cluster
        )
    else:
        raise ValueError(f"Invalid cluster_selection_method {cluster_selection_method}")
    
    if len(selected_clusters) > 1 and cluster_selection_epsilon > 0.0:
        selected_clusters = cluster_epsilon_search(
            selected_clusters, cluster_tree,
            min_persistence=cluster_selection_epsilon,
        )

    clusters = get_cluster_label_vector(condensed_tree, selected_clusters, cluster_selection_epsilon)
    membership_strengths = get_point_membership_strength_vector(
        condensed_tree, selected_clusters, clusters
    )

    if return_trees:
        return clusters, membership_strengths, linkage_tree, condensed_tree, sorted_mst
    return clusters, membership_strengths


class HDBSCAN(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        *,
        min_cluster_size=5,
        min_samples=None,
        cluster_selection_method="eom",
        allow_single_cluster=False,
        cluster_selection_epsilon=0.0,
        **kwargs,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.cluster_selection_epsilon = cluster_selection_epsilon

    def fit(self, X, y=None, **fit_params):
        X = check_array(X, accept_sparse="csr", force_all_finite=False)
        self._raw_data = X

        self._all_finite = np.all(np.isfinite(X))
        if ~self._all_finite:
            # Pass only the purely finite indices into hdbscan
            # We will later assign all non-finite points to the background -1 cluster
            finite_index = np.where(np.isfinite(X).sum(axis=1) == X.shape[1])[0]
            clean_data = X[finite_index]
            internal_to_raw = {
                x: y for x, y in zip(range(len(finite_index)), finite_index)
            }
            outliers = list(set(range(X.shape[0])) - set(finite_index))
        else:
            clean_data = X

        kwargs = self.get_params()

        (
            self.labels_,
            self.probabilities_,
            self._single_linkage_tree,
            self._condensed_tree,
            self._min_spanning_tree,
        ) = fast_hdbscan(clean_data, return_trees=True, **kwargs)

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
        return get_cluster_labelling_at_cut(
            self._single_linkage_tree,
            epsilon,
            self.min_samples if self.min_samples is not None else self.min_cluster_size,
        )

    @property
    def condensed_tree_(self):
        if self._condensed_tree is not None:
            return CondensedTree(
                self._condensed_tree,
                self.cluster_selection_method,
                self.allow_single_cluster,
            )
        else:
            raise AttributeError(
                "No condensed tree was generated; try running fit first."
            )

    @property
    def single_linkage_tree_(self):
        if self._single_linkage_tree is not None:
            return SingleLinkageTree(self._single_linkage_tree)
        else:
            raise AttributeError(
                "No single linkage tree was generated; try running fit first."
            )

    @property
    def minimum_spanning_tree_(self):
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
