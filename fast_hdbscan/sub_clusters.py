"""
Converts data point lens values into edge distances and looks for clusters
induced by those distances within the clusters found by HDBSCAN.
"""

import numba
import numpy as np
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.base import BaseEstimator, ClusterMixin

from .hdbscan import to_numpy_rec_array
from .core_graph import core_graph_clusters, core_graph_to_edge_list


def compute_sub_clusters_in_cluster(
    cluster,
    data,
    labels,
    probabilities,
    neighbors,
    core_distances,
    min_spanning_tree,
    parent_labels,
    child_labels,
    lens_callback,
    sample_weights=None,
    **kwargs,
):
    # Convert to within cluster indices (-1 indicates invalid neighbor)
    points = np.nonzero(labels == cluster)[0]

    in_cluster_ids = np.full(data.shape[0], -1, dtype=np.int32)
    in_cluster_ids[points] = np.arange(len(points))
    neighbors = in_cluster_ids[neighbors[points, :]]
    core_distances = core_distances[points]
    min_spanning_tree = min_spanning_tree[
        (parent_labels == cluster) & (child_labels == cluster)
    ]
    min_spanning_tree[:, :2] = in_cluster_ids[min_spanning_tree[:, :2].astype(np.int64)]

    # Compute lens_values
    lens_values = lens_callback(
        data, probabilities, neighbors, core_distances, min_spanning_tree, points
    )

    # Compute branches from core graph
    return (
        *core_graph_clusters(
            lens_values,
            neighbors,
            core_distances,
            min_spanning_tree,
            sample_weights=(
                sample_weights[points] if sample_weights is not None else None
            ),
            **kwargs,
        ),
        lens_values,
        points,
    )


def compute_sub_clusters_per_cluster(
    data,
    labels,
    probabilities,
    neighbors,
    core_distances,
    min_spanning_tree,
    lens_callback,
    num_clusters,
    **kwargs,
):
    # Loop could be parallel over clusters, but njit-compiling all called
    # functions slows down imports with a factor > 2 for small gains. Instead,
    # parts of each loop are parallel over points in the clusters.
    parent_labels = labels[min_spanning_tree[:, 0].astype(np.int64)]
    child_labels = labels[min_spanning_tree[:, 1].astype(np.int64)]
    return [
        compute_sub_clusters_in_cluster(
            cluster,
            data,
            labels,
            probabilities,
            neighbors,
            core_distances,
            min_spanning_tree,
            parent_labels,
            child_labels,
            lens_callback,
            **kwargs,
        )
        for cluster in range(num_clusters)
    ]


def update_labels(
    cluster_probabilities,
    sub_labels_list,
    sub_probabilities_list,
    lens_values_list,
    points_list,
    data_size,
):
    labels = np.full(data_size, -1, dtype=np.int64)
    probabilities = cluster_probabilities.copy()
    sub_labels = np.zeros(data_size, dtype=np.int64)
    sub_probabilities = np.zeros(data_size, dtype=np.float32)
    lens_values = np.zeros(data_size, dtype=np.float32)

    running_id = 0
    for points, _labels, _probs, _lens in zip(
        points_list,
        sub_labels_list,
        sub_probabilities_list,
        lens_values_list,
    ):
        unique_labels = np.unique(_labels)
        labels[points] = _labels + int(unique_labels[0] == -1) + running_id
        sub_labels[points] = _labels
        sub_probabilities[points] = _probs
        probabilities[points] += _probs
        probabilities[points] /= 2
        lens_values[points] = _lens
        running_id += len(unique_labels)

    return labels, probabilities, sub_labels, sub_probabilities, lens_values


@numba.njit()
def propagate_labels_per_cluster(graph, sub_labels):
    # create undirected core graph
    undirected = [
        {np.int64(0): np.float64(0.0) for _ in range(0)} for _ in range(len(sub_labels))
    ]
    for idx, (start, end) in enumerate(zip(graph.indptr, graph.indptr[1:])):
        for i in range(start, end):
            neighbor = graph.indices[i]
            if sub_labels[neighbor] == -1:
                undirected[neighbor][idx] = 1 / graph.weights[i]
            if sub_labels[idx] == -1:
                undirected[idx][neighbor] = 1 / graph.weights[i]

    # repeat density-weighted majority votes on noise points until all are assigned
    while True:
        noise_idx = np.nonzero(sub_labels == -1)[0]
        if noise_idx.shape[0] == 0:
            break
        for idx in noise_idx:
            candidates = {np.int64(0): np.float64(0.0) for _ in range(0)}
            for neighbor_idx, weight in undirected[idx].items():
                label = sub_labels[neighbor_idx]
                if label == -1:
                    continue
                candidates[label] = candidates.get(label, 0.0) + weight

            if len(candidates) == 0:
                continue
            max_weight = -np.inf
            max_candidate = -1
            for candidate, weight in candidates.items():
                if weight > max_weight:
                    max_weight = weight
                    max_candidate = candidate
            sub_labels[idx] = max_candidate
    return sub_labels


def propagate_sub_cluster_labels(labels, sub_labels, graph_list, points_list):
    running_id = 0
    for points, core_graph in zip(
        points_list,
        graph_list,
    ):
        unique_sub_labels = np.unique(sub_labels[points])
        has_noise = unique_sub_labels[0] == -1 and len(unique_sub_labels) > 1
        if has_noise:
            sub_labels[points] = propagate_labels_per_cluster(
                core_graph, sub_labels[points]
            )
        labels[points] = sub_labels[points] + running_id
        running_id += len(unique_sub_labels) - int(has_noise)

    return labels, sub_labels


def remap_results(
    labels,
    probabilities,
    cluster_labels,
    cluster_probabilities,
    sub_labels,
    sub_probabilities,
    lens_values,
    points,
    finite_index,
    num_points,
):
    new_labels = np.full(num_points, -1, dtype=labels.dtype)
    new_labels[finite_index] = labels
    labels = new_labels

    new_probabilities = np.full(num_points, 0.0, dtype=probabilities.dtype)
    new_probabilities[finite_index] = probabilities
    probabilities = new_probabilities

    new_cluster_labels = np.full(num_points, -1, dtype=cluster_labels.dtype)
    new_cluster_labels[finite_index] = cluster_labels
    cluster_labels = new_cluster_labels

    new_cluster_probabilities = np.full(
        num_points, 0.0, dtype=cluster_probabilities.dtype
    )
    new_cluster_probabilities[finite_index] = cluster_probabilities
    cluster_probabilities = new_cluster_probabilities

    new_sub_labels = np.full(num_points, 0, dtype=sub_labels.dtype)
    new_sub_labels[finite_index] = sub_labels
    sub_labels = new_sub_labels

    new_sub_probabilities = np.full(num_points, 1.0, dtype=sub_probabilities.dtype)
    new_sub_probabilities[finite_index] = sub_probabilities
    sub_probabilities = new_sub_probabilities

    new_lens_values = np.full(num_points, 0.0, dtype=lens_values.dtype)
    new_lens_values[finite_index] = lens_values
    lens_values = new_lens_values

    for pts in points:
        pts[:] = finite_index[pts]

    return (
        labels,
        probabilities,
        cluster_labels,
        cluster_probabilities,
        sub_labels,
        sub_probabilities,
        lens_values,
        points,
    )


def find_sub_clusters(
    clusterer,
    cluster_labels=None,
    cluster_probabilities=None,
    sample_weights=None,
    lens_callback=None,
    *,
    min_cluster_size=None,
    max_cluster_size=None,
    allow_single_cluster=None,
    cluster_selection_method=None,
    cluster_selection_epsilon=0.0,
    cluster_selection_persistence=0.0,
    propagate_labels=False,
):
    check_is_fitted(
        clusterer,
        "_min_spanning_tree",
        msg="You first need to fit the HDBSCAN model before detecting sub clusters.",
    )

    # Validate input parameters
    if cluster_labels is None:
        cluster_labels = clusterer.labels_
    elif cluster_probabilities is None:
        cluster_probabilities = np.ones(cluster_labels.shape[0], dtype=np.float32)
    if cluster_probabilities is None:
        cluster_probabilities = clusterer.probabilities_
    if sample_weights is not None:
        sample_weights = _check_sample_weight(
            sample_weights, clusterer._raw_data, dtype=np.float32
        )
    if min_cluster_size is None:
        min_cluster_size = clusterer.min_cluster_size
    if max_cluster_size is None:
        max_cluster_size = clusterer.max_cluster_size
    if allow_single_cluster is None:
        allow_single_cluster = clusterer.allow_single_cluster
    if cluster_selection_method is None:
        cluster_selection_method = clusterer.cluster_selection_method

    if not (
        np.issubdtype(type(min_cluster_size), np.integer) and min_cluster_size >= 2
    ):
        raise ValueError(
            f"min_cluster_size must be an integer greater or equal "
            f"to 2,  {min_cluster_size} given."
        )
    if max_cluster_size <= 0:
        raise ValueError(
            f"max_cluster_size must be greater 0, {max_cluster_size} given."
        )
    if not (
        np.issubdtype(type(cluster_selection_persistence), np.floating)
        and cluster_selection_persistence >= 0.0
    ):
        raise ValueError(
            f"cluster_selection_persistence must be a float greater or equal to "
            f"0.0, {cluster_selection_persistence} given."
        )
    if not (
        np.issubdtype(type(cluster_selection_epsilon), np.floating)
        and cluster_selection_epsilon >= 0.0
    ):
        raise ValueError(
            f"cluster_selection_epsilon must be a float greater or equal to "
            f"0.0, {cluster_selection_epsilon} given."
        )
    if cluster_selection_method not in ("eom", "leaf"):
        raise ValueError(
            f"Invalid cluster_selection_method: {cluster_selection_method}\n"
            f'Should be one of: "eom", "leaf"\n'
        )
    if np.all(cluster_labels == -1):
        raise ValueError("Input contains only noise points.")

    # Recover finite data points
    data = clusterer._raw_data
    num_points = data.shape[0]
    last_outlier = np.searchsorted(
        clusterer._condensed_tree["lambda_val"], 0.0, side="right"
    )
    if last_outlier > 0:
        finite_index = np.setdiff1d(
            np.arange(data.shape[0]), clusterer._condensed_tree["child"][:last_outlier]
        )
        data = data[finite_index]
        cluster_labels = cluster_labels[finite_index]
        cluster_probabilities = cluster_probabilities[finite_index]
        sample_weights = (
            sample_weights[finite_index] if sample_weights is not None else None
        )

    # Convert lens value array to callback
    if isinstance(lens_callback, np.ndarray):
        if len(lens_callback) != num_points:
            raise ValueError(
                "when providing values as lens_callback, they must have"
                f"the same length as the data, {len(lens_callback)} != {data.shape[0]}"
            )
        if last_outlier > 0:
            lens_values = lens_callback[finite_index]
        else:
            lens_values = lens_callback

        lens_callback = lambda a, b, c, d, e, pts: lens_values[pts]

    # Compute per-cluster sub clusters
    num_clusters = np.max(cluster_labels) + 1
    (
        sub_labels,
        sub_probabilities,
        linkage_trees,
        condensed_trees,
        spanning_trees,
        core_graphs,
        lens_values,
        points,
    ) = zip(
        *compute_sub_clusters_per_cluster(
            data,
            cluster_labels,
            cluster_probabilities,
            clusterer._neighbors,
            clusterer._core_distances,
            clusterer._min_spanning_tree,
            lens_callback,
            num_clusters,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            allow_single_cluster=allow_single_cluster,
            cluster_selection_method=cluster_selection_method,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_persistence=cluster_selection_persistence,
            sample_weights=sample_weights,
        )
    )

    # Handle override labels failure cases
    condensed_trees = [
        to_numpy_rec_array(tree) if tree.parent.shape[0] > 0 else None
        for tree in condensed_trees
    ]
    linkage_trees = [tree if tree.shape[0] > 0 else None for tree in linkage_trees]

    # Aggregate the results
    (labels, probabilities, sub_labels, sub_probabilities, lens_values) = update_labels(
        cluster_probabilities,
        sub_labels,
        sub_probabilities,
        lens_values,
        points,
        data.shape[0],
    )

    # Propagate labels if requested
    if propagate_labels:
        labels, sub_labels = propagate_sub_cluster_labels(
            labels, sub_labels, core_graphs, points
        )

    # Reset for infinite data points
    if last_outlier > 0:
        (
            labels,
            probabilities,
            cluster_labels,
            cluster_probabilities,
            sub_labels,
            sub_probabilities,
            lens_values,
            points,
        ) = remap_results(
            labels,
            probabilities,
            cluster_labels,
            cluster_probabilities,
            sub_labels,
            sub_probabilities,
            lens_values,
            points,
            finite_index,
            clusterer._raw_data.shape[0],
        )

    return (
        labels,
        probabilities,
        cluster_labels,
        cluster_probabilities,
        sub_labels,
        sub_probabilities,
        core_graphs,
        condensed_trees,
        linkage_trees,
        spanning_trees,
        lens_values,
        points,
    )


class SubClusterDetector(ClusterMixin, BaseEstimator):
    """Performs a lens-value sub-cluster detection post-processing step
    on a HDBSCAN clusterer."""

    def __init__(
        self,
        *,
        lens_values=None,
        min_cluster_size=None,
        max_cluster_size=None,
        allow_single_cluster=False,
        cluster_selection_method="eom",
        cluster_selection_epsilon=0.0,
        cluster_selection_persistence=0.0,
        propagate_labels=False,
    ):
        self.lens_values = lens_values
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.allow_single_cluster = allow_single_cluster
        self.cluster_selection_method = cluster_selection_method
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_persistence = cluster_selection_persistence
        self.propagate_labels = propagate_labels

    def fit(
        self,
        clusterer,
        labels=None,
        probabilities=None,
        sample_weight=None,
        lens_callback=None,
    ):
        """labels and probabilities override the clusterer's values."""
        # get_params breaks with inherited classes!
        (
            self.labels_,
            self.probabilities_,
            self.cluster_labels_,
            self.cluster_probabilities_,
            self.sub_cluster_labels_,
            self.sub_cluster_probabilities_,
            self._approximation_graphs,
            self._condensed_trees,
            self._linkage_trees,
            self._spanning_trees,
            self.lens_values_,
            self.cluster_points_,
        ) = find_sub_clusters(
            clusterer,
            labels,
            probabilities,
            sample_weight,
            self.lens_values if lens_callback is None else lens_callback,
            min_cluster_size=self.min_cluster_size,
            max_cluster_size=self.max_cluster_size,
            allow_single_cluster=self.allow_single_cluster,
            cluster_selection_method=self.cluster_selection_method,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            cluster_selection_persistence=self.cluster_selection_persistence,
            propagate_labels=self.propagate_labels,
        )
        # also store the core distances and raw data for the member functions
        self._raw_data = clusterer._raw_data
        self._core_distances = clusterer._core_distances
        return self

    def propagated_labels(self):
        """Propagate sub-cluster labels to noise points."""
        check_is_fitted(
            self,
            "labels_",
            msg="You first need to fit the SubClusterDetector model before propagating the labels.",
        )
        return propagate_sub_cluster_labels(
            self.labels_.copy(),
            self.sub_cluster_labels_.copy(),
            self._approximation_graphs,
            self.cluster_points_,
        )

    @property
    def approximation_graph_(self):
        """See :class:`~hdbscan.plots.ApproximationGraph` for documentation."""
        return self._make_approximation_graph()

    def _make_approximation_graph(self, lens_name=None, sub_cluster_name=None):
        from hdbscan.plots import ApproximationGraph

        check_is_fitted(
            self,
            "_approximation_graphs",
            msg="You first need to fit the BranchDetector model before accessing the approximation graphs",
        )

        edge_lists = []
        for graph, points in zip(self._approximation_graphs, self.cluster_points_):
            edges = core_graph_to_edge_list(graph)
            edges[:, 0] = points[edges[:, 0].astype(np.int64)]
            edges[:, 1] = points[edges[:, 1].astype(np.int64)]
            edge_lists.append(edges)

        return ApproximationGraph(
            edge_lists,
            self.labels_,
            self.probabilities_,
            self.lens_values_,
            self.cluster_labels_,
            self.cluster_probabilities_,
            self.sub_cluster_labels_,
            self.sub_cluster_probabilities_,
            lens_name=lens_name,
            sub_cluster_name=sub_cluster_name,
            raw_data=self._raw_data,
        )

    @property
    def condensed_trees_(self):
        """See :class:`~hdbscan.plots.CondensedTree` for documentation."""
        from hdbscan.plots import CondensedTree

        check_is_fitted(
            self,
            "_condensed_trees",
            msg="You first need to fit the BranchDetector model before accessing the condensed trees",
        )
        return [
            (
                CondensedTree(
                    tree,
                    self.sub_cluster_labels_[points],
                )
                if tree is not None
                else None
            )
            for tree, points in zip(
                self._condensed_trees,
                self.cluster_points_,
            )
        ]

    @property
    def linkage_trees_(self):
        """See :class:`~hdbscan.plots.SingleLinkageTree` for documentation."""
        from hdbscan.plots import SingleLinkageTree

        check_is_fitted(
            self,
            "_linkage_trees",
            msg="You first need to fit the BranchDetector model before accessing the linkage trees",
        )
        return [
            SingleLinkageTree(tree) if tree is not None else None
            for tree in self._linkage_trees
        ]

    @property
    def spanning_trees_(self):
        """See :class:`~hdbscan.plots.MinimumSpanningTree` for documentation."""
        from hdbscan.plots import MinimumSpanningTree

        check_is_fitted(
            self,
            "_spanning_trees",
            msg="You first need to fit the BranchDetector model before accessing the linkage trees",
        )
        return [
            MinimumSpanningTree(tree, self._raw_data) for tree in self._spanning_trees
        ]
