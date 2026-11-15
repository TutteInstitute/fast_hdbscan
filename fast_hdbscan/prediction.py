# -*- coding: utf-8 -*-
# Adapted from hdbscan.prediction (https://github.com/scikit-learn-contrib/hdbscan)
# Original author: Leland McInnes <leland.mcinnes@gmail.com>
# Original license: BSD 3 clause

from collections import namedtuple
from warnings import warn

import numba
import numpy as np

from .cluster_trees import CondensedTree, max_lambdas, subtree_max_lambdas
from .numba_kdtree import build_kdtree, parallel_tree_query
from .variables import NUMBA_CACHE

# Everything the prediction kernels read, indexed by condensed tree node id
# (points first, then clusters from ``root`` up), so they need no dictionaries.
PlacementTree = namedtuple(
    "PlacementTree",
    [
        "root",
        "min_samples",
        "core_distances",
        "fit_row_index",
        "node_parent",
        "node_lambda",
        "cluster_label",
        "cluster_max_lambda",
        "subtree_max_lambda",
        "root_noise_lambda",
        "selected_clusters",
        "max_child_lambda",
    ],
)


@numba.njit(cache=NUMBA_CACHE)
def _cluster_labels(parent, child, lambda_val, child_size, node_parent, root, selected_clusters):
    """Label of the selected cluster at or above each node, and the max lambda
    ``fit`` divides by for ``probabilities_``.
    """
    n_nodes = node_parent.shape[0]
    cluster_label = np.full(n_nodes, -1, dtype=np.intp)
    cluster_max_lambda = np.zeros(n_nodes, dtype=np.float64)
    if selected_clusters.shape[0] == 0:
        return cluster_label, cluster_max_lambda

    deaths = max_lambdas(
        CondensedTree(parent, child, lambda_val, child_size), set(selected_clusters)
    )
    for label in range(selected_clusters.shape[0]):
        cluster = selected_clusters[label]
        cluster_label[cluster] = label
        cluster_max_lambda[cluster] = deaths[cluster]
    # parents have smaller ids than their children, so one ascending pass
    # gives every cluster below a selected one its label
    for cluster in range(root + 1, n_nodes):
        if cluster_label[cluster] < 0 and cluster_label[node_parent[cluster]] >= 0:
            cluster_label[cluster] = cluster_label[node_parent[cluster]]
            cluster_max_lambda[cluster] = cluster_max_lambda[node_parent[cluster]]
    return cluster_label, cluster_max_lambda


def _root_noise_lambda(lambda_val, parent, root, selected_clusters, cluster_selection_epsilon):
    """Smallest lambda a point needs to join the root when the root is the only
    selected cluster, as ``get_cluster_label_vector`` decides at fit time.
    """
    if selected_clusters.shape[0] != 1 or selected_clusters[0] != root:
        return 0.0
    if cluster_selection_epsilon > 0.0:
        return 1.0 / cluster_selection_epsilon
    return lambda_val[parent == root].max()


class PredictionData(object):
    """Data needed to place new points in the clustering of a fitted HDBSCAN.

    Built by ``HDBSCAN(prediction_data=True)`` or
    ``HDBSCAN.generate_prediction_data()``.

    Attributes
    ----------
    raw_data : ndarray, shape (n_finite_samples, n_features)
        The finite training points, in the order the core distances use.

    tree : numba KD-tree
        KD-tree over ``raw_data``.

    placement : PlacementTree
        Condensed tree and fit results in the flat form the kernels read.

    exemplars : list of ndarray
        The exemplar points of each selected cluster, in label order.
    """

    def __init__(
        self,
        data,
        fit_row_index,
        condensed_tree,
        selected_clusters,
        core_distances,
        min_samples,
        cluster_selection_epsilon,
        exemplars=None,
    ):
        self.raw_data = data
        self.tree = build_kdtree(data)
        self.has_tree = condensed_tree.shape[0] > 0
        self.n_selected_clusters = len(selected_clusters)
        self.exemplars = exemplars
        if not self.has_tree:
            self.placement = None
            return

        parent = np.ascontiguousarray(condensed_tree["parent"])
        child = np.ascontiguousarray(condensed_tree["child"])
        lambda_val = np.ascontiguousarray(condensed_tree["lambda_val"])
        child_size = np.ascontiguousarray(condensed_tree["child_size"])
        root = parent.min()
        selected_clusters = np.sort(np.asarray(selected_clusters, dtype=np.intp))

        n_nodes = max(parent.max(), child.max()) + 1
        node_parent = np.arange(n_nodes, dtype=np.intp)
        node_lambda = np.zeros(n_nodes, dtype=np.float64)
        node_parent[child] = parent
        node_lambda[child] = lambda_val
        max_child_lambda = np.zeros(n_nodes, dtype=np.float64)
        np.maximum.at(max_child_lambda, parent, lambda_val)

        cluster_label, cluster_max_lambda = _cluster_labels(
            parent, child, lambda_val, child_size, node_parent, root, selected_clusters
        )
        self.placement = PlacementTree(
            root=root,
            min_samples=min_samples,
            core_distances=np.asarray(core_distances, dtype=np.float64),
            fit_row_index=np.asarray(fit_row_index, dtype=np.intp),
            node_parent=node_parent,
            node_lambda=node_lambda,
            cluster_label=cluster_label,
            cluster_max_lambda=cluster_max_lambda,
            subtree_max_lambda=subtree_max_lambdas(parent, child, lambda_val, root),
            root_noise_lambda=_root_noise_lambda(
                lambda_val, parent, root, selected_clusters, cluster_selection_epsilon
            ),
            selected_clusters=selected_clusters,
            max_child_lambda=max_child_lambda,
        )

    def query(self, points):
        # as in hdbscan: twice min_samples candidates for the nearest
        # mutual reachability neighbor
        n_candidates = min(2 * self.placement.min_samples, self.raw_data.shape[0])
        distances, neighbors = parallel_tree_query(
            self.tree, np.ascontiguousarray(points, dtype=np.float32), k=n_candidates
        )
        return distances.astype(np.float64), neighbors


@numba.njit(cache=NUMBA_CACHE)
def nearest_mutual_reachability_neighbor(
    neighbor_distances, neighbor_indices, core_distances, min_samples
):
    # A new point is not in the KD-tree, so its min_samples-th nearest training
    # point is at index min_samples - 1. Fit excludes the point itself, too.
    point_core_distance = 0.0 if min_samples == 1 else neighbor_distances[min_samples - 1]
    best_index = neighbor_indices[0]
    best_distance = np.inf
    for i in range(neighbor_indices.shape[0]):
        distance = max(
            core_distances[neighbor_indices[i]], point_core_distance, neighbor_distances[i]
        )
        if distance < best_distance:
            best_distance = distance
            best_index = neighbor_indices[i]
    lambda_ = 1.0 / best_distance if best_distance > 0.0 else np.inf
    return best_index, lambda_


@numba.njit(cache=NUMBA_CACHE)
def place_point(neighbor_distances, neighbor_indices, placement):
    """Training point (fit row) a new point connects to, the cluster that
    point falls out of, and the new point's lambda."""
    nearest, lambda_ = nearest_mutual_reachability_neighbor(
        neighbor_distances, neighbor_indices, placement.core_distances, placement.min_samples
    )
    neighbor = placement.fit_row_index[nearest]
    return neighbor, placement.node_parent[neighbor], lambda_


@numba.njit(parallel=True, cache=NUMBA_CACHE)
def predict_clusters(neighbor_distances, neighbor_indices, placement):
    n_points = neighbor_indices.shape[0]
    labels = np.empty(n_points, dtype=np.intp)
    probabilities = np.empty(n_points, dtype=np.float64)
    connecting_points = np.empty(n_points, dtype=np.intp)

    for i in numba.prange(n_points):
        neighbor, cluster, lambda_ = place_point(
            neighbor_distances[i], neighbor_indices[i], placement
        )
        if placement.node_lambda[neighbor] > lambda_:
            while cluster > placement.root and placement.node_lambda[cluster] >= lambda_:
                cluster = placement.node_parent[cluster]

        label = placement.cluster_label[cluster]
        if lambda_ < placement.root_noise_lambda:
            label = -1
        probability = 0.0
        if label >= 0:
            # same rule as get_point_membership_strength_vector
            max_lambda = placement.cluster_max_lambda[cluster]
            if max_lambda == 0.0 or not np.isfinite(lambda_):
                probability = 1.0
            else:
                probability = min(lambda_, max_lambda) / max_lambda

        labels[i] = label
        probabilities[i] = probability
        connecting_points[i] = neighbor

    return labels, probabilities, connecting_points


@numba.njit(parallel=True, cache=NUMBA_CACHE)
def predict_outlier_scores(neighbor_distances, neighbor_indices, placement):
    n_points = neighbor_indices.shape[0]
    scores = np.empty(n_points, dtype=np.float64)

    for i in numba.prange(n_points):
        neighbor, cluster, lambda_ = place_point(
            neighbor_distances[i], neighbor_indices[i], placement
        )
        # an exact duplicate of a training point scores like that point
        if neighbor_distances[i, 0] == 0.0:
            lambda_ = placement.node_lambda[neighbor]
        max_lambda = placement.subtree_max_lambda[cluster]
        if max_lambda == 0.0 or not np.isfinite(lambda_):
            scores[i] = 0.0
        else:
            scores[i] = (max_lambda - lambda_) / max_lambda

    return scores


def _checked_prediction_data(clusterer, points_to_predict):
    prediction_data = getattr(clusterer, "prediction_data_", None)
    if prediction_data is None:
        raise ValueError(
            "Clusterer does not have prediction data! Try fitting with"
            " prediction_data=True set, or run generate_prediction_data on the clusterer"
        )
    points_to_predict = np.asarray(points_to_predict, dtype=np.float64)
    if points_to_predict.ndim != 2 or points_to_predict.shape[1] != prediction_data.raw_data.shape[1]:
        raise ValueError("New points dimension does not match fit data!")
    if not np.isfinite(points_to_predict).all():
        raise ValueError("New points must be finite.")
    return prediction_data, points_to_predict


def approximate_predict(clusterer, points_to_predict, return_connecting_points=False):
    """Predict the cluster label of new points.

    Each new point is placed in the condensed tree of the fitted clusterer
    through its nearest mutual reachability neighbor, without refitting, so
    the clustering itself never changes and no new clusters are found.

    Parameters
    ----------
    clusterer : HDBSCAN
        A fitted clusterer with prediction data (``prediction_data=True`` or
        ``generate_prediction_data()``).

    points_to_predict : array, shape (n_samples, n_features)
        The new points.

    return_connecting_points : bool, optional (default False)
        Also return, for each new point, the index of the training point it
        connects to.

    Returns
    -------
    labels : array, shape (n_samples,)
        Predicted labels, -1 for noise.

    probabilities : array, shape (n_samples,)
        Membership strength of each point in its predicted cluster, computed
        like ``probabilities_``.

    connecting_points : array, shape (n_samples,)
        Only with ``return_connecting_points=True``.
    """
    prediction_data, points_to_predict = _checked_prediction_data(
        clusterer, points_to_predict
    )
    n_points = points_to_predict.shape[0]
    if prediction_data.n_selected_clusters == 0:
        warn(
            "Clusterer does not have any defined clusters, new data"
            " will be automatically predicted as noise."
        )
        labels = np.full(n_points, -1, dtype=np.intp)
        probabilities = np.zeros(n_points, dtype=np.float64)
        connecting_points = np.full(n_points, -1, dtype=np.intp)
    else:
        neighbor_distances, neighbor_indices = prediction_data.query(points_to_predict)
        labels, probabilities, connecting_points = predict_clusters(
            neighbor_distances, neighbor_indices, prediction_data.placement
        )

    if return_connecting_points:
        return labels, probabilities, connecting_points
    return labels, probabilities


def approximate_predict_scores(clusterer, points_to_predict):
    """Predict GLOSH outlier scores for new points.

    Parameters
    ----------
    clusterer : HDBSCAN
        A fitted clusterer with prediction data.

    points_to_predict : array, shape (n_samples, n_features)
        The new points.

    Returns
    -------
    scores : array, shape (n_samples,)
        Outlier score of each point, computed like ``outlier_scores_``.
    """
    prediction_data, points_to_predict = _checked_prediction_data(
        clusterer, points_to_predict
    )
    if not prediction_data.has_tree:
        warn(
            "Clusterer does not have a condensed tree, new data"
            " will be automatically predicted as outliers."
        )
        return np.ones(points_to_predict.shape[0], dtype=np.float64)

    neighbor_distances, neighbor_indices = prediction_data.query(points_to_predict)
    return predict_outlier_scores(
        neighbor_distances, neighbor_indices, prediction_data.placement
    )


@numba.njit(parallel=True, cache=NUMBA_CACHE)
def merge_heights(points, placement):
    """Lambda at which each point's cluster joins each selected cluster, or
    the point's own lambda when one of them sits below the other, as in
    hdbscan's ``merge_height``.
    """
    clusters = placement.selected_clusters
    heights = np.empty((points.shape[0], clusters.shape[0]), dtype=np.float64)
    for i in numba.prange(points.shape[0]):
        point_cluster = placement.node_parent[points[i]]
        for j in range(clusters.shape[0]):
            left = point_cluster
            right = clusters[j]
            last = left
            took_left = False
            took_right = False
            # parents have smaller ids than their children
            while left != right:
                if left > right:
                    took_left = True
                    last = left
                    left = placement.node_parent[left]
                else:
                    took_right = True
                    last = right
                    right = placement.node_parent[right]
            if took_left and took_right:
                heights[i, j] = placement.node_lambda[last]
            else:
                heights[i, j] = placement.node_lambda[points[i]]
    return heights


@numba.njit(parallel=True, cache=NUMBA_CACHE)
def min_exemplar_distances(points, exemplars, exemplar_labels, n_clusters):
    """Euclidean distance from each point to the nearest exemplar of each cluster."""
    # a scalar loop over features avoids allocating a temporary array per pair
    distances = np.full((points.shape[0], n_clusters), np.inf)
    for i in numba.prange(points.shape[0]):
        for e in range(exemplars.shape[0]):
            squared_distance = 0.0
            for f in range(points.shape[1]):
                difference = points[i, f] - exemplars[e, f]
                squared_distance += difference * difference
            if squared_distance < distances[i, exemplar_labels[e]]:
                distances[i, exemplar_labels[e]] = squared_distance
        for j in range(n_clusters):
            distances[i, j] = np.sqrt(distances[i, j])
    return distances


def all_points_membership_vectors(clusterer):
    """Soft cluster membership of every point the clusterer was fit on.

    Follows hdbscan's ``all_points_membership_vectors``: each point's vector
    combines its distance to each cluster's exemplars with the lambda at which
    its place in the condensed tree merges with each cluster, scaled by the
    probability that it belongs to any cluster at all.

    Parameters
    ----------
    clusterer : HDBSCAN
        A fitted clusterer with prediction data (``prediction_data=True`` or
        ``generate_prediction_data()``).

    Returns
    -------
    membership_vectors : array, shape (n_samples, n_clusters)
        The probability that point ``i`` belongs to cluster ``j``. Rows of
        non-finite training points are zero.
    """
    prediction_data = getattr(clusterer, "prediction_data_", None)
    if prediction_data is None:
        raise ValueError(
            "Clusterer does not have prediction data! Try fitting with"
            " prediction_data=True set, or run generate_prediction_data on the clusterer"
        )
    # columns follow labels_: epsilon can select the root yet label every point noise
    n_clusters = clusterer.labels_.max() + 1
    result = np.zeros((clusterer.labels_.shape[0], n_clusters), dtype=np.float64)
    if n_clusters == 0:
        return result

    placement = prediction_data.placement
    points = placement.fit_row_index
    if (placement.node_parent[points] == points).any():
        raise ValueError(
            "Some training points are missing from the condensed tree, as"
            " min_cluster_size=1 can leave them; soft clustering needs them all."
        )
    exemplar_labels = np.repeat(
        np.arange(n_clusters), [len(exemplars) for exemplars in prediction_data.exemplars]
    )
    distances = min_exemplar_distances(
        prediction_data.raw_data.astype(np.float64),
        np.vstack(prediction_data.exemplars).astype(np.float64),
        exemplar_labels,
        n_clusters,
    )
    with np.errstate(divide="ignore"):
        distance_vectors = 1.0 / distances
    # an exemplar is at distance zero from its own cluster
    distance_vectors[distances == 0] = np.finfo(np.float64).max / n_clusters
    distance_vectors /= distance_vectors.sum(axis=1, keepdims=True)

    heights = merge_heights(points, placement)
    point_lambda = placement.node_lambda[points]
    point_max_lambda = placement.max_child_lambda[placement.node_parent[points]] + 1e-8
    with np.errstate(divide="ignore"):
        scores = np.exp(-(point_max_lambda[:, None] / heights))
    outlier_vectors = np.exp(scores - np.nanmax(scores))
    outlier_vectors /= outlier_vectors.sum(axis=1, keepdims=True)

    nearest_cluster = placement.selected_clusters[heights.argmax(axis=1)]
    prob_in_some_cluster = heights.max(axis=1) / np.maximum(
        placement.max_child_lambda[nearest_cluster], point_lambda
    )

    vectors = distance_vectors * outlier_vectors
    vectors /= vectors.sum(axis=1, keepdims=True)
    result[points] = vectors * prob_in_some_cluster[:, None]
    return result
