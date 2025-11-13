import numpy as np
import numba

from .cluster_trees import (
    condense_tree,
    extract_leaves,
    get_cluster_label_vector,
    mst_to_linkage_tree,
)
from .numba_kdtree import kdtree_to_numba
from .boruvka import parallel_boruvka

from sklearn.neighbors import KDTree

from typing import NewType, List, Tuple, Dict, Optional

from .variables import NUMBA_CACHE

ClusterTree = NewType("ClusterTree", Dict[Tuple[int, int], List[Tuple[int, int]]])


@numba.njit(cache=NUMBA_CACHE)
def binary_search_for_n_clusters(
    uncondensed_tree, approx_n_clusters, n_samples, sample_weights=None,
):  # pragma: no cover
    lower_bound_min_cluster_size = 2
    upper_bound_min_cluster_size = n_samples // 2
    mid_min_cluster_size = int(
        round((lower_bound_min_cluster_size + upper_bound_min_cluster_size) / 2.0)
    )
    min_n_clusters = 0

    upper_tree = condense_tree(uncondensed_tree, upper_bound_min_cluster_size, sample_weights=sample_weights)
    leaves = extract_leaves(upper_tree)
    upper_n_clusters = len(leaves)

    lower_tree = condense_tree(uncondensed_tree, lower_bound_min_cluster_size, sample_weights=sample_weights)
    leaves = extract_leaves(lower_tree)
    lower_n_clusters = len(leaves)

    while upper_bound_min_cluster_size - lower_bound_min_cluster_size > 1:
        mid_min_cluster_size = int(
            round((lower_bound_min_cluster_size + upper_bound_min_cluster_size) / 2.0)
        )
        if (
            mid_min_cluster_size == lower_bound_min_cluster_size
            or mid_min_cluster_size == upper_bound_min_cluster_size
        ):
            break
        mid_tree = condense_tree(uncondensed_tree, mid_min_cluster_size, sample_weights=sample_weights)
        leaves = extract_leaves(mid_tree)
        mid_n_clusters = len(leaves)

        if mid_n_clusters < approx_n_clusters:
            upper_bound_min_cluster_size = mid_min_cluster_size
            upper_n_clusters = mid_n_clusters
        elif mid_n_clusters >= approx_n_clusters:
            lower_bound_min_cluster_size = mid_min_cluster_size
            lower_n_clusters = mid_n_clusters

    if abs(lower_n_clusters - approx_n_clusters) < abs(
        upper_n_clusters - approx_n_clusters
    ):
        lower_tree = condense_tree(uncondensed_tree, lower_bound_min_cluster_size, sample_weights=sample_weights)
        leaves = extract_leaves(lower_tree)
        clusters = get_cluster_label_vector(lower_tree, leaves, 0.0, n_samples)
        return leaves, clusters
    elif abs(lower_n_clusters - approx_n_clusters) > abs(
        upper_n_clusters - approx_n_clusters
    ):
        upper_tree = condense_tree(uncondensed_tree, upper_bound_min_cluster_size, sample_weights=sample_weights)
        leaves = extract_leaves(upper_tree)
        clusters = get_cluster_label_vector(upper_tree, leaves, 0.0, n_samples)
        return leaves, clusters
    else:
        lower_tree = condense_tree(uncondensed_tree, lower_bound_min_cluster_size, sample_weights=sample_weights)
        lower_leaves = extract_leaves(lower_tree)
        lower_clusters = get_cluster_label_vector(
            lower_tree, lower_leaves, 0.0, n_samples
        )
        upper_tree = condense_tree(uncondensed_tree, upper_bound_min_cluster_size, sample_weights=sample_weights)
        upper_leaves = extract_leaves(upper_tree)
        upper_clusters = get_cluster_label_vector(
            upper_tree, upper_leaves, 0.0, n_samples
        )

        if np.sum(lower_clusters >= 0) > np.sum(upper_clusters >= 0):
            return lower_leaves, lower_clusters
        else:
            return upper_leaves, upper_clusters


def build_raw_cluster_layers(
    data: np.ndarray,
    *,
    min_clusters: int = 3,
    min_samples: int = 5,
    base_min_cluster_size: int = 10,
    base_n_clusters: Optional[int] = None,
    next_cluster_size_quantile: float = 0.8,
    sample_weights: Optional[np.ndarray] = None,
    verbose=False,
) -> List[np.ndarray]:
    """
    Build hierarchical cluster layers from raw data using a KDTree and Boruvka's algorithm.

    Parameters
    ----------
    data : np.ndarray
        The input data array of shape (n_samples, n_features).
    min_clusters : int, optional
        The minimum number of clusters to form in each layer, by default 3.
    min_samples : int, optional
        The minimum number of samples in a cluster, by default 5.
    base_min_cluster_size : int, optional
        The initial minimum cluster size, by default 10.
    base_n_clusters : Optional[int], optional
        The initial number of clusters, by default None. If None, base_min_cluster_size is used.
        If not None, this value will override base_min_cluster_size.
    next_cluster_size_quantile : float, optional
        The quantile to determine the next minimum cluster size, by default 0.8.
    sample_weights : Optional[np.ndarray], optional
        The sample weights to use in the clustering, by default None.
    verbose : bool, optional
        Whether to print verbose output, by default False.

    Returns
    -------
    List[np.ndarray]
        A list of numpy arrays, each representing cluster labels for a layer.
    """
    n_samples = data.shape[0]
    cluster_layers = []
    min_cluster_size = base_min_cluster_size

    sklearn_tree = KDTree(data)
    numba_tree = kdtree_to_numba(sklearn_tree)
    edges, _, _ = parallel_boruvka(
        numba_tree, min_samples=min_cluster_size if min_samples is None else min_samples, sample_weights=sample_weights
    )
    sorted_mst = edges[np.argsort(edges.T[2])]
    uncondensed_tree = mst_to_linkage_tree(sorted_mst)
    if base_n_clusters is not None:
        leaves, clusters = binary_search_for_n_clusters(
            uncondensed_tree, base_n_clusters, n_samples=n_samples, sample_weights=sample_weights
        )
        cluster_sizes = np.bincount(clusters[clusters >= 0])
        min_cluster_size = np.min(cluster_sizes)
    else:
        new_tree = condense_tree(uncondensed_tree, base_min_cluster_size, sample_weights=sample_weights)
        leaves = extract_leaves(new_tree)
        clusters = get_cluster_label_vector(new_tree, leaves, 0.0, n_samples)

    n_clusters_in_layer = clusters.max() + 1

    while n_clusters_in_layer >= min_clusters:
        if verbose:
            print(f"Layer {len(cluster_layers)} found {n_clusters_in_layer} clusters")
        cluster_layers.append(clusters)
        cluster_sizes = np.bincount(clusters[clusters >= 0])
        next_min_cluster_size = int(
            np.quantile(cluster_sizes, next_cluster_size_quantile)
        )
        if next_min_cluster_size <= min_cluster_size + 1:
            break
        else:
            min_cluster_size = next_min_cluster_size
        new_tree = condense_tree(uncondensed_tree, min_cluster_size, sample_weights=sample_weights)
        leaves = extract_leaves(new_tree)
        clusters = get_cluster_label_vector(new_tree, leaves, 0.0, n_samples)
        n_clusters_in_layer = clusters.max() + 1

    return cluster_layers


@numba.njit(cache=NUMBA_CACHE)
def _build_cluster_tree(labels: np.ndarray) -> List[Tuple[int, int, int, int]]:

    mapping = [(-1, -1, -1, -1) for _ in range(0)]
    found = [set([-1]) for _ in range(len(labels))]
    for upper_layer in range(1, labels.shape[0]):
        upper_layer_unique_labels = np.unique(labels[upper_layer])
        for lower_layer in range(upper_layer - 1, -1, -1):
            upper_cluster_order = np.argsort(labels[upper_layer])
            cluster_groups = np.split(
                labels[lower_layer][upper_cluster_order],
                np.cumsum(np.bincount(labels[upper_layer] + 1))[:-1],
            )
            # If there is no noise we are off by one, and need to drop the first cluster group
            if len(cluster_groups) > upper_layer_unique_labels.shape[0]:
                cluster_groups = cluster_groups[1:]
            for i, label in enumerate(upper_layer_unique_labels):
                if label >= 0:
                    for child in cluster_groups[i]:
                        if child >= 0 and child not in found[lower_layer]:
                            mapping.append((upper_layer, label, lower_layer, child))
                            found[lower_layer].add(child)

    for lower_layer in range(labels.shape[0] - 1, -1, -1):
        for child in range(labels[lower_layer].max() + 1):
            if child >= 0 and child not in found[lower_layer]:
                mapping.append((labels.shape[0], 0, lower_layer, child))

    return mapping


def build_layer_cluster_tree(
    labels: List[np.ndarray],
) -> ClusterTree:
    """
    Builds a cluster tree from the given labels.

    Parameters
    ----------
    labels : List[np.ndarray]
        A list of numpy arrays where each array represents the labels of clusters at a specific layer.

    Returns
    -------
    ClusterTree
        A dictionary where the keys are tuples representing the parent cluster (layer, cluster index)
        and the values are lists of tuples representing the child clusters (layer, cluster index).
    """
    result = {}
    raw_mapping = _build_cluster_tree(np.vstack(labels))
    for parent_layer, parent_cluster, child_layer, child_cluster in raw_mapping:
        parent_name = (parent_layer, parent_cluster)
        if parent_name in result:
            result[parent_name].append((child_layer, child_cluster))
        else:
            result[parent_name] = [(child_layer, child_cluster)]
    return result