"""
PyNNDescent-based approximate nearest neighbor support for fast_hdbscan.

This module provides functions to build KNN graphs using pynndescent for
arbitrary distance metrics, and to compute MSTs from those KNN graphs
via Kruskal's algorithm.

pynndescent is an optional dependency — it is only imported when a metric
other than 'euclidean' or 'precomputed' is used.

Rich Hakim & Leland McInnes 2026_03_06.
"""

import numpy as np
import scipy.sparse


def _check_pynndescent_available():
    """Check that pynndescent is installed, raise informative error if not."""
    try:
        import pynndescent  # noqa: F401

        return True
    except ImportError:
        raise ImportError(
            "The pynndescent package is required to use metrics other than "
            "'euclidean' and 'precomputed'. Install it with:\n"
            "    pip install pynndescent\n"
            "or:\n"
            "    pip install fast_hdbscan[nndescent]"
        )


def build_knn_graph(
    data, n_neighbors, metric="euclidean", metric_kwds=None, random_state=None
):
    """
    Build a KNN graph using pynndescent's NNDescent algorithm.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        The input data.
    n_neighbors : int
        Number of neighbors to find (excluding self).
    metric : str or callable
        Distance metric for pynndescent.
    metric_kwds : dict or None
        Additional keyword arguments for the metric function.
    random_state : int, RandomState, or None
        Random state for reproducibility.

    Returns
    -------
    nndescent_index : pynndescent.NNDescent
        The fitted NNDescent index (retained for potential graph connection).
    knn_indices : ndarray of shape (n_samples, n_neighbors), int32
        Indices of the k nearest neighbors (self excluded).
    knn_distances : ndarray of shape (n_samples, n_neighbors), float64
        Distances to the k nearest neighbors (self excluded).
    """
    from pynndescent import NNDescent

    if metric_kwds is None:
        metric_kwds = {}

    # +1 because NNDescent includes self as a neighbor
    index = NNDescent(
        data,
        metric=metric,
        metric_kwds=metric_kwds,
        n_neighbors=n_neighbors + 1,
        random_state=random_state,
        low_memory=True,
    )

    # neighbor_graph applies distance correction (e.g., alternative cosine → cosine)
    knn_indices, knn_distances = index.neighbor_graph

    # Strip self-neighbor (index 0 after deheap_sort)
    knn_indices = knn_indices[:, 1:].astype(np.int32)
    knn_distances = knn_distances[:, 1:].astype(np.float64)

    return index, knn_indices, knn_distances


def try_connect_knn_graph(knn_indices, knn_distances, nndescent_index):
    """
    Attempt to connect a disconnected KNN graph using pynndescent's
    graph connection utilities.

    Uses scipy connected_components to detect disconnection, then
    pynndescent.graph_utils.connect_graph to find real inter-component
    edges via the NNDescent search graph.

    Parameters
    ----------
    knn_indices : ndarray of shape (n, k), int32
    knn_distances : ndarray of shape (n, k), float64
    nndescent_index : pynndescent.NNDescent
        The fitted NNDescent index.

    Returns
    -------
    connected : bool
        True if connection was successful (now single component).
    adj_matrix : scipy.sparse.csr_matrix or None
        The connected adjacency matrix if successful, None otherwise.
    """
    from pynndescent.graph_utils import (
        adjacency_matrix_representation,
        connect_graph,
    )
    from scipy.sparse.csgraph import connected_components

    # Build adjacency matrix from KNN graph
    adj = adjacency_matrix_representation(knn_indices, knn_distances.astype(np.float32))

    n_components, _ = connected_components(adj)
    if n_components <= 1:
        return True, adj

    try:
        # Ensure NNDescent has the search graph prepared
        if not hasattr(nndescent_index, "_search_graph"):
            nndescent_index.prepare()

        connect_graph(adj, nndescent_index)

        n_components_after, _ = connected_components(adj)
        if n_components_after <= 1:
            return True, adj
        else:
            return False, adj
    except Exception:
        # If connect_graph fails for any reason, fall back gracefully
        return False, None


def _knn_graph_to_sparse(knn_indices, knn_distances, n):
    """
    Convert a KNN graph (dense arrays) into a symmetric CSR matrix.

    Parameters
    ----------
    knn_indices : ndarray of shape (n, k), int32
    knn_distances : ndarray of shape (n, k), float64
    n : int
        Number of data points.

    Returns
    -------
    sym_csr : scipy.sparse.csr_matrix, shape (n, n), float64 data
    """
    k = knn_indices.shape[1]
    rows = np.repeat(np.arange(n, dtype=np.int32), k)
    cols = knn_indices.ravel().astype(np.int32)
    data = knn_distances.ravel().astype(np.float64)

    # Filter out invalid neighbors (index == -1)
    valid = cols >= 0
    rows = rows[valid]
    cols = cols[valid]
    data = data[valid]

    # Build COO and symmetrize by taking minimum weight per undirected pair
    coo = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n, n))
    csr = coo.tocsr()

    # Symmetrize: for each (i,j), take min(w_ij, w_ji)
    csr_t = csr.T.tocsr()
    # Maximum of the two sparse matrices gives us the union of edges,
    # then we take minimum weights where both exist
    sym = csr.maximum(csr_t)  # union of sparsity patterns
    # For entries that exist in both, take the minimum
    both = csr.multiply(csr_t)  # intersection
    both_min = both.copy()
    # Where both exist, use minimum; where only one exists, use that value
    sym = sym - both + both_min.minimum(both.T)

    # Simpler approach: just symmetrize by taking max of patterns, min of values
    # Actually, the simplest correct approach:
    sym = csr.copy()
    sym_t = csr_t.copy()
    # Take minimum where both exist, otherwise take whichever exists
    sym = sym.maximum(sym_t)
    # Now overwrite with min where both have values
    overlap = csr.multiply(csr_t > 0)  # entries in csr where csr_t also has entry
    overlap_t = csr_t.multiply(csr > 0)  # entries in csr_t where csr also has entry
    min_overlap = overlap.minimum(overlap_t)
    # Replace overlap entries in sym with min values
    sym = sym - overlap.maximum(overlap_t) + min_overlap + min_overlap.T
    # This is getting complicated. Let's use the proven approach from precomputed.py

    # Use the simple vectorized approach: stack both directions, group, take min
    from .precomputed import _symmetrize_min_csr

    input_coo = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n, n))
    return _symmetrize_min_csr(input_coo)


def compute_mst_from_knn_graph(
    data,
    min_samples,
    metric="euclidean",
    metric_kwds=None,
    knn_k=None,
    random_state=None,
):
    """
    Compute HDBSCAN MST using a pynndescent KNN graph for arbitrary metrics.

    This is the main entry point for non-euclidean, non-precomputed metrics.
    Uses Kruskal's algorithm on the mutual-reachability-distance KNN graph.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
    min_samples : int
    metric : str or callable
    metric_kwds : dict or None
    knn_k : int or None
        Number of neighbors for KNN graph. If None, defaults to
        max(3 * min_samples, 15).
    random_state : int, RandomState, or None

    Returns
    -------
    mst_edges : ndarray of shape (n-1, 3), float64
    neighbors : ndarray of shape (n, k), int32
    core_distances : ndarray of shape (n,), float64
    """
    from .precomputed import bridge_forest_with_inf

    n = data.shape[0]

    # Determine knn_k
    if knn_k is None:
        knn_k = max(3 * min_samples, 15)
    knn_k = max(knn_k, min_samples)

    # Build KNN graph via pynndescent
    nndescent_index, knn_indices, knn_distances = build_knn_graph(
        data,
        knn_k,
        metric=metric,
        metric_kwds=metric_kwds,
        random_state=random_state,
    )

    # Compute core distances
    if min_samples <= 1:
        core_distances = np.zeros(n, dtype=np.float64)
    elif min_samples <= knn_k:
        core_distances = knn_distances[:, min_samples - 1].copy()
    else:
        core_distances = np.full(n, np.inf, dtype=np.float64)

    # Neighbor output (for model storage)
    neighbors_out = (
        knn_indices[:, :min_samples].copy()
        if min_samples > 0
        else knn_indices[:, :1].copy()
    )

    # Kruskal MST on MRD-weighted KNN edges
    mst_edges, n_components, component_labels = _kruskal_mst_from_knn(
        knn_indices,
        knn_distances,
        core_distances,
        n,
    )

    # Handle disconnected components by bridging with +inf edges
    if n_components > 1:
        mst_edges = bridge_forest_with_inf(mst_edges, component_labels, n)

    return mst_edges, neighbors_out, core_distances


def _kruskal_mst_from_knn(knn_indices, knn_distances, core_distances, n):
    """
    Run Kruskal MST on a KNN graph with MRD weights.

    Returns
    -------
    mst_edges : ndarray of shape (n_mst, 3)
    n_components : int
    component_labels : ndarray of shape (n,)
    """
    from .kruskal import (
        _build_mrd_edges_knn,
        _kruskal_core,
        _get_component_labels_jit,
    )

    # Build MRD edge list from KNN graph (parallel)
    weights, row_idx, col_idx = _build_mrd_edges_knn(
        knn_indices,
        knn_distances,
        core_distances,
    )

    # Filter self-loops and infinite weights
    valid = (row_idx != col_idx) & np.isfinite(weights)
    weights = weights[valid]
    row_idx = row_idx[valid]
    col_idx = col_idx[valid]

    # Global stable sort + Kruskal DSU
    sorted_order = np.argsort(weights, kind="stable").astype(np.int32)
    mst_edges, predecessors = _kruskal_core(weights, row_idx, col_idx, sorted_order, n)

    component_labels = _get_component_labels_jit(predecessors)
    n_components = int(np.unique(component_labels).shape[0])

    return mst_edges, n_components, component_labels
