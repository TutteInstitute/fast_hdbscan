"""
Sparse precomputed graph support

These functions are intended to allow for hdbscan to be called with a 
precomputed sparse graph (i.e., metric='precomputed' and X a scipy sparse 
matrix). They handle conversion of a scipy sparse pairwise weight matrix into 
the CoreGraph CSR format expected by the Borůvka MST routine in core_graph.py.

The contract:
- Input graphs are expected to be square, pairwise distances/weights 
  (non-negative, finite).
- Missing/sparse entries mean "no edge" (treated as +inf in MST).
- Explicit stored 0.0 edges are valid and preserved (no .eliminate_zeros() 
  calls).
- Asymmetric weights are symmetrized by undirected min: w_ij = min(w_ij, w_ji).
- Disconnected components (forest trunks) are bridged with deterministic +inf 
  edges.

Rich Hakim 2026_02_28. Editing and formatting with Claude Code Opus 4.6 .
"""

import numpy as np
import scipy.sparse

from .core_graph import CoreGraph, minimum_spanning_tree


def validate_precomputed_sparse_graph(X):
    """
    Validate a scipy sparse pairwise weight matrix for precomputed mode.

    Parameters
    ----------
    X : object
        Must be a scipy sparse matrix (CSR, CSC, or COO), square, with
        finite non-negative stored values.

    Raises
    ------
    ValueError
        If X is not sparse, not square, or contains invalid (negative/NaN/Inf) values.
    """
    if not scipy.sparse.issparse(X):
        raise ValueError(
            "When metric='precomputed', X must be a scipy sparse matrix "
            "(CSR, CSC, or COO). Got: %s" % type(X).__name__
        )
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError(
            "When metric='precomputed', X must be a square matrix. "
            "Got shape: %s" % str(X.shape)
        )
    coo = X.tocoo()
    data = coo.data
    if len(data) > 0:
        if np.any(~np.isfinite(data)):
            raise ValueError(
                "When metric='precomputed', all stored edge weights must be "
                "finite. Found NaN or Inf values."
            )
        if np.any(data < 0):
            raise ValueError(
                "When metric='precomputed', all stored edge weights must be "
                "non-negative. Found negative values."
            )


def extract_undirected_min_edges(X):
    """
    Extract canonical undirected edges from a (possibly asymmetric) sparse matrix.

    - Off-diagonal stored entries become undirected candidate edges.
    - Pairs (i,j) and (j,i) are symmetrized by taking the minimum weight.
    - Explicit stored 0.0 values are preserved (no eliminate_zeros() call).

    Parameters
    ----------
    X : scipy sparse matrix

    Returns
    -------
    edges : list of (u, v, w) tuples
        Canonical undirected edges with u < v, one per unordered pair.
    """
    coo = X.tocoo()
    rows, cols, data = coo.row, coo.col, coo.data

    # Use dict with canonical (u=min(i,j), v=max(i,j)) keys to accumulate min weight.
    # We iterate raw data (not .nonzero()) to preserve explicit zeros.
    edge_dict = {}
    for i, j, w in zip(rows, cols, data):
        if i == j:
            continue  # skip self-loops/diagonal
        u = int(min(i, j))
        v = int(max(i, j))
        key = (u, v)
        w = float(w)
        if key in edge_dict:
            # Asymmetric resolution: keep minimum weight
            if w < edge_dict[key]:
                edge_dict[key] = w
        else:
            edge_dict[key] = w

    return [(u, v, w) for (u, v), w in edge_dict.items()]


def build_adjacency_lists(n, undirected_edges):
    """
    Build per-node adjacency lists from a list of undirected edges.

    Parameters
    ----------
    n : int
        Number of nodes.
    undirected_edges : list of (u, v, w) tuples

    Returns
    -------
    adjacency : list of lists, length n
        adjacency[i] = list of (neighbor_idx, weight) pairs (both directions).
    """
    adjacency = [[] for _ in range(n)]
    for u, v, w in undirected_edges:
        adjacency[u].append((v, w))
        adjacency[v].append((u, w))
    return adjacency


def compute_sparse_core_distances(adjacency, min_samples):
    """
    Compute core distances for nodes from adjacency lists.

    - If min_samples <= 1: core_distance[i] = 0.0 for all i (fast path).
    - Else: core_distance[i] = distance to the min_samples-th nearest neighbor.
      Nodes with fewer than min_samples neighbors get core_distance = np.inf.

    Also returns a neighbors array shaped (n, k) where k = max(1, min_samples),
    filled with nearest-neighbor indices (padded with -1 when unavailable).

    Parameters
    ----------
    adjacency : list of lists
        Per-node (neighbor, weight) lists.
    min_samples : int

    Returns
    -------
    neighbors : np.ndarray, int32, shape (n, k)
    core_distances : np.ndarray, float64, shape (n,)
    """
    n = len(adjacency)
    k = max(1, min_samples)
    neighbors = np.full((n, k), -1, dtype=np.int32)
    core_distances = np.zeros(n, dtype=np.float64)

    if min_samples <= 1:
        # Core distance = 0; just populate top-k neighbors
        for i, nbrs in enumerate(adjacency):
            sorted_nbrs = sorted(nbrs, key=lambda x: x[1])
            for j, (nb_idx, _) in enumerate(sorted_nbrs[:k]):
                neighbors[i, j] = nb_idx
        return neighbors, core_distances

    for i, nbrs in enumerate(adjacency):
        if len(nbrs) == 0:
            core_distances[i] = np.inf
        else:
            sorted_nbrs = sorted(nbrs, key=lambda x: x[1])
            for j, (nb_idx, _) in enumerate(sorted_nbrs[:k]):
                neighbors[i, j] = nb_idx
            if len(sorted_nbrs) >= min_samples:
                core_distances[i] = sorted_nbrs[min_samples - 1][1]
            else:
                core_distances[i] = np.inf

    return neighbors, core_distances


def apply_mutual_reachability(undirected_edges, core_distances):
    """
    Convert base edge weights to mutual reachability distances (MRD).

    MRD(i, j) = max(core_dist[i], core_dist[j], base_weight(i, j))

    Parameters
    ----------
    undirected_edges : list of (u, v, w) tuples
    core_distances : np.ndarray, shape (n,)

    Returns
    -------
    mrd_edges : list of (u, v, mrd) triples
    """
    mrd_edges = []
    for u, v, w in undirected_edges:
        mrd = max(float(core_distances[u]), float(core_distances[v]), w)
        mrd_edges.append((u, v, mrd))
    return mrd_edges


def to_core_graph_arrays(n, mrd_edges):
    """
    Build a CoreGraph CSR structure from a list of MRD edges.

    Each undirected edge (u, v, mrd) is stored in both directions.
    Each row is sorted by (weight, neighbor_idx) for determinism — this ensures
    Borůvka's select_components always picks the minimum-weight outgoing edge.

    In the precomputed path, weights == distances (both are MRD values).

    Parameters
    ----------
    n : int
        Number of nodes.
    mrd_edges : list of (u, v, mrd) triples

    Returns
    -------
    CoreGraph namedtuple with arrays (weights, distances, indices, indptr).
    """
    # Build per-node sorted edge lists: (mrd, neighbor_idx)
    adj = [[] for _ in range(n)]
    for u, v, mrd in mrd_edges:
        adj[u].append((mrd, v))
        adj[v].append((mrd, u))

    # Sort each row by (weight, neighbor_idx) for determinism
    for i in range(n):
        adj[i].sort()

    # Build CSR indptr
    indptr = np.zeros(n + 1, dtype=np.int32)
    for i in range(n):
        indptr[i + 1] = indptr[i] + len(adj[i])
    nnz = int(indptr[-1])

    weights = np.empty(nnz, dtype=np.float32)
    distances = np.empty(nnz, dtype=np.float32)
    indices = np.full(nnz, -1, dtype=np.int32)

    for i in range(n):
        start = int(indptr[i])
        for j, (w, nb) in enumerate(adj[i]):
            weights[start + j] = w
            distances[start + j] = w  # weight == distance in precomputed path
            indices[start + j] = nb

    return CoreGraph(weights, distances, indices, indptr)


def bridge_forest_with_inf(edges, component_labels, n):
    """
    Connect multiple MST components with deterministic +inf bridge edges.

    If the graph is disconnected, Borůvka returns fewer than n-1 edges.
    This function appends +inf edges to complete the spanning tree, so that
    downstream cluster extraction (which expects exactly n-1 edges) works.

    Bridge edges are deterministic:
    - One representative per component = smallest node id in that component.
    - Representatives sorted ascending.
    - Bridge: (rep[i], rep[i+1], +inf) for consecutive representatives.

    Points connected only via +inf edges are merged at lambda=0 in the
    condensed tree and will be treated as noise by cluster extraction.

    Parameters
    ----------
    edges : np.ndarray, shape (k, 3) where k < n-1
    component_labels : np.ndarray, shape (n,), per-node component id
    n : int

    Returns
    -------
    edges : np.ndarray, shape (n-1, 3)
    """
    unique_components = np.unique(component_labels)
    if len(unique_components) == 1:
        return edges

    # Deterministic representative = smallest node id per component
    representatives = []
    for comp in unique_components:
        nodes_in_comp = np.where(component_labels == comp)[0]
        representatives.append(int(nodes_in_comp.min()))

    representatives.sort()

    # +inf bridge edges between consecutive representatives
    bridge_edges = np.array(
        [
            (float(representatives[i]), float(representatives[i + 1]), np.inf)
            for i in range(len(representatives) - 1)
        ],
        dtype=np.float64,
    )

    if edges.shape[0] == 0:
        return bridge_edges
    return np.vstack([edges, bridge_edges])


def compute_mst_from_precomputed_sparse(X, min_samples):
    """
    Compute the MST from a scipy sparse precomputed pairwise weight matrix.

    Orchestrates the full precomputed pipeline:
    1. Validate input.
    2. Extract undirected min-weight edges (preserving explicit zeros).
    3. Build adjacency lists.
    4. Compute core distances (and neighbor indices).
    5. Apply mutual reachability distances.
    6. Build CoreGraph CSR structure.
    7. Run Borůvka MST.
    8. Bridge any disconnected components with +inf edges.

    Parameters
    ----------
    X : scipy sparse matrix, shape (n, n)
        Pairwise distance/weight graph. Missing entries = no edge (+inf).
    min_samples : int
        Number of neighbors to use for core distance computation.

    Returns
    -------
    edges : np.ndarray, shape (n-1, 3), columns [src, dst, mrd_weight]
    neighbors : np.ndarray, int32, shape (n, k)
    core_distances : np.ndarray, float64, shape (n,)
    """
    validate_precomputed_sparse_graph(X)
    n = X.shape[0]

    undirected_edges = extract_undirected_min_edges(X)
    adjacency = build_adjacency_lists(n, undirected_edges)
    neighbors, core_distances = compute_sparse_core_distances(adjacency, min_samples)
    mrd_edges = apply_mutual_reachability(undirected_edges, core_distances)
    core_graph = to_core_graph_arrays(n, mrd_edges)

    # Borůvka MST; returns (n_components, component_labels, edges)
    n_components, component_labels, mst_edges = minimum_spanning_tree(core_graph)

    # Bridge any remaining disconnected components with +inf edges
    if n_components > 1:
        mst_edges = bridge_forest_with_inf(mst_edges, component_labels, n)

    return mst_edges, neighbors, core_distances
