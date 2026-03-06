"""
Kruskal's MST algorithm for fast_hdbscan.

Entry points:
  - kruskal_mst_from_feature_matrix : feature data (euclidean metric)
      knn_k=None  -> exact MST via full pairwise distances  (O(n^2) memory)
      knn_k=<int> -> approximate MST via KNN subgraph       (O(n*k) memory)
  - kruskal_mst_from_csr            : precomputed sparse distance matrix

Rich Hakim 2026_03_05.  JIT and algorithm design with Claude Code.
"""

import numpy as np
import numba

from .numba_kdtree import parallel_tree_query
from .variables import NUMBA_CACHE


# ---------------------------------------------------------------------------
# JIT primitives
# ---------------------------------------------------------------------------

@numba.njit(cache=NUMBA_CACHE)
def _kruskal_core(weights, row_indices, col_indices, sorted_order, n_verts):
    """
    Kruskal's algorithm with union-by-rank DSU and full path compression.

    Iterates over edges in ascending weight order (via *sorted_order*) and
    greedily adds each edge whose endpoints lie in different components.
    Self-loops are skipped.  Stops as soon as n_verts-1 edges have been
    selected (fully connected spanning tree).

    Parameters
    ----------
    weights      : float64[:], edge weights (not necessarily sorted)
    row_indices  : int32[:],   source vertex for edge i
    col_indices  : int32[:],   target vertex for edge i
    sorted_order : int32[:],   indices that sort *weights* ascending
    n_verts      : int,        number of vertices  (nodes 0 … n_verts-1)

    Returns
    -------
    mst_edges    : float64[:, :] shape (n_mst, 3)  — columns [src, dst, w]
                   n_mst < n_verts-1 when the input graph is disconnected.
    predecessors : int32[:] shape (n_verts,)
                   DSU parent array after algorithm; roots encode components.
    """
    predecessors = np.arange(n_verts, dtype=np.int32)
    rank = np.zeros(n_verts, dtype=np.int32)

    mst_edges = np.empty((n_verts - 1, 3), dtype=np.float64)
    n_mst = 0

    for idx in range(len(sorted_order)):
        if n_mst == n_verts - 1:
            break
        j = sorted_order[idx]
        u = row_indices[j]
        v = col_indices[j]

        if u == v:          # skip self-loops
            continue

        # ----- find root of u with full path compression -----
        root_u = u
        while predecessors[root_u] != root_u:
            root_u = predecessors[root_u]
        curr = u
        while curr != root_u:
            nxt = predecessors[curr]
            predecessors[curr] = root_u
            curr = nxt

        # ----- find root of v with full path compression -----
        root_v = v
        while predecessors[root_v] != root_v:
            root_v = predecessors[root_v]
        curr = v
        while curr != root_v:
            nxt = predecessors[curr]
            predecessors[curr] = root_v
            curr = nxt

        # ----- add edge if different components -----
        if root_u != root_v:
            mst_edges[n_mst, 0] = np.float64(u)
            mst_edges[n_mst, 1] = np.float64(v)
            mst_edges[n_mst, 2] = weights[j]
            n_mst += 1

            # union by rank
            if rank[root_u] > rank[root_v]:
                predecessors[root_v] = root_u
            elif rank[root_u] < rank[root_v]:
                predecessors[root_u] = root_v
            else:
                predecessors[root_v] = root_u
                rank[root_u] += 1

    return mst_edges[:n_mst], predecessors


@numba.njit(cache=NUMBA_CACHE)
def _get_component_labels_jit(predecessors):
    """
    Find the root (component id) for every node from DSU *predecessors*.

    Uses iterative path-following (no mutation of predecessors here so that
    the array can be reused by the caller if needed).
    """
    n = len(predecessors)
    labels = np.empty(n, dtype=np.int32)
    for i in range(n):
        root = predecessors[i]
        while predecessors[root] != root:
            root = predecessors[root]
        labels[i] = root
    return labels


@numba.njit(parallel=True, cache=NUMBA_CACHE)
def _compute_mrd_csr_flat(data, indices, indptr, core_distances):
    """
    Compute MRD = max(cd[i], cd[j], base_weight) for every CSR edge, in
    parallel over rows.  Returns a float64 array aligned with *data*.

    Parameters
    ----------
    data           : float64[:], stored edge weights in CSR format
    indices        : int32[:],   CSR column indices
    indptr         : int32[:],   CSR row pointers
    core_distances : float64[:], per-node core distances

    Returns
    -------
    mrd : float64[:], same length as *data*
    """
    n = len(indptr) - 1
    mrd = np.empty(len(data), dtype=np.float64)
    for i in numba.prange(n):
        cd_i = core_distances[i]
        start = indptr[i]
        end = indptr[i + 1]
        for p in range(start, end):
            j = indices[p]
            w = data[p]
            mrd_val = cd_i
            if core_distances[j] > mrd_val:
                mrd_val = core_distances[j]
            if w > mrd_val:
                mrd_val = w
            mrd[p] = mrd_val
    return mrd


@numba.njit(parallel=True, cache=NUMBA_CACHE)
def _build_mrd_edges_knn(knn_indices, knn_dists, core_distances):
    """
    Build a flat MRD-weighted edge list from a KNN graph, in parallel.

    For each (i, j) in the KNN graph:
        weight = max(core_distances[i], core_distances[j], knn_dists[i, j])

    Invalid neighbors (index == -1) are encoded as self-loops (i, i) with
    weight=inf so the caller can filter them with a single boolean mask.

    Parameters
    ----------
    knn_indices    : int32[:, :],   shape (n, k), KNN column indices
    knn_dists      : float64[:, :], shape (n, k), actual distances (not rdist)
    core_distances : float64[:],    shape (n,)

    Returns
    -------
    weights  : float64[:] shape (n*k,)
    row_idx  : int32[:]   shape (n*k,)
    col_idx  : int32[:]   shape (n*k,)
    """
    n = knn_indices.shape[0]
    k = knn_indices.shape[1]
    total = n * k

    weights = np.empty(total, dtype=np.float64)
    row_idx = np.empty(total, dtype=np.int32)
    col_idx = np.empty(total, dtype=np.int32)

    for i in numba.prange(n):
        cd_i = core_distances[i]
        for j_pos in range(k):
            edge_pos = i * k + j_pos
            j = knn_indices[i, j_pos]
            if j < 0:
                # Pad with a self-loop (will be filtered out)
                weights[edge_pos] = np.inf
                row_idx[edge_pos] = np.int32(i)
                col_idx[edge_pos] = np.int32(i)
            else:
                cd_j = core_distances[j]
                d = knn_dists[i, j_pos]
                mrd = cd_i
                if cd_j > mrd:
                    mrd = cd_j
                if d > mrd:
                    mrd = d
                weights[edge_pos] = mrd
                row_idx[edge_pos] = np.int32(i)
                col_idx[edge_pos] = np.int32(j)

    return weights, row_idx, col_idx


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def kruskal_mst_from_csr(sym_data, sym_indices, sym_indptr, core_distances):
    """
    Kruskal MST from a symmetric CSR graph with precomputed core distances.

    Computes mutual reachability distances (MRD) in float64 for every stored
    edge, sorts once globally, and applies Kruskal's DSU.  Because the entire
    pipeline stays in float64, no post-hoc weight-patching step is needed
    (unlike the Borůvka path which uses a float32 CoreGraph internally).

    This function is called by ``compute_mst_from_precomputed_sparse_kruskal``
    in ``precomputed.py``.

    Parameters
    ----------
    sym_data    : float64[:], CSR edge weights (symmetric, no diagonal)
    sym_indices : int32[:],   CSR column indices
    sym_indptr  : int32[:],   CSR row pointers (length n+1)
    core_distances : float64[:], shape (n,)

    Returns
    -------
    n_components    : int
    component_labels : int32[:] shape (n,) — root id per node
    mst_edges       : float64[:, :] shape (n_mst, 3) — [src, dst, mrd_weight]
                      n_mst == n-1 when graph is connected; caller bridges if not.
    """
    n = int(len(sym_indptr) - 1)

    # MRD for every stored edge (parallel over nodes)
    mrd_weights = _compute_mrd_csr_flat(
        sym_data, sym_indices, sym_indptr, core_distances
    )

    # Row index for each non-zero entry (vectorised)
    row_indices = np.repeat(
        np.arange(n, dtype=np.int32),
        np.diff(sym_indptr.astype(np.int64)),
    )

    # Global stable sort by MRD (deterministic tie-breaking)
    sorted_order = np.argsort(mrd_weights, kind="stable").astype(np.int32)

    # Kruskal DSU
    mst_edges, predecessors = _kruskal_core(
        mrd_weights,
        row_indices,
        sym_indices.astype(np.int32),
        sorted_order,
        n,
    )

    component_labels = _get_component_labels_jit(predecessors)
    n_components = int(np.unique(component_labels).shape[0])

    return n_components, component_labels, mst_edges


def kruskal_mst_from_feature_matrix(
    numba_tree,
    min_samples,
    knn_k=None,
    sample_weights=None,
    reproducible=False,  # noqa: ARG001  (Kruskal is always deterministic)
):
    """
    Kruskal MST for feature data.

    Parameters
    ----------
    numba_tree    : NumbaKDTree
    min_samples   : int
    knn_k         : int or None
        Number of nearest neighbors for the edge set.
        - None : compute exact MST via full pairwise distances (O(n^2) memory).
        - int  : approximate MST via KNN subgraph (O(n*k) memory, faster).
    sample_weights : float32[:] or None
        Weighted core distances (Boruvka-compatible).  Not supported with
        knn_k=None (full pairwise).
    reproducible  : bool
        Ignored; Kruskal is inherently deterministic.

    Returns
    -------
    mst_edges      : float64[:, :] shape (n-1, 3)
    neighbors      : int32[:, :]   shape (n, k), KNN indices excluding self
    core_distances : float64[:]    shape (n,)
    """
    if knn_k is None:
        if sample_weights is not None:
            raise NotImplementedError(
                "sample_weights with knn_k=None (full pairwise) is not supported. "
                "Use knn_k=<int> or algorithm='boruvka'."
            )
        return _kruskal_mst_brute(numba_tree, min_samples)
    return _kruskal_mst_knn(numba_tree, min_samples, knn_k, sample_weights)


def _kruskal_mst_brute(numba_tree, min_samples):
    """
    Exact Kruskal MST from full pairwise distances.

    Computes all n*(n-1)/2 pairwise Euclidean distances, applies mutual
    reachability, and runs Kruskal.  O(n^2) memory and O(n^2 log n) time.
    """
    from scipy.spatial.distance import cdist

    data = np.asarray(numba_tree.data)  # float32 from KD-tree
    n = data.shape[0]

    # Full pairwise Euclidean distances (float64 via cdist)
    D = cdist(data, data).astype(np.float64)

    # Core distances (column 0 of sorted row = self at distance 0).
    # min_samples=1 -> core_distance=0 by HDBSCAN convention (self-loop).
    # min_samples=k>1 -> distance to the k-th nearest neighbor (index k
    # in a row sorted ascending, since index 0 is self at distance 0).
    if min_samples <= 1:
        core_distances = np.zeros(n, dtype=np.float64)
    elif min_samples < n:
        core_distances = np.partition(D, kth=min_samples, axis=1)[
            :, min_samples
        ].copy()
    else:
        core_distances = np.full(n, np.inf, dtype=np.float64)

    # Nearest-neighbor indices for output (exclude self)
    k_out = min(max(1, min_samples), n - 1)
    k_part = min(k_out + 1, n)  # +1 to include self in partition
    part = np.argpartition(D, kth=k_part - 1, axis=1)[:, :k_part]
    row_arange = np.arange(n)[:, None]
    part_dists = D[row_arange, part]
    part_order = np.argsort(part_dists, axis=1)
    sorted_part = np.take_along_axis(part, part_order, axis=1)
    neighbors = sorted_part[:, 1 : k_out + 1].astype(np.int32)  # skip self

    # Upper-triangle MRD edge list (each undirected pair counted once)
    tri_r, tri_c = np.triu_indices(n, k=1)
    tri_r = tri_r.astype(np.int32)
    tri_c = tri_c.astype(np.int32)
    weights = np.maximum(
        np.maximum(core_distances[tri_r], core_distances[tri_c]),
        D[tri_r, tri_c],
    )

    sorted_order = np.argsort(weights, kind="stable").astype(np.int32)
    mst_edges, _predecessors = _kruskal_core(
        weights, tri_r, tri_c, sorted_order, n
    )

    # Full pairwise graph is always connected — no bridging needed.
    return mst_edges, neighbors, core_distances


def _kruskal_mst_knn(numba_tree, min_samples, knn_k, sample_weights=None):
    """
    Approximate Kruskal MST from a KNN subgraph.

    Builds a KNN graph with *knn_k* neighbors per point, applies mutual
    reachability distances, sorts globally, and runs Kruskal's DSU.
    Disconnected components are bridged with +inf edges.
    """
    from .boruvka import sample_weight_core_distance
    from .precomputed import bridge_forest_with_inf

    n = numba_tree.data.shape[0]

    if sample_weights is not None:
        mean_sw = float(np.mean(sample_weights))
        k_for_core = int(2 * min_samples / mean_sw)
        k_query = max(knn_k, k_for_core) + 1  # +1 for self
        distances_raw, neighbors_raw = parallel_tree_query(
            numba_tree, numba_tree.data, k=k_query
        )
        core_distances = sample_weight_core_distance(
            distances_raw, neighbors_raw, sample_weights, min_samples
        ).astype(np.float64)
        knn_dists = distances_raw[:, 1:].astype(np.float64)
        knn_indices = neighbors_raw[:, 1:].astype(np.int32)
        neighbors_out = neighbors_raw[:, 1:]

    elif min_samples > 1:
        k_query = max(knn_k, min_samples) + 1  # +1 for self
        distances_rdist, neighbors_raw = parallel_tree_query(
            numba_tree, numba_tree.data, k=k_query, output_rdist=True
        )
        distances = np.sqrt(distances_rdist.astype(np.float64))
        core_distances = distances[:, min_samples].copy()
        knn_dists = distances[:, 1:]
        knn_indices = neighbors_raw[:, 1:].astype(np.int32)
        neighbors_out = neighbors_raw[:, 1 : min_samples + 1]

    else:  # min_samples == 1 -> zero core distances
        k_query = knn_k + 1  # +1 for self
        distances_rdist, neighbors_raw = parallel_tree_query(
            numba_tree, numba_tree.data, k=k_query, output_rdist=True
        )
        distances = np.sqrt(distances_rdist.astype(np.float64))
        core_distances = np.zeros(n, dtype=np.float64)
        knn_dists = distances[:, 1:]
        knn_indices = neighbors_raw[:, 1:].astype(np.int32)
        neighbors_out = neighbors_raw[:, 1:2]

    # Build MRD edge list from KNN graph (parallel over nodes)
    weights, row_idx, col_idx = _build_mrd_edges_knn(
        knn_indices, knn_dists, core_distances
    )

    # Filter self-loops (encoded -1 neighbours) and infinite weights
    valid = (row_idx != col_idx) & np.isfinite(weights)
    weights = weights[valid]
    row_idx = row_idx[valid]
    col_idx = col_idx[valid]

    # Global stable sort by MRD (deterministic tie-breaking)
    sorted_order = np.argsort(weights, kind="stable").astype(np.int32)

    # Kruskal DSU
    mst_edges, predecessors = _kruskal_core(
        weights, row_idx, col_idx, sorted_order, n
    )

    component_labels = _get_component_labels_jit(predecessors)
    n_components = int(np.unique(component_labels).shape[0])

    if n_components > 1:
        mst_edges = bridge_forest_with_inf(mst_edges, component_labels, n)

    return mst_edges, neighbors_out, core_distances
