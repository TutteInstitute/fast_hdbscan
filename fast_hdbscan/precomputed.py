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

Rich Hakim 2026_02_28. Editing , formatting, and jit help with Claude Code Opus 4.6 .
"""

import numpy as np
import scipy.sparse
import numba

from .core_graph import CoreGraph, boruvka_mst
from .variables import NUMBA_CACHE


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


def _symmetrize_min_csr(X):
    """
    Convert any sparse matrix to a symmetric CSR by taking min weight per
    undirected pair. Diagonal entries are dropped. Explicit zeros are preserved.

    Uses numpy vectorized operations (lexsort + minimum.reduceat) to avoid
    Python-level loops over edges.

    Parameters
    ----------
    X : scipy sparse matrix, shape (n, n)

    Returns
    -------
    X_sym : scipy.sparse.csr_matrix, shape (n, n), symmetric, float64 data.
        Built via the (data, indices, indptr) CSR constructor to prevent
        scipy from silently eliminating explicit stored zeros.
    """
    coo = X.tocoo()
    rows, cols, data = coo.row, coo.col, coo.data
    n = X.shape[0]

    # Remove self-loops
    off_diag = rows != cols
    rows, cols, data = rows[off_diag], cols[off_diag], data[off_diag]

    if len(data) == 0:
        empty_indptr = np.zeros(n + 1, dtype=np.int32)
        return scipy.sparse.csr_matrix(
            (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int32), empty_indptr),
            shape=(n, n),
        )

    # Canonicalize to upper triangle: u = min(i,j), v = max(i,j)
    u = np.minimum(rows, cols)
    v = np.maximum(rows, cols)

    # Lexsort by (u, v) to group both orientations of each pair together
    order = np.lexsort((v.astype(np.int64), u.astype(np.int64)))
    u = u[order]
    v = v[order]
    data = data[order].astype(np.float64)

    # Locate the first occurrence of each unique (u, v) pair
    keys = u.astype(np.int64) * n + v.astype(np.int64)
    group_starts = np.concatenate(([0], np.where(keys[1:] != keys[:-1])[0] + 1))

    # Minimum weight per undirected pair
    min_data = np.minimum.reduceat(data, group_starts)
    u_out = u[group_starts].astype(np.int32)
    v_out = v[group_starts].astype(np.int32)
    m = len(u_out)

    # Mirror each canonical edge in both directions
    sym_rows = np.empty(2 * m, dtype=np.int32)
    sym_rows[:m] = u_out
    sym_rows[m:] = v_out
    sym_cols = np.empty(2 * m, dtype=np.int32)
    sym_cols[:m] = v_out
    sym_cols[m:] = u_out
    sym_data = np.concatenate([min_data, min_data])

    # Sort by row so we can build CSR indptr directly
    row_order = np.argsort(sym_rows, kind="stable")
    sym_rows = sym_rows[row_order]
    sym_cols = sym_cols[row_order]
    sym_data = sym_data[row_order]

    # Build CSR indptr without going through the COO->CSR path (which would
    # call sum_duplicates/eliminate_zeros and drop explicit zeros)
    counts = np.bincount(sym_rows, minlength=n).astype(np.int32)
    indptr = np.zeros(n + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(counts)

    return scipy.sparse.csr_matrix((sym_data, sym_cols, indptr), shape=(n, n))


@numba.njit(parallel=True, cache=NUMBA_CACHE)
def _core_distances_csr(data, indices, indptr, min_samples):
    """
    Compute per-node core distances from a symmetric CSR graph.

    For each node i the core distance is the base distance to its
    min_samples-th nearest neighbor. Nodes with fewer than min_samples
    stored neighbors get core_distance = inf. Runs in parallel over nodes.

    Parameters
    ----------
    data : float64 array, CSR edge weights
    indices : int32 array, CSR column indices
    indptr : int32 array, CSR row pointers
    min_samples : int

    Returns
    -------
    neighbors : int32 array, shape (n, k) where k = max(1, min_samples)
        Sorted nearest-neighbor indices; -1 where unavailable.
    core_distances : float64 array, shape (n,)
    """
    n = len(indptr) - 1
    k = max(1, min_samples)
    neighbors = np.full((n, k), -1, dtype=np.int32)
    core_distances = np.zeros(n, dtype=np.float64)

    for i in numba.prange(n):
        start = indptr[i]
        end = indptr[i + 1]
        deg = end - start

        if deg == 0:
            if min_samples > 1:
                core_distances[i] = np.inf
            continue

        row_data = data[start:end].copy()
        row_indices = indices[start:end].copy()

        order = np.argsort(row_data)

        fill = min(k, deg)
        for j in range(fill):
            neighbors[i, j] = row_indices[order[j]]

        if min_samples > 1:
            if deg >= min_samples:
                core_distances[i] = row_data[order[min_samples - 1]]
            else:
                core_distances[i] = np.inf

    return neighbors, core_distances


@numba.njit(parallel=True, cache=NUMBA_CACHE)
def _build_core_graph_csr(data, indices, indptr, core_distances):
    """
    Apply mutual reachability distances and build CoreGraph arrays.

    For each edge (i, j, w): MRD = max(core_dist[i], core_dist[j], w).
    Weights are stored as float32 (required by Borůvka) and sorted ascending
    within each row so that select_components picks the lightest outgoing edge.
    Ties in MRD are broken by ascending neighbor index for determinism.
    Runs in parallel over nodes.

    Parameters
    ----------
    data : float64 array, base edge weights (from symmetric CSR)
    indices : int32 array, CSR column indices
    indptr : int32 array, CSR row pointers
    core_distances : float64 array, shape (n,)

    Returns
    -------
    weights : float32 array (MRD values, for Borůvka comparison)
    distances : float32 array (same as weights in precomputed path)
    new_indices : int32 array (column indices reordered by ascending MRD)
    """
    n = len(indptr) - 1
    nnz = len(data)

    weights = np.empty(nnz, dtype=np.float32)
    distances = np.empty(nnz, dtype=np.float32)
    new_indices = np.empty(nnz, dtype=np.int32)

    for i in numba.prange(n):
        start = indptr[i]
        end = indptr[i + 1]
        cd_i = core_distances[i]
        deg = end - start

        if deg == 0:
            continue

        # Compute MRD for each neighbor of node i
        row_mrd = np.empty(deg, dtype=np.float32)
        for p in range(deg):
            nbr = indices[start + p]
            mrd = max(cd_i, core_distances[nbr], data[start + p])
            row_mrd[p] = np.float32(mrd)

        # Sort by (MRD, neighbor_idx): sort by neighbor_idx first, then
        # stable-sort by MRD so that ties preserve the neighbor-index order.
        # This matches the old (mrd, v) tuple-sort behaviour for determinism.
        row_nbr = indices[start:end].copy()
        idx_order = np.argsort(row_nbr)
        sorted_mrd = row_mrd[idx_order]
        sorted_nbr = row_nbr[idx_order]

        mrd_order = np.argsort(sorted_mrd, kind="mergesort")
        for p in range(deg):
            tgt = start + p
            sp = mrd_order[p]
            weights[tgt] = sorted_mrd[sp]
            distances[tgt] = sorted_mrd[sp]
            new_indices[tgt] = sorted_nbr[sp]

    return weights, distances, new_indices


@numba.njit(cache=NUMBA_CACHE)
def _patch_mst_weights(mst_edges, sym_data, sym_indices, sym_indptr, core_distances):
    """
    Restore exact float64 MRD weights to Borůvka's float32-rounded MST edges.

    Borůvka identifies correct MST topology even with float32 weights (~6e-5
    relative error), but the stored weight values are rounded. This function
    looks up the original base weight for each finite MST edge and recomputes
    MRD exactly in float64, preserving precise lambda values for the condensed
    tree — particularly important for cluster_selection_method='leaf'.

    Bridge edges (weight > 1e200) are left unchanged.

    Parameters
    ----------
    mst_edges : float64 array, shape (n-1, 3)
    sym_data : float64 array, base edge weights from symmetric CSR
    sym_indices : int32 array
    sym_indptr : int32 array
    core_distances : float64 array, shape (n,)

    Returns
    -------
    patched_weights : float64 array, shape (n-1,)
    """
    patched = mst_edges[:, 2].copy()
    for i in range(len(mst_edges)):
        if mst_edges[i, 2] > 1e200:  # bridge edge — leave as inf
            continue
        src = np.int32(mst_edges[i, 0])
        dst = np.int32(mst_edges[i, 1])
        base_w = 0.0
        for p in range(sym_indptr[src], sym_indptr[src + 1]):
            if sym_indices[p] == dst:
                base_w = sym_data[p]
                break
        patched[i] = max(core_distances[src], core_distances[dst], base_w)
    return patched


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


def unbridge_mst(edges):
    """
    Strip +inf bridge edges from a bridged MST, returning the MSF and
    per-node component labels.

    Inverse of ``bridge_forest_with_inf``: given an (n-1, 3) bridged MST,
    returns the finite edges and an array mapping each node to its connected
    component (integer label, 0-indexed, ordered by smallest node id).

    Parameters
    ----------
    edges : np.ndarray, shape (n-1, 3)
        Bridged MST with columns (src, dst, weight).  Bridge edges have
        weight == +inf.

    Returns
    -------
    finite_edges : np.ndarray, shape (m, 3)
        MSF edges (all finite weights).  m <= n-1.
    component_labels : np.ndarray, shape (n,), dtype int32
        Per-node component id.  Components are numbered 0..k-1 in order of
        their smallest node id.
    """
    n = int(edges.shape[0]) + 1
    finite_mask = np.isfinite(edges[:, 2])
    finite_edges = edges[finite_mask]

    # Build component labels via union-find on finite edges
    parent = np.arange(n, dtype=np.int32)

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(finite_edges.shape[0]):
        u, v = int(finite_edges[i, 0]), int(finite_edges[i, 1])
        ru, rv = _find(u), _find(v)
        if ru != rv:
            parent[rv] = ru

    # Resolve all roots and relabel 0..k-1 by smallest node id
    roots = np.array([_find(i) for i in range(n)], dtype=np.int32)
    _, inverse = np.unique(roots, return_inverse=True)
    component_labels = inverse.astype(np.int32)

    return finite_edges, component_labels


def compute_mst_from_precomputed_sparse(X, min_samples):
    """
    Compute the MST from a scipy sparse precomputed pairwise weight matrix.

    Orchestrates the full precomputed pipeline:
    1. Validate input.
    2. Symmetrize by min weight, drop diagonal (vectorized numpy).
    3. Compute core distances in parallel (Numba JIT).
    4. Build CoreGraph with MRD weights in parallel (Numba JIT).
    5. Run Borůvka MST (Numba JIT, float32 internally).
    6. Bridge disconnected components with +inf edges.
    7. Restore exact float64 MRD weights (Numba JIT).

    Borůvka is used for its O(m log n) efficiency.  It operates on float32
    weights internally (~6e-5 relative rounding), which does not change which
    edges are selected (MST topology is correct) but does shift the stored
    weight values.  After Borůvka, each MST edge weight is recomputed exactly
    in float64 from the original base weights and core distances, so that
    condensed-tree lambda = 1/weight is not degraded — this matters for
    cluster_selection_method='leaf' where small lambda differences can shift
    cluster boundaries.

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

    # 1. Symmetrize: min weight per undirected pair, no diagonal, explicit zeros preserved.
    X_sym = _symmetrize_min_csr(X)

    # _symmetrize_min_csr guarantees float64 data and int32 indices/indptr.
    # 2. Core distances and nearest-neighbor indices (parallel over nodes).
    neighbors, core_distances = _core_distances_csr(
        X_sym.data, X_sym.indices, X_sym.indptr, min_samples
    )

    # 3. Build CoreGraph with MRD weights, sorted per row (parallel over nodes).
    weights, distances, cg_indices = _build_core_graph_csr(
        X_sym.data, X_sym.indices, X_sym.indptr, core_distances
    )
    core_graph = CoreGraph(weights, distances, cg_indices, X_sym.indptr)

    # 4. Borůvka MST (float32 internally — fast).
    n_components, component_labels, mst_edges = boruvka_mst(core_graph)

    # 5. Bridge disconnected components with +inf edges.
    if n_components > 1:
        mst_edges = bridge_forest_with_inf(mst_edges, component_labels, n)

    # 6. Restore float64 precision: recompute MRD exactly for each MST edge.
    mst_weights = _patch_mst_weights(
        mst_edges, X_sym.data, X_sym.indices, X_sym.indptr, core_distances
    )
    mst_edges = np.column_stack([mst_edges[:, :2], mst_weights])

    return mst_edges, neighbors, core_distances


def compute_mst_from_precomputed_sparse_kruskal(
    X, min_samples, cannot_link=None, validate_cannot_link=True,
    cannot_link_groups=None,
):
    """
    Compute the MST from a scipy sparse precomputed pairwise weight matrix
    using Kruskal's algorithm.

    Compared to the Borůvka path, this function is simpler because Kruskal
    operates directly on float64 MRD weights — no float32 CoreGraph
    intermediate and no post-hoc weight-patching step are required.

    Pipeline:
    1. Validate input.
    2. Symmetrize by min weight, drop diagonal (vectorized numpy).
    3. Compute core distances in parallel (Numba JIT).
    4. Compute float64 MRD for every stored edge (Numba JIT).
    5. Global sort + Kruskal DSU (Numba JIT), with optional CL constraints.
    6. Bridge disconnected components with +inf edges.

    Parameters
    ----------
    X : scipy sparse matrix, shape (n, n)
        Pairwise distance/weight graph.  Missing entries = no edge (+inf).
    min_samples : int
        Number of neighbors to use for core distance computation.
    cannot_link : scipy sparse matrix or None
        Pairwise cannot-link constraint matrix.  Entry (i, j) != 0 means
        samples i and j must not co-cluster.  Mutually exclusive with
        ``cannot_link_groups``.
    validate_cannot_link : bool
        If True (default), validate and symmetrize the cannot-link matrix.
        Set to False when you know the input is already a symmetric CSR.
    cannot_link_groups : array-like of int or None
        Group-label cannot-link constraints.  ``int32[n]`` where samples
        sharing the same non-negative label cannot co-cluster; ``-1``
        means unconstrained.  O(n) alternative to the sparse
        ``cannot_link`` matrix for block-diagonal constraints.
        Mutually exclusive with ``cannot_link``.

    Returns
    -------
    edges : float64[:, :], shape (n - 1, 3)
        MST edges, columns ``[src, dst, mrd_weight]``.
    neighbors : int32[:, :], shape (n, k)
        Nearest-neighbor indices per sample.
    core_distances : float64[:], shape (n,)
        Core distance per sample.
    """
    from .kruskal import kruskal_mst_from_csr, _resolve_cl_params

    validate_precomputed_sparse_graph(X)
    n = X.shape[0]

    # 1. Symmetrize: min weight per undirected pair, no diagonal.
    X_sym = _symmetrize_min_csr(X)

    # 2. Core distances and nearest-neighbor indices (parallel over nodes).
    neighbors, core_distances = _core_distances_csr(
        X_sym.data, X_sym.indices, X_sym.indptr, min_samples
    )

    # 3. Resolve CL constraints (pairwise or group-label, mutually exclusive).
    cl = _resolve_cl_params(
        n, cannot_link=cannot_link,
        validate_cannot_link=validate_cannot_link,
        cannot_link_groups=cannot_link_groups,
    )

    # 4. Kruskal MST (float64 throughout — no patching needed).
    n_components, component_labels, mst_edges = kruskal_mst_from_csr(
        X_sym.data, X_sym.indices, X_sym.indptr, core_distances, cl=cl,
    )

    # 5. Bridge disconnected components with +inf edges.
    if n_components > 1:
        mst_edges = bridge_forest_with_inf(mst_edges, component_labels, n)

    return mst_edges, neighbors, core_distances
