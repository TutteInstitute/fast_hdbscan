"""
Kruskal's MST algorithm for fast_hdbscan.

Entry points:
  - kruskal_mst_from_feature_matrix : feature data (euclidean metric)
      knn_k=None  -> exact MST via full pairwise distances  (O(n^2) memory)
      knn_k=<int> -> approximate MST via KNN subgraph       (O(n*k) memory)
  - kruskal_mst_from_csr            : precomputed sparse distance matrix

Cannot-link constraint modes (mutually exclusive):
  - Pairwise sparse matrix (cannot_link): linked-list pool in DSU,
    O(CL_nnz) total conflict-check work.
  - Group-label array (cannot_link_groups): bitmask per DSU component,
    O(1) conflict check per merge for ≤64 groups, O(ceil(n_groups/64))
    for more.  Ideal for block-diagonal CL (same-session / same-batch).

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


@numba.njit(cache=NUMBA_CACHE)
def _kruskal_core_constrained(weights, row_indices, col_indices, sorted_order,
                               n_verts, cl_indices, cl_indptr):
    """
    Kruskal's MST with cannot-link constraints -> Minimum Spanning Forest.

    Edges that would merge two components containing a cannot-link vertex
    pair are skipped.  Per-component conflict lists are stored in a
    pre-allocated linked-list pool.  On each merge attempt the smaller
    conflict set is scanned (amortized O(M log n) total).  Conflict list
    concatenation during successful merges is O(1).

    Parameters
    ----------
    weights      : float64[:], edge weights
    row_indices  : int32[:],   source vertex per edge
    col_indices  : int32[:],   target vertex per edge
    sorted_order : int32[:],   argsort of weights ascending
    n_verts      : int
    cl_indices   : int32[:],   CSR column indices of symmetric CL graph
    cl_indptr    : int32[:],   CSR row pointers (length n_verts+1)

    Returns
    -------
    mst_edges    : float64[:, :] shape (n_mst, 3) -- [src, dst, weight]
    predecessors : int32[:]
    """
    predecessors = np.arange(n_verts, dtype=np.int32)
    rank = np.zeros(n_verts, dtype=np.int32)

    # --- linked-list pool for per-component conflict tracking ---
    M = len(cl_indices)
    pool_vertex = np.empty(M, dtype=np.int32)
    pool_next = np.full(M, -1, dtype=np.int32)

    comp_head = np.full(n_verts, -1, dtype=np.int32)
    comp_tail = np.full(n_verts, -1, dtype=np.int32)
    comp_csize = np.zeros(n_verts, dtype=np.int32)

    # Initialise from symmetric CL CSR (tail insertion)
    pool_idx = 0
    for i in range(n_verts):
        for p in range(cl_indptr[i], cl_indptr[i + 1]):
            pool_vertex[pool_idx] = cl_indices[p]
            if comp_tail[i] >= 0:
                pool_next[comp_tail[i]] = np.int32(pool_idx)
            else:
                comp_head[i] = np.int32(pool_idx)
            comp_tail[i] = np.int32(pool_idx)
            comp_csize[i] += 1
            pool_idx += 1

    mst_edges = np.empty((n_verts - 1, 3), dtype=np.float64)
    n_mst = 0

    for idx in range(len(sorted_order)):
        if n_mst == n_verts - 1:
            break
        j = sorted_order[idx]
        u = row_indices[j]
        v = col_indices[j]

        if u == v:
            continue

        # find root of u with full path compression
        root_u = u
        while predecessors[root_u] != root_u:
            root_u = predecessors[root_u]
        curr = u
        while curr != root_u:
            nxt = predecessors[curr]
            predecessors[curr] = root_u
            curr = nxt

        # find root of v with full path compression
        root_v = v
        while predecessors[root_v] != root_v:
            root_v = predecessors[root_v]
        curr = v
        while curr != root_v:
            nxt = predecessors[curr]
            predecessors[curr] = root_v
            curr = nxt

        if root_u == root_v:
            continue

        # --- conflict check: iterate the smaller conflict set ---
        # Fast path: if either component has no constraints, skip check
        if comp_csize[root_u] > 0 and comp_csize[root_v] > 0:
            if comp_csize[root_u] <= comp_csize[root_v]:
                small_root = root_u
                big_root = root_v
            else:
                small_root = root_v
                big_root = root_u

            conflict = False
            cur = comp_head[small_root]
            while cur >= 0:
                v_cl = pool_vertex[cur]
                # read-only find (no path compression — avoids dirtying
                # cache lines during the scan; outer-loop find on u/v
                # compresses these paths transitively over time)
                root_cl = v_cl
                while predecessors[root_cl] != root_cl:
                    root_cl = predecessors[root_cl]

                if root_cl == big_root:
                    conflict = True
                    break
                cur = pool_next[cur]

            if conflict:
                continue

        # --- accept edge ---
        mst_edges[n_mst, 0] = np.float64(u)
        mst_edges[n_mst, 1] = np.float64(v)
        mst_edges[n_mst, 2] = weights[j]
        n_mst += 1

        # union by rank
        if rank[root_u] > rank[root_v]:
            new_root = root_u
            old_root = root_v
        elif rank[root_u] < rank[root_v]:
            new_root = root_v
            old_root = root_u
        else:
            new_root = root_u
            old_root = root_v
            rank[new_root] += 1

        predecessors[old_root] = new_root

        # merge conflict lists: O(1) concatenation
        if comp_head[old_root] >= 0:
            if comp_tail[new_root] >= 0:
                pool_next[comp_tail[new_root]] = comp_head[old_root]
            else:
                comp_head[new_root] = comp_head[old_root]
            comp_tail[new_root] = comp_tail[old_root]
            comp_csize[new_root] += comp_csize[old_root]

        comp_head[old_root] = np.int32(-1)
        comp_tail[old_root] = np.int32(-1)
        comp_csize[old_root] = 0

    return mst_edges[:n_mst], predecessors


@numba.njit(cache=NUMBA_CACHE)
def _kruskal_core_group_constrained(weights, row_indices, col_indices,
                                     sorted_order, n_verts,
                                     group_labels, n_groups):
    """
    Kruskal's MST with group-label cannot-link constraints -> MSF.

    Two samples in the same group (same non-negative label) cannot end up
    in the same DSU component.  Label ``-1`` means unconstrained.

    Conflict tracking uses a **bitmask per DSU component**: each bit
    represents a group.  Conflict check is ``(mask_A & mask_B) != 0``;
    merge is ``mask_new = mask_A | mask_B``.  For ≤64 groups this is a
    single ``uint64`` AND/OR (1 CPU instruction); for more groups, a
    short inner loop over ``n_words = ceil(n_groups / 64)`` words.

    This is semantically equivalent to ``_kruskal_core_constrained`` with
    the full block-diagonal CL sparse matrix, but dramatically faster:
    O(1) conflict check per edge vs O(CL pairs in smaller component).

    Parameters
    ----------
    weights      : float64[:]
        Edge weights (not necessarily sorted).
    row_indices  : int32[:]
        Source vertex for each edge.
    col_indices  : int32[:]
        Target vertex for each edge.
    sorted_order : int32[:]
        Indices that sort *weights* ascending (from ``np.argsort``).
    n_verts      : int
        Number of vertices (nodes 0 … n_verts-1).
    group_labels : int32[:]
        Group label per vertex.  Values ≥ 0 identify a group; ``-1``
        means the vertex is unconstrained and may merge with any group.
    n_groups     : int
        Number of distinct non-negative groups.  Determines the bitmask
        width: ``n_words = ceil(n_groups / 64)``.

    Returns
    -------
    mst_edges : float64[:, :], shape (n_mst, 3)
        Accepted MST/MSF edges, columns ``[src, dst, weight]``.
        ``n_mst < n_verts - 1`` when constraints cause disconnection.
    predecessors : int32[:], shape (n_verts,)
        DSU parent array after the algorithm.  Roots encode components.
    """
    predecessors = np.arange(n_verts, dtype=np.int32)
    rank = np.zeros(n_verts, dtype=np.int32)

    ## Bitmask init: comp_mask[root, w] tracks which groups are present
    ## in the component rooted at `root`.  Bit (g % 64) in word (g // 64)
    ## is set if the component contains a vertex with group label g.
    n_words = (n_groups + 63) // 64  # ceil(n_groups / 64)
    comp_mask = np.zeros((n_verts, n_words), dtype=np.uint64)

    for i in range(n_verts):
        g = group_labels[i]
        if g >= 0:
            w = g // 64             # which uint64 word
            b = np.uint64(g % 64)   # which bit within that word
            comp_mask[i, w] = comp_mask[i, w] | (np.uint64(1) << b)

    mst_edges = np.empty((n_verts - 1, 3), dtype=np.float64)
    n_mst = 0

    for idx in range(len(sorted_order)):
        if n_mst == n_verts - 1:
            break
        j = sorted_order[idx]
        u = row_indices[j]
        v = col_indices[j]

        if u == v:  # skip self-loops
            continue

        ## Find root of u with full path compression
        root_u = u
        while predecessors[root_u] != root_u:
            root_u = predecessors[root_u]
        curr = u
        while curr != root_u:
            nxt = predecessors[curr]
            predecessors[curr] = root_u
            curr = nxt

        ## Find root of v with full path compression
        root_v = v
        while predecessors[root_v] != root_v:
            root_v = predecessors[root_v]
        curr = v
        while curr != root_v:
            nxt = predecessors[curr]
            predecessors[curr] = root_v
            curr = nxt

        if root_u == root_v:  # already in same component
            continue

        ## Conflict check: if any group bit is set in BOTH components,
        ## merging would put two same-group vertices together -> skip.
        conflict = False
        for w in range(n_words):
            if (comp_mask[root_u, w] & comp_mask[root_v, w]) != np.uint64(0):
                conflict = True
                break
        if conflict:
            continue

        ## Accept edge: add to MST/MSF
        mst_edges[n_mst, 0] = np.float64(u)
        mst_edges[n_mst, 1] = np.float64(v)
        mst_edges[n_mst, 2] = weights[j]
        n_mst += 1

        ## Union by rank
        if rank[root_u] > rank[root_v]:
            new_root = root_u
            old_root = root_v
        elif rank[root_u] < rank[root_v]:
            new_root = root_v
            old_root = root_u
        else:
            new_root = root_u
            old_root = root_v
            rank[new_root] += 1

        predecessors[old_root] = new_root

        ## Merge bitmasks: OR the old_root's mask into new_root's.  O(n_words).
        for w in range(n_words):
            comp_mask[new_root, w] = (
                comp_mask[new_root, w] | comp_mask[old_root, w]
            )

    return mst_edges[:n_mst], predecessors


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

def _validate_cannot_link(cannot_link, n, validate=True):
    """
    Validate and normalize a cannot-link constraint matrix.

    The input should be a **symmetric** sparse matrix where non-zero entry
    (i, j) marks a cannot-link constraint between vertices i and j.
    Diagonal entries are harmless and ignored by the algorithm.

    Parameters
    ----------
    cannot_link : scipy sparse matrix, shape (n, n)
        Symmetric sparse matrix of CL constraints.
    n : int
        Number of vertices.
    validate : bool
        If True (default), check shape/type and symmetrize via
        ``cl_csr + cl_csr.T``.  If False, skip all checks and
        symmetrization — the caller guarantees a symmetric CSR matrix
        of the correct shape.  This avoids an O(nnz) symmetrization
        step that dominates runtime for large constraint matrices.

    Returns
    -------
    cl_indices : int32[:], CSR column indices
    cl_indptr  : int32[:], CSR row pointers (length n+1)
    """
    import scipy.sparse

    if not validate:
        cl_csr = scipy.sparse.csr_matrix(cannot_link)
        if cl_csr.nnz == 0:
            return np.empty(0, dtype=np.int32), np.zeros(n + 1, dtype=np.int32)
        return cl_csr.indices.astype(np.int32), cl_csr.indptr.astype(np.int32)

    if not scipy.sparse.issparse(cannot_link):
        raise ValueError("cannot_link must be a scipy sparse matrix.")

    if cannot_link.shape != (n, n):
        raise ValueError(
            f"cannot_link shape {cannot_link.shape} does not match "
            f"data size ({n}, {n})."
        )

    cl_csr = scipy.sparse.csr_matrix(cannot_link)

    if cl_csr.nnz == 0:
        return np.empty(0, dtype=np.int32), np.zeros(n + 1, dtype=np.int32)

    # Symmetrize (handles upper-only, lower-only, or already symmetric).
    cl_sym = cl_csr + cl_csr.T

    return cl_sym.indices.astype(np.int32), cl_sym.indptr.astype(np.int32)


def _validate_cannot_link_groups(cannot_link_groups, n):
    """
    Validate and normalize a group-label cannot-link array.

    Each entry is a non-negative group index, or -1 for unconstrained.
    Samples sharing the same non-negative group label cannot co-cluster.

    Parameters
    ----------
    cannot_link_groups : array-like, shape (n,)
        Integer group labels.  -1 means unconstrained.
    n : int
        Expected length (number of samples).

    Returns
    -------
    group_labels : int32[:], shape (n,)
        Validated and cast group labels.
    n_groups : int
        Number of distinct non-negative groups (max label + 1).

    Raises
    ------
    ValueError
        If shape or values are invalid.
    """
    group_labels = np.asarray(cannot_link_groups, dtype=np.int32).ravel()

    ## Length check
    if group_labels.shape[0] != n:
        raise ValueError(
            f"cannot_link_groups length {group_labels.shape[0]} does not "
            f"match data size {n}."
        )

    ## Value range check: only -1 (unconstrained) and non-negative labels allowed
    if np.any(group_labels < -1):
        raise ValueError(
            "cannot_link_groups must contain only values >= -1. "
            "Found values < -1."
        )

    ## Early return: if all labels are -1, no constraints are active
    mask = group_labels >= 0
    if not np.any(mask):
        return group_labels, 0

    ## n_groups = max_label + 1 (labels need not be contiguous)
    n_groups = int(group_labels.max()) + 1
    return group_labels, n_groups


def _resolve_cl_params(n, cannot_link=None, validate_cannot_link=True,
                       cannot_link_groups=None):
    """
    Validate and normalize CL constraints into a uniform internal format.

    Accepts two mutually exclusive user-facing constraint modes and returns
    a dict of low-level arrays that ``_run_kruskal_core`` dispatches on.

    Parameters
    ----------
    n : int
        Number of vertices (samples).
    cannot_link : scipy sparse matrix or None
        Pairwise CL constraint matrix, shape (n, n).  Entry (i, j) != 0
        means samples i and j must not co-cluster.
    validate_cannot_link : bool
        If True, validate and symmetrize ``cannot_link``.
    cannot_link_groups : array-like or None
        Group-label CL array, shape (n,).  ``int32`` with values ≥ 0
        for group membership, ``-1`` for unconstrained.

    Returns
    -------
    cl : dict
        Uniform representation with four keys:

        - ``cl_indices`` (int32[:] or None): CSR column indices for pairwise CL.
        - ``cl_indptr``  (int32[:] or None): CSR row pointers for pairwise CL.
        - ``cl_group_labels`` (int32[:] or None): validated group labels.
        - ``cl_n_groups`` (int): number of groups (0 if unconstrained).

    Raises
    ------
    ValueError
        If both ``cannot_link`` and ``cannot_link_groups`` are provided.
    """
    if cannot_link is not None and cannot_link_groups is not None:
        raise ValueError(
            "cannot_link and cannot_link_groups are mutually exclusive. "
            "Provide one or the other, not both."
        )

    cl = dict(cl_indices=None, cl_indptr=None,
              cl_group_labels=None, cl_n_groups=0)

    if cannot_link is not None:
        cl["cl_indices"], cl["cl_indptr"] = _validate_cannot_link(
            cannot_link, n, validate=validate_cannot_link
        )
    elif cannot_link_groups is not None:
        cl["cl_group_labels"], cl["cl_n_groups"] = (
            _validate_cannot_link_groups(cannot_link_groups, n)
        )

    return cl


def _run_kruskal_core(weights, row_indices, col_indices, sorted_order, n, cl):
    """
    Dispatch to the appropriate Kruskal JIT core based on constraint mode.

    Selects one of three code paths:

    1. ``_kruskal_core_group_constrained`` — if group labels are active.
    2. ``_kruskal_core_constrained``       — if pairwise CSR CL is active.
    3. ``_kruskal_core``                   — unconstrained (no CL).

    Parameters
    ----------
    weights      : float64[:]
        MRD edge weights.
    row_indices  : int32[:]
        Source vertex per edge.
    col_indices  : int32[:]
        Target vertex per edge.
    sorted_order : int32[:]
        Indices sorting *weights* ascending (from ``np.argsort``).
    n            : int
        Number of vertices.
    cl           : dict
        Resolved CL parameters from ``_resolve_cl_params``.

    Returns
    -------
    mst_edges    : float64[:, 3]
        Accepted MST/MSF edges ``[src, dst, weight]``.
    predecessors : int32[:]
        DSU parent array after the algorithm.
    """
    if cl["cl_group_labels"] is not None and cl["cl_n_groups"] > 0:
        return _kruskal_core_group_constrained(
            weights, row_indices, col_indices, sorted_order, n,
            cl["cl_group_labels"], cl["cl_n_groups"],
        )
    if cl["cl_indices"] is not None and len(cl["cl_indices"]) > 0:
        return _kruskal_core_constrained(
            weights, row_indices, col_indices, sorted_order, n,
            cl["cl_indices"], cl["cl_indptr"],
        )
    return _kruskal_core(weights, row_indices, col_indices, sorted_order, n)


def _finish_mst(mst_edges, predecessors, n):
    """
    Post-Kruskal: resolve DSU components and bridge disconnected ones.

    When CL constraints (or a sparse KNN graph) produce a spanning forest
    with multiple components, this function inserts ``+inf`` weighted
    bridge edges between component representatives so that downstream
    condensed-tree construction sees a single connected tree.

    Parameters
    ----------
    mst_edges    : float64[:, 3]
        MST/MSF edges from a Kruskal core.
    predecessors : int32[:]
        DSU parent array from the same Kruskal core.
    n            : int
        Number of vertices.

    Returns
    -------
    mst_edges : float64[:, 3]
        Original edges if single-component; otherwise concatenated with
        ``(n_components - 1)`` bridge edges of weight ``+inf``.
    """
    component_labels = _get_component_labels_jit(predecessors)
    n_components = int(np.unique(component_labels).shape[0])
    if n_components > 1:
        from .precomputed import bridge_forest_with_inf
        mst_edges = bridge_forest_with_inf(mst_edges, component_labels, n)
    return mst_edges


def kruskal_mst_from_csr(sym_data, sym_indices, sym_indptr, core_distances,
                          cl=None, **cl_kw):
    """
    Kruskal MST from a symmetric CSR distance graph.

    Pipeline: compute MRD weights for every CSR edge (Numba parallel) →
    global stable argsort → Kruskal DSU (with optional CL constraints).
    Operates entirely in float64 — no CoreGraph intermediary and no
    post-hoc weight-patching step are required (unlike the Borůvka path).

    Called by ``compute_mst_from_precomputed_sparse_kruskal`` in
    ``precomputed.py``.

    Parameters
    ----------
    sym_data : float64[:]
        CSR edge weights of a symmetric distance graph (no diagonal).
    sym_indices : int32[:]
        CSR column indices.
    sym_indptr : int32[:]
        CSR row pointers, length ``n + 1``.
    core_distances : float64[:], shape (n,)
        Per-vertex core distance.
    cl : dict or None
        Pre-resolved CL parameters from ``_resolve_cl_params``.  If
        None, any remaining keyword arguments (``cannot_link``,
        ``validate_cannot_link``, ``cannot_link_groups``) are forwarded
        to ``_resolve_cl_params``.

    Returns
    -------
    n_components : int
        Number of connected components in the resulting MST/MSF.
    component_labels : int32[:], shape (n,)
        Component root id per vertex.
    mst_edges : float64[:, :], shape (n_mst, 3)
        MST edges ``[src, dst, mrd_weight]``.  ``n_mst == n - 1`` when
        fully connected; fewer when CL constraints fragment the graph.
    """
    n = int(len(sym_indptr) - 1)
    if cl is None:
        cl = _resolve_cl_params(n, **cl_kw)

    # MRD for every stored edge (parallel over nodes)
    mrd_weights = _compute_mrd_csr_flat(
        sym_data, sym_indices, sym_indptr, core_distances
    )

    # Row index for each non-zero entry (vectorised)
    row_indices = np.repeat(
        np.arange(n, dtype=np.int32),
        np.diff(sym_indptr.astype(np.int64)),
    )

    sorted_order = np.argsort(mrd_weights, kind="stable").astype(np.int32)

    mst_edges, predecessors = _run_kruskal_core(
        mrd_weights, row_indices, sym_indices.astype(np.int32),
        sorted_order, n, cl,
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
    cannot_link=None,
    validate_cannot_link=True,
    cannot_link_groups=None,
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
    cannot_link   : scipy sparse matrix or None
        Boolean sparse matrix of cannot-link constraints.  Must be symmetric.
        Diagonal entries are ignored.  Mutually exclusive with
        ``cannot_link_groups``.
    validate_cannot_link : bool
        If True (default), validate and symmetrize the cannot-link matrix.
    cannot_link_groups : array-like of int32 or None
        Group label per sample.  Samples sharing the same non-negative label
        cannot co-cluster.  -1 means unconstrained.  Mutually exclusive with
        ``cannot_link``.

    Returns
    -------
    mst_edges      : float64[:, :] shape (n-1, 3)
    neighbors      : int32[:, :]   shape (n, k), KNN indices excluding self
    core_distances : float64[:]    shape (n,)
    """
    n = numba_tree.data.shape[0]
    cl = _resolve_cl_params(
        n, cannot_link=cannot_link,
        validate_cannot_link=validate_cannot_link,
        cannot_link_groups=cannot_link_groups,
    )

    if knn_k is None:
        if sample_weights is not None:
            raise NotImplementedError(
                "sample_weights with knn_k=None (full pairwise) is not supported. "
                "Use knn_k=<int> or algorithm='boruvka'."
            )
        return _kruskal_mst_brute(numba_tree, min_samples, cl)
    return _kruskal_mst_knn(numba_tree, min_samples, knn_k, sample_weights, cl)


def _kruskal_mst_brute(numba_tree, min_samples, cl):
    """
    Exact Kruskal MST from full pairwise Euclidean distances.

    Computes all n*(n-1)/2 pairwise distances via ``cdist``, builds MRD
    weights for the upper triangle, sorts globally, and runs Kruskal's
    DSU.  O(n^2) memory and O(n^2 log n) time.

    Parameters
    ----------
    numba_tree  : NumbaKDTree
        KD-tree wrapping the feature data.
    min_samples : int
        k for core distance computation.
    cl          : dict
        Resolved CL parameters from ``_resolve_cl_params``.

    Returns
    -------
    mst_edges      : float64[:, :], shape (n - 1, 3)
    neighbors      : int32[:, :],   shape (n, k)
    core_distances : float64[:],    shape (n,)
    """
    from scipy.spatial.distance import cdist

    data = np.asarray(numba_tree.data)  # float32 from KD-tree
    n = data.shape[0]

    # Full pairwise Euclidean distances (float64 via cdist)
    D = cdist(data, data).astype(np.float64)

    # Core distances: min_samples=1 → 0; min_samples=k>1 → k-th NN distance
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

    mst_edges, predecessors = _run_kruskal_core(
        weights, tri_r, tri_c, sorted_order, n, cl,
    )
    mst_edges = _finish_mst(mst_edges, predecessors, n)
    return mst_edges, neighbors, core_distances


def _kruskal_mst_knn(numba_tree, min_samples, knn_k, sample_weights, cl):
    """
    Approximate Kruskal MST from a KNN subgraph.

    Pipeline: KD-tree KNN query → MRD edge list (Numba parallel) →
    filter invalid edges → global stable argsort → Kruskal DSU →
    bridge disconnected components with +inf edges.

    Parameters
    ----------
    numba_tree     : NumbaKDTree
        KD-tree wrapping the feature data.
    min_samples    : int
        k for core distance computation.
    knn_k          : int
        Number of nearest neighbors per point for the edge set.
    sample_weights : float32[:] or None
        Per-sample weights for weighted core distances.
    cl             : dict
        Resolved CL parameters from ``_resolve_cl_params``.

    Returns
    -------
    mst_edges      : float64[:, :], shape (n - 1, 3)
    neighbors_out  : int32[:, :],   shape (n, k)
    core_distances : float64[:],    shape (n,)
    """
    from .boruvka import sample_weight_core_distance

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

    sorted_order = np.argsort(weights, kind="stable").astype(np.int32)

    mst_edges, predecessors = _run_kruskal_core(
        weights, row_idx, col_idx, sorted_order, n, cl,
    )
    mst_edges = _finish_mst(mst_edges, predecessors, n)
    return mst_edges, neighbors_out, core_distances
