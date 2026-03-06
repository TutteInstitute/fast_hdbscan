"""
Benchmarking suite for fast_hdbscan.

Measures wall-clock time, peak memory, and clustering outputs across:
  - Algorithm variants:  boruvka, kruskal_brute, kruskal_knn
  - Metrics:             euclidean, precomputed
  - Data generators:     blobs, moons, circles, uniform noise, single-cluster
  - Scaling axes:        n_samples, n_features, sparsity (k for KNN graph)
  - Parameter sweeps:    min_samples, min_cluster_size, knn_k
  - Time complexity:     scaling exponent estimation (n_repeats per size)
  - Density sweep:       precomputed graph density from 0.0001 to 1.0 (10x steps)

Usage:
    python benchmarking/run_benchmarks.py [--quick] [--output PATH]

    --quick   : run a small subset for smoke-testing (~1 min)
    --output  : JSON output path (default: benchmarking/results.json)

Rich Hakim 2026-03-05.  Benchmark design with Claude Code.
"""

import argparse
import gc
import json
import os
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler

# Ensure fast_hdbscan is importable from repo root.
_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo))

from fast_hdbscan import HDBSCAN
from fast_hdbscan.hdbscan import compute_minimum_spanning_tree


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def _make_blobs(n, d, rng):
    centers = max(3, d)  # at least 3 clusters
    X, _ = make_blobs(n_samples=n, n_features=d, centers=centers,
                      cluster_std=0.5, random_state=rng)
    return StandardScaler().fit_transform(X)


def _make_moons_2d(n, d, rng):
    X, _ = make_moons(n_samples=n, noise=0.05, random_state=rng)
    if d > 2:
        # Pad with Gaussian noise dimensions
        extra = rng.standard_normal((n, d - 2)) * 0.1
        X = np.hstack([X, extra])
    return StandardScaler().fit_transform(X)


def _make_circles_2d(n, d, rng):
    X, _ = make_circles(n_samples=n, noise=0.04, factor=0.5, random_state=rng)
    if d > 2:
        extra = rng.standard_normal((n, d - 2)) * 0.1
        X = np.hstack([X, extra])
    return StandardScaler().fit_transform(X)


def _make_uniform(n, d, rng):
    return rng.uniform(-1, 1, size=(n, d))


def _make_single_cluster(n, d, rng):
    return rng.standard_normal((n, d)) * 0.3


DATA_GENERATORS = {
    "blobs": _make_blobs,
    "moons": _make_moons_2d,
    "circles": _make_circles_2d,
    "uniform": _make_uniform,
    "single_cluster": _make_single_cluster,
}


# ---------------------------------------------------------------------------
# Sparse KNN graph builder (for precomputed benchmarks)
# ---------------------------------------------------------------------------

def build_knn_sparse(X, k_graph):
    """Build a symmetric sparse KNN distance graph from feature data."""
    import scipy.sparse

    n = X.shape[0]
    D = cdist(X, X).astype(np.float64)
    np.fill_diagonal(D, np.inf)  # exclude self

    # k nearest neighbors per point
    knn_idx = np.argpartition(D, kth=k_graph, axis=1)[:, :k_graph]
    row_arange = np.arange(n)[:, None]
    knn_dists = D[row_arange, knn_idx]

    rows = np.repeat(np.arange(n), k_graph)
    cols = knn_idx.ravel()
    data = knn_dists.ravel()

    G = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    # Symmetrize by min
    G_T = G.T.tocsr()
    G = G.minimum(G_T)
    G.eliminate_zeros()
    return G


def build_knn_sparse_fast(X, k_graph):
    """Build sparse KNN graph using fast_hdbscan's KD-tree (avoids O(n^2) cdist)."""
    import scipy.sparse
    from fast_hdbscan.numba_kdtree import build_kdtree, parallel_tree_query

    n = X.shape[0]
    tree = build_kdtree(X)
    dists_rdist, indices = parallel_tree_query(tree, tree.data, k=k_graph + 1,
                                               output_rdist=True)
    dists = np.sqrt(dists_rdist.astype(np.float64))
    # Drop self (column 0)
    knn_dists = dists[:, 1:]
    knn_idx = indices[:, 1:]

    rows = np.repeat(np.arange(n, dtype=np.int32), k_graph)
    cols = knn_idx.ravel().astype(np.int32)
    data = knn_dists.ravel()

    G = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    G_T = G.T.tocsr()
    G = G.minimum(G_T)
    G.eliminate_zeros()
    return G


# ---------------------------------------------------------------------------
# Sparse graph at arbitrary density (for density sweep)
# ---------------------------------------------------------------------------

def build_density_sparse(X, density, rng):
    """
    Build a sparse distance graph at a given edge density.

    density=1.0 means all n*(n-1)/2 undirected pairs are stored (full graph).
    density=0.001 means ~0.1% of pairs.  Edge weights are Euclidean distances.
    At low density the graph may be disconnected — HDBSCAN bridges with +inf.
    """
    import scipy.sparse

    n = X.shape[0]
    if density >= 1.0:
        # Full pairwise (stored as sparse for API consistency)
        D = cdist(X, X).astype(np.float64)
        np.fill_diagonal(D, 0.0)
        G = scipy.sparse.csr_matrix(D)
        G.eliminate_zeros()
        return G

    # Sample edges: for each upper-triangle pair, include with probability=density
    n_possible = n * (n - 1) // 2
    n_edges = max(n, int(n_possible * density))  # at least n edges for some connectivity
    n_edges = min(n_edges, n_possible)

    # Random sample of upper-triangle indices
    if n_edges >= n_possible * 0.5:
        # High density: generate all and subsample
        tri_r, tri_c = np.triu_indices(n, k=1)
        perm = rng.permutation(n_possible)[:n_edges]
        tri_r = tri_r[perm]
        tri_c = tri_c[perm]
    else:
        # Low density: sample random pairs
        pairs = set()
        while len(pairs) < n_edges:
            batch_size = min(n_edges - len(pairs), n_edges)
            a = rng.randint(0, n, size=batch_size)
            b = rng.randint(0, n, size=batch_size)
            for i, j in zip(a, b):
                if i != j:
                    pairs.add((min(i, j), max(i, j)))
        pairs = np.array(list(pairs))
        tri_r = pairs[:n_edges, 0]
        tri_c = pairs[:n_edges, 1]

    # Compute Euclidean distances for selected pairs
    diffs = X[tri_r] - X[tri_c]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))

    # Build symmetric sparse matrix
    rows = np.concatenate([tri_r, tri_c])
    cols = np.concatenate([tri_c, tri_r])
    data = np.concatenate([dists, dists])
    G = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    G.eliminate_zeros()
    return G


# ---------------------------------------------------------------------------
# JIT warmup
# ---------------------------------------------------------------------------

def warmup_jit():
    """
    Run every algorithm/metric combination on tiny data to trigger Numba JIT
    compilation before benchmarks start.  Without this, the first benchmark
    to use each code path pays a ~1s compilation penalty.
    """
    import scipy.sparse

    rng = np.random.RandomState(0)
    X_tiny = rng.standard_normal((30, 3)).astype(np.float32)

    # Build a tiny sparse graph for precomputed paths
    D = cdist(X_tiny, X_tiny).astype(np.float64)
    np.fill_diagonal(D, 0.0)
    G_tiny = scipy.sparse.csr_matrix(D)
    G_tiny.eliminate_zeros()

    combos = [
        # (data, metric, algorithm, extra_kw)
        (X_tiny, "euclidean", "boruvka", {}),
        (X_tiny, "euclidean", "kruskal", {"knn_k": 10}),
        (X_tiny, "euclidean", "kruskal", {"knn_k": None}),
        (G_tiny, "precomputed", "boruvka", {}),
        (G_tiny, "precomputed", "kruskal", {}),
    ]
    for data, metric, algo, kw in combos:
        compute_minimum_spanning_tree(
            data, min_samples=3, metric=metric, algorithm=algo, **kw
        )


# ---------------------------------------------------------------------------
# Timing + memory measurement
# ---------------------------------------------------------------------------

def _run_timed(fn, *args, **kwargs):
    """Run fn(*args, **kwargs) and return (result, elapsed_sec, peak_mem_bytes)."""
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak


# ---------------------------------------------------------------------------
# Individual benchmark runners
# ---------------------------------------------------------------------------

def bench_mst(X, algorithm, min_samples, metric="euclidean", knn_k=None,
              k_graph=20, n_repeats=1):
    """
    Benchmark compute_minimum_spanning_tree.

    Returns dict with timing, MST stats, and parameters.
    When n_repeats > 1, runs multiple times and reports median/all timings.
    """
    if metric == "precomputed":
        n = X.shape[0]
        # Build sparse graph (time this separately)
        if n <= 5000:
            G = build_knn_sparse(X, k_graph)
        else:
            G = build_knn_sparse_fast(X, k_graph)
        data_input = G
    else:
        data_input = X

    kw = dict(min_samples=min_samples, metric=metric, algorithm=algorithm)
    if algorithm == "kruskal" and metric == "euclidean":
        kw["knn_k"] = knn_k

    timings = []
    peak_mems = []
    for _ in range(n_repeats):
        (mst, neighbors, core_dists), elapsed, peak_mem = _run_timed(
            compute_minimum_spanning_tree, data_input, **kw
        )
        timings.append(elapsed)
        peak_mems.append(peak_mem)

    n_inf = int(np.sum(np.isinf(mst[:, 2])))
    finite_weights = mst[:, 2][np.isfinite(mst[:, 2])]

    result = {
        "elapsed_sec": round(float(np.median(timings)), 6),
        "peak_mem_bytes": int(np.max(peak_mems)),
        "mst_n_edges": int(mst.shape[0]),
        "mst_n_inf_edges": n_inf,
        "mst_total_weight": float(np.sum(finite_weights)) if len(finite_weights) else 0.0,
        "mst_max_weight": float(np.max(finite_weights)) if len(finite_weights) else 0.0,
        "core_dist_mean": float(np.mean(core_dists)),
        "neighbors_shape": list(neighbors.shape),
    }
    if n_repeats > 1:
        result["all_timings_sec"] = [round(t, 6) for t in timings]
        result["timing_std_sec"] = round(float(np.std(timings)), 6)
    return result


def bench_mst_with_sparse(G, algorithm, min_samples, n_repeats=1):
    """
    Benchmark compute_minimum_spanning_tree on a prebuilt sparse graph.

    Used by the density sweep where graph construction is separate from MST timing.
    """
    kw = dict(min_samples=min_samples, metric="precomputed", algorithm=algorithm)

    timings = []
    peak_mems = []
    for _ in range(n_repeats):
        (mst, neighbors, core_dists), elapsed, peak_mem = _run_timed(
            compute_minimum_spanning_tree, G, **kw
        )
        timings.append(elapsed)
        peak_mems.append(peak_mem)

    n_inf = int(np.sum(np.isinf(mst[:, 2])))
    finite_weights = mst[:, 2][np.isfinite(mst[:, 2])]

    result = {
        "elapsed_sec": round(float(np.median(timings)), 6),
        "peak_mem_bytes": int(np.max(peak_mems)),
        "mst_n_edges": int(mst.shape[0]),
        "mst_n_inf_edges": n_inf,
        "mst_total_weight": float(np.sum(finite_weights)) if len(finite_weights) else 0.0,
        "mst_max_weight": float(np.max(finite_weights)) if len(finite_weights) else 0.0,
        "core_dist_mean": float(np.mean(core_dists)),
        "neighbors_shape": list(neighbors.shape),
        "graph_nnz": int(G.nnz),
        "graph_density": round(G.nnz / (G.shape[0] ** 2), 6),
    }
    if n_repeats > 1:
        result["all_timings_sec"] = [round(t, 6) for t in timings]
        result["timing_std_sec"] = round(float(np.std(timings)), 6)
    return result


def bench_hdbscan(X, algorithm, min_samples, min_cluster_size,
                  metric="euclidean", knn_k=None, k_graph=20):
    """
    Benchmark full HDBSCAN pipeline (fit).

    Returns dict with timing, cluster stats, and parameters.
    """
    if metric == "precomputed":
        n = X.shape[0]
        if n <= 5000:
            G = build_knn_sparse(X, k_graph)
        else:
            G = build_knn_sparse_fast(X, k_graph)
        data_input = G
    else:
        data_input = X

    kw = dict(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        algorithm=algorithm,
        metric=metric,
    )
    if algorithm == "kruskal" and metric == "euclidean":
        kw["knn_k"] = knn_k

    model = HDBSCAN(**kw)
    _, elapsed, peak_mem = _run_timed(model.fit, data_input)

    labels = model.labels_
    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))

    return {
        "elapsed_sec": round(elapsed, 6),
        "peak_mem_bytes": peak_mem,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_fraction": round(n_noise / len(labels), 4) if len(labels) else 0.0,
    }


# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------

def _configs_scaling(quick=False):
    """Vary n_samples, fixed params.  Core benchmark for speed comparison."""
    sizes = ([200, 1000, 5000] if quick else
             [200, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000])
    configs = []
    for n in sizes:
        for algo_label, algo, knn_k in [
            ("boruvka", "boruvka", None),
            ("kruskal_knn_20", "kruskal", 20),
        ]:
            configs.append({
                "group": "scaling_euclidean",
                "name": f"{algo_label}_n{n}",
                "kind": "hdbscan",
                "data_gen": "blobs",
                "n_samples": n,
                "n_features": 10,
                "algorithm": algo,
                "metric": "euclidean",
                "min_samples": 5,
                "min_cluster_size": 15,
                "knn_k": knn_k,
            })

    # Brute-force Kruskal only for small n (O(n^2) memory)
    brute_sizes = [200, 1000] if quick else [200, 500, 1000, 2000, 5000, 7000]
    for n in brute_sizes:
        configs.append({
            "group": "scaling_euclidean",
            "name": f"kruskal_brute_n{n}",
            "kind": "hdbscan",
            "data_gen": "blobs",
            "n_samples": n,
            "n_features": 10,
            "algorithm": "kruskal",
            "metric": "euclidean",
            "min_samples": 5,
            "min_cluster_size": 15,
            "knn_k": None,
        })

    return configs


def _configs_scaling_precomputed(quick=False):
    """Scaling with precomputed sparse graphs."""
    sizes = ([200, 1000, 5000] if quick else
             [200, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000])
    configs = []
    for n in sizes:
        for algo_label, algo in [("boruvka", "boruvka"), ("kruskal", "kruskal")]:
            configs.append({
                "group": "scaling_precomputed",
                "name": f"{algo_label}_precomp_n{n}",
                "kind": "hdbscan",
                "data_gen": "blobs",
                "n_samples": n,
                "n_features": 10,
                "algorithm": algo,
                "metric": "precomputed",
                "min_samples": 5,
                "min_cluster_size": 15,
                "k_graph": 20,
            })
    return configs


def _configs_dimensionality(quick=False):
    """Vary n_features, fixed n."""
    dims = [2, 10, 50] if quick else [2, 5, 10, 25, 50, 100, 200]
    n = 2000 if quick else 20000
    configs = []
    for d in dims:
        for algo_label, algo, knn_k in [
            ("boruvka", "boruvka", None),
            ("kruskal_knn_20", "kruskal", 20),
        ]:
            configs.append({
                "group": "dimensionality",
                "name": f"{algo_label}_d{d}",
                "kind": "hdbscan",
                "data_gen": "blobs",
                "n_samples": n,
                "n_features": d,
                "algorithm": algo,
                "metric": "euclidean",
                "min_samples": 5,
                "min_cluster_size": 15,
                "knn_k": knn_k,
            })
    return configs


def _configs_data_shapes(quick=False):
    """Different data generators, fixed n and algorithm."""
    gens = ["blobs", "moons", "circles"] if quick else list(DATA_GENERATORS.keys())
    n = 2000 if quick else 20000
    configs = []
    for gen_name in gens:
        for algo_label, algo, knn_k in [
            ("boruvka", "boruvka", None),
            ("kruskal_knn_20", "kruskal", 20),
        ]:
            configs.append({
                "group": "data_shapes",
                "name": f"{algo_label}_{gen_name}",
                "kind": "hdbscan",
                "data_gen": gen_name,
                "n_samples": n,
                "n_features": 2,
                "algorithm": algo,
                "metric": "euclidean",
                "min_samples": 5,
                "min_cluster_size": 15,
                "knn_k": knn_k,
            })
    return configs


def _configs_min_samples_sweep(quick=False):
    """Sweep min_samples parameter."""
    ms_values = [1, 5, 20] if quick else [1, 3, 5, 10, 20, 50]
    n = 2000 if quick else 20000
    configs = []
    for ms in ms_values:
        for algo_label, algo, knn_k in [
            ("boruvka", "boruvka", None),
            ("kruskal_knn_20", "kruskal", 20),
        ]:
            configs.append({
                "group": "min_samples_sweep",
                "name": f"{algo_label}_ms{ms}",
                "kind": "hdbscan",
                "data_gen": "blobs",
                "n_samples": n,
                "n_features": 10,
                "algorithm": algo,
                "metric": "euclidean",
                "min_samples": ms,
                "min_cluster_size": 15,
                "knn_k": knn_k,
            })
    return configs


def _configs_knn_k_sweep(quick=False):
    """Sweep knn_k for Kruskal KNN, measure accuracy vs speed tradeoff."""
    k_values = [5, 15, 30] if quick else [5, 10, 15, 20, 30, 50, 100]
    n = 3000 if quick else 20000
    configs = []
    for k in k_values:
        configs.append({
            "group": "knn_k_sweep",
            "name": f"kruskal_knn_k{k}",
            "kind": "mst",
            "data_gen": "blobs",
            "n_samples": n,
            "n_features": 10,
            "algorithm": "kruskal",
            "metric": "euclidean",
            "min_samples": 5,
            "knn_k": k,
        })
    # Baselines: boruvka at same n, brute at smaller n (O(n^2) memory)
    configs.append({
        "group": "knn_k_sweep",
        "name": "boruvka_baseline",
        "kind": "mst",
        "data_gen": "blobs",
        "n_samples": n,
        "n_features": 10,
        "algorithm": "boruvka",
        "metric": "euclidean",
        "min_samples": 5,
        "knn_k": None,
    })
    brute_n = 3000 if quick else 5000  # brute is O(n^2), keep it feasible
    configs.append({
        "group": "knn_k_sweep",
        "name": "kruskal_brute_baseline",
        "kind": "mst",
        "data_gen": "blobs",
        "n_samples": brute_n,
        "n_features": 10,
        "algorithm": "kruskal",
        "metric": "euclidean",
        "min_samples": 5,
        "knn_k": None,
    })
    return configs


def _configs_sparsity_sweep(quick=False):
    """Sweep k_graph for precomputed sparse graphs."""
    k_values = [5, 15, 30] if quick else [5, 10, 15, 20, 30, 50, 100]
    n = 3000 if quick else 20000
    configs = []
    for k in k_values:
        for algo_label, algo in [("boruvka", "boruvka"), ("kruskal", "kruskal")]:
            configs.append({
                "group": "sparsity_sweep",
                "name": f"{algo_label}_kgraph{k}",
                "kind": "hdbscan",
                "data_gen": "blobs",
                "n_samples": n,
                "n_features": 10,
                "algorithm": algo,
                "metric": "precomputed",
                "min_samples": 5,
                "min_cluster_size": 15,
                "k_graph": k,
            })
    return configs


def _configs_min_cluster_size_sweep(quick=False):
    """Sweep min_cluster_size."""
    mcs_values = [5, 15, 50] if quick else [3, 5, 10, 15, 25, 50, 100]
    n = 3000 if quick else 20000
    configs = []
    for mcs in mcs_values:
        for algo_label, algo, knn_k in [
            ("boruvka", "boruvka", None),
            ("kruskal_knn_20", "kruskal", 20),
        ]:
            configs.append({
                "group": "min_cluster_size_sweep",
                "name": f"{algo_label}_mcs{mcs}",
                "kind": "hdbscan",
                "data_gen": "blobs",
                "n_samples": n,
                "n_features": 10,
                "algorithm": algo,
                "metric": "euclidean",
                "min_samples": 5,
                "min_cluster_size": mcs,
                "knn_k": knn_k,
            })
    return configs


def _configs_time_complexity(quick=False):
    """
    Time complexity estimation: run each algorithm/metric combo at many n values
    with multiple repeats per size.  Produces enough data points to fit a
    power-law T(n) ~ C * n^alpha and estimate alpha.
    """
    n_repeats = 3

    # Euclidean: boruvka and kruskal_knn scale as ~O(n log n) to O(n^1.5)
    euc_sizes = ([500, 2000, 8000] if quick else
                 [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000])
    configs = []
    for n in euc_sizes:
        for algo_label, algo, knn_k in [
            ("boruvka", "boruvka", None),
            ("kruskal_knn_20", "kruskal", 20),
        ]:
            configs.append({
                "group": "time_complexity_euclidean",
                "name": f"{algo_label}_n{n}",
                "kind": "mst",
                "data_gen": "blobs",
                "n_samples": n,
                "n_features": 10,
                "algorithm": algo,
                "metric": "euclidean",
                "min_samples": 5,
                "knn_k": knn_k,
                "n_repeats": n_repeats,
            })

    # Euclidean: kruskal_brute is O(n^2 log n) — only small n
    brute_sizes = [200, 500, 1000] if quick else [500, 1000, 2000, 3000, 5000, 7000]
    for n in brute_sizes:
        configs.append({
            "group": "time_complexity_euclidean",
            "name": f"kruskal_brute_n{n}",
            "kind": "mst",
            "data_gen": "blobs",
            "n_samples": n,
            "n_features": 10,
            "algorithm": "kruskal",
            "metric": "euclidean",
            "min_samples": 5,
            "knn_k": None,
            "n_repeats": n_repeats,
        })

    # Precomputed: boruvka and kruskal on KNN sparse graphs (k=20)
    pre_sizes = ([500, 2000, 8000] if quick else
                 [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000])
    for n in pre_sizes:
        for algo_label, algo in [("boruvka", "boruvka"), ("kruskal", "kruskal")]:
            configs.append({
                "group": "time_complexity_precomputed",
                "name": f"{algo_label}_n{n}",
                "kind": "mst",
                "data_gen": "blobs",
                "n_samples": n,
                "n_features": 10,
                "algorithm": algo,
                "metric": "precomputed",
                "min_samples": 5,
                "k_graph": 20,
                "n_repeats": n_repeats,
            })

    return configs


def _configs_precomputed_density(quick=False):
    """
    Vary graph density from 0.0001 to 1.0 in 10x steps on a large-ish graph.

    density = fraction of n*(n-1)/2 undirected pairs that have an edge.
    At density=0.0001 the graph is extremely sparse and likely disconnected.
    At density=1.0 the graph is fully connected (all pairwise distances stored).
    """
    n = 1000 if quick else 5000
    densities = [0.0001, 0.001, 0.01, 0.1, 1.0]
    n_repeats = 3

    configs = []
    for d in densities:
        for algo_label, algo in [("boruvka", "boruvka"), ("kruskal", "kruskal")]:
            configs.append({
                "group": "precomputed_density",
                "name": f"{algo_label}_density{d}",
                "kind": "mst_density",
                "data_gen": "blobs",
                "n_samples": n,
                "n_features": 10,
                "algorithm": algo,
                "metric": "precomputed",
                "min_samples": 5,
                "density": d,
                "n_repeats": n_repeats,
            })
    return configs


def _configs_mst_only(quick=False):
    """MST-only benchmarks (no clustering) to isolate MST cost."""
    sizes = ([500, 2000, 5000] if quick else
             [500, 1000, 2000, 5000, 10000, 25000, 50000, 100000])
    configs = []
    for n in sizes:
        for algo_label, algo, knn_k in [
            ("boruvka", "boruvka", None),
            ("kruskal_knn_20", "kruskal", 20),
        ]:
            configs.append({
                "group": "mst_only",
                "name": f"{algo_label}_n{n}",
                "kind": "mst",
                "data_gen": "blobs",
                "n_samples": n,
                "n_features": 10,
                "algorithm": algo,
                "metric": "euclidean",
                "min_samples": 5,
                "knn_k": knn_k,
            })
    return configs


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def collect_configs(quick=False):
    """Assemble all benchmark configurations."""
    configs = []
    configs += _configs_scaling(quick)
    configs += _configs_scaling_precomputed(quick)
    configs += _configs_dimensionality(quick)
    configs += _configs_data_shapes(quick)
    configs += _configs_min_samples_sweep(quick)
    configs += _configs_knn_k_sweep(quick)
    configs += _configs_sparsity_sweep(quick)
    configs += _configs_min_cluster_size_sweep(quick)
    configs += _configs_mst_only(quick)
    configs += _configs_time_complexity(quick)
    configs += _configs_precomputed_density(quick)
    return configs


def run_single(cfg, rng):
    """Run a single benchmark config and return a result dict."""
    gen_fn = DATA_GENERATORS[cfg["data_gen"]]
    X = gen_fn(cfg["n_samples"], cfg["n_features"], rng)
    n_repeats = cfg.get("n_repeats", 1)

    if cfg["kind"] == "mst_density":
        # Build sparse graph at given density, then benchmark MST only
        density = cfg["density"]
        G = build_density_sparse(X, density, rng)
        result = bench_mst_with_sparse(
            G,
            algorithm=cfg["algorithm"],
            min_samples=cfg["min_samples"],
            n_repeats=n_repeats,
        )
    elif cfg["kind"] == "mst":
        result = bench_mst(
            X,
            algorithm=cfg["algorithm"],
            min_samples=cfg["min_samples"],
            metric=cfg.get("metric", "euclidean"),
            knn_k=cfg.get("knn_k"),
            k_graph=cfg.get("k_graph", 20),
            n_repeats=n_repeats,
        )
    else:  # hdbscan
        result = bench_hdbscan(
            X,
            algorithm=cfg["algorithm"],
            min_samples=cfg["min_samples"],
            min_cluster_size=cfg.get("min_cluster_size", 15),
            metric=cfg.get("metric", "euclidean"),
            knn_k=cfg.get("knn_k"),
            k_graph=cfg.get("k_graph", 20),
        )

    result["config"] = cfg
    return result


def run_all(configs, seed=42):
    """Run all benchmarks and return list of result dicts."""
    rng = np.random.RandomState(seed)
    results = []
    total = len(configs)

    for i, cfg in enumerate(configs, 1):
        label = f"[{i}/{total}] {cfg['group']}/{cfg['name']}"
        print(f"  {label} ...", end=" ", flush=True)

        try:
            result = run_single(cfg, rng)
            elapsed = result["elapsed_sec"]
            print(f"{elapsed:.4f}s")
        except Exception as e:
            result = {"config": cfg, "error": str(e)}
            print(f"ERROR: {e}")

        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="fast_hdbscan benchmark suite")
    parser.add_argument("--quick", action="store_true",
                        help="Run a small subset for smoke-testing")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: benchmarking/results.json)")
    args = parser.parse_args()

    if args.output is None:
        out_dir = Path(__file__).resolve().parent
        args.output = str(out_dir / "results.json")

    configs = collect_configs(quick=args.quick)
    print(f"fast_hdbscan benchmark suite: {len(configs)} configurations")
    print(f"Mode: {'quick' if args.quick else 'full'}")
    print(f"Output: {args.output}")
    print()

    print("Warming up JIT (all algorithm/metric combos) ...", end=" ", flush=True)
    t_warmup = time.perf_counter()
    warmup_jit()
    print(f"done ({time.perf_counter() - t_warmup:.1f}s)")
    print()

    t_start = time.perf_counter()
    results = run_all(configs)
    t_total = time.perf_counter() - t_start

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "quick" if args.quick else "full",
        "n_benchmarks": len(results),
        "total_elapsed_sec": round(t_total, 2),
        "results": results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    n_errors = sum(1 for r in results if "error" in r)
    print(f"\nDone. {len(results)} benchmarks in {t_total:.1f}s "
          f"({n_errors} errors). Saved to {args.output}")


if __name__ == "__main__":
    main()
