"""
Cannot-link constraint benchmarks for fast_hdbscan.

Compares three constraint modes of `fast_hdbscan()` with `algorithm='kruskal'`:
  1. No CL       — unconstrained baseline
  2. Group-label  — `cannot_link_groups=int32_array` (bitmask approach)
  3. Sparse matrix — `cannot_link=sparse_matrix` (pairwise approach)

Scaling axes:
  A. n_samples scaling (fixed n_groups=10)
  B. n_groups scaling  (fixed n=5000)
  C. CL density scaling (varies group size / samples-per-group ratio)

Measures wall-clock time and peak memory (tracemalloc) for each
`fast_hdbscan()` call.

Usage:
    python benchmarking/bench_cannot_link.py [--quick] [--output PATH]

    --quick   : smaller sizes for smoke-testing (~1 min)
    --output  : JSON output path (default: benchmarking/cl_results.json)

Rich Hakim 2026-03-17.  Benchmark design with Claude Code.
"""

import argparse
import gc
import json
import os
import signal
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.sparse
from sklearn.datasets import make_blobs

# Ensure fast_hdbscan is importable from repo root.
_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo))

from fast_hdbscan.hdbscan import fast_hdbscan


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_SAMPLES = 5
MIN_CLUSTER_SIZE = 10
KNN_K = 20
N_FEATURES = 5
N_CENTERS = 3
N_REPEATS = 3          # median of this many runs per config
SEED = 42
TIMEOUT_SEC = 60.0     # skip sparse-matrix runs predicted to exceed this
MAX_MEM_BYTES = 4 * (1024 ** 3)  # 4 GB memory guard for sparse matrix


# ---------------------------------------------------------------------------
# Interrupt handling for graceful partial output
# ---------------------------------------------------------------------------

_interrupted = False


def _sigint_handler(signum, frame):
    global _interrupted
    _interrupted = True
    print("\n[Ctrl-C] Finishing current run, will print partial results...")


signal.signal(signal.SIGINT, _sigint_handler)


# ---------------------------------------------------------------------------
# Data & constraint generation
# ---------------------------------------------------------------------------

def make_data(n_samples, rng):
    """Generate blob data: (n_samples, N_FEATURES), float32."""
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=N_FEATURES,
        centers=N_CENTERS,
        cluster_std=0.5,
        random_state=rng,
    )
    return X.astype(np.float32)


def make_group_labels(n_samples, n_groups):
    """
    Assign group labels round-robin: sample i gets group (i % n_groups).

    Returns int32 array of shape (n_samples,) with values in [0, n_groups).
    """
    return (np.arange(n_samples, dtype=np.int32) % n_groups)


def make_sparse_cl_from_groups(group_labels, n_groups):
    """
    Build a block-diagonal sparse CL matrix from group labels.

    For each group g, every pair (i, j) with group_labels[i]==g and
    group_labels[j]==g is a CL constraint.  The result is a symmetric
    CSR matrix with shape (n, n).

    Uses scipy.sparse.block_diag of dense intra-group blocks.
    """
    n = len(group_labels)

    # Collect indices per group
    group_members = [np.where(group_labels == g)[0] for g in range(n_groups)]

    # Build dense blocks: all-ones with zero diagonal for each group
    blocks = []
    block_indices = []  # row indices of each block's top-left corner
    for members in group_members:
        k = len(members)
        if k < 2:
            continue
        block = np.ones((k, k), dtype=np.float64)
        np.fill_diagonal(block, 0.0)
        blocks.append(scipy.sparse.csr_matrix(block))
        block_indices.append(members)

    if not blocks:
        return scipy.sparse.csr_matrix((n, n), dtype=np.float64)

    # block_diag gives us a matrix where block g occupies rows/cols for group g
    # but we need the original sample indices.  Build COO manually.
    rows_list = []
    cols_list = []
    for members, blk in zip(block_indices, blocks):
        blk_coo = blk.tocoo()
        rows_list.append(members[blk_coo.row])
        cols_list.append(members[blk_coo.col])

    all_rows = np.concatenate(rows_list)
    all_cols = np.concatenate(cols_list)
    all_data = np.ones(len(all_rows), dtype=np.float64)

    cl_matrix = scipy.sparse.csr_matrix(
        (all_data, (all_rows, all_cols)), shape=(n, n)
    )
    return cl_matrix


def estimate_sparse_cl_nnz(n_samples, n_groups):
    """
    Estimate the number of non-zeros in the CL sparse matrix for a
    round-robin group assignment with n_groups groups.

    Each group has ~n/g members, contributing ~(n/g)*(n/g - 1) entries.
    Total nnz ~ g * (n/g) * (n/g - 1) = n * (n/g - 1).
    """
    group_size = n_samples / n_groups
    nnz_per_group = group_size * (group_size - 1)
    return int(n_groups * nnz_per_group)


def estimate_sparse_cl_mem(n_samples, n_groups):
    """Rough estimate of CSR memory in bytes for the CL matrix."""
    nnz = estimate_sparse_cl_nnz(n_samples, n_groups)
    # CSR: data (float64) + indices (int32) + indptr (int32)
    mem = nnz * 8 + nnz * 4 + (n_samples + 1) * 4
    return mem


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


def run_hdbscan_timed(data, cannot_link=None, cannot_link_groups=None):
    """
    Run fast_hdbscan with Kruskal KNN and return (elapsed_sec, peak_mem_bytes).

    The full pipeline (MST + condensed tree + cluster extraction) is timed.
    """
    kw = dict(
        min_samples=MIN_SAMPLES,
        min_cluster_size=MIN_CLUSTER_SIZE,
        algorithm="kruskal",
        knn_k=KNN_K,
        cannot_link=cannot_link,
        cannot_link_groups=cannot_link_groups,
        validate_cannot_link=True,
    )
    _, elapsed, peak = _run_timed(fast_hdbscan, data, **kw)
    return elapsed, peak


# ---------------------------------------------------------------------------
# JIT warmup
# ---------------------------------------------------------------------------

def warmup_jit():
    """
    Trigger Numba JIT compilation for all three CL paths on tiny data.
    Without this the first benchmark pays a multi-second compilation penalty.
    """
    rng = np.random.RandomState(0)
    X_tiny = make_data(50, rng)

    # Unconstrained
    fast_hdbscan(
        X_tiny, min_samples=3, min_cluster_size=5,
        algorithm="kruskal", knn_k=10,
    )

    # Group-label CL (2 groups)
    gl = make_group_labels(50, 2)
    fast_hdbscan(
        X_tiny, min_samples=3, min_cluster_size=5,
        algorithm="kruskal", knn_k=10, cannot_link_groups=gl,
    )

    # Sparse-matrix CL
    cl_sparse = make_sparse_cl_from_groups(gl, 2)
    fast_hdbscan(
        X_tiny, min_samples=3, min_cluster_size=5,
        algorithm="kruskal", knn_k=10, cannot_link=cl_sparse,
    )


# ---------------------------------------------------------------------------
# Single configuration runner
# ---------------------------------------------------------------------------

def run_config(data, n_groups, n_repeats=N_REPEATS, skip_sparse=False):
    """
    Run all three CL modes for a given dataset and n_groups.

    Parameters
    ----------
    data         : float32[:, :], shape (n, d)
    n_groups     : int, number of CL groups
    n_repeats    : int, take median of this many runs
    skip_sparse  : bool, if True skip the sparse-matrix path

    Returns
    -------
    results : dict with keys 'no_cl', 'group_cl', 'sparse_cl', each containing
              {'time_sec': float, 'peak_mem_bytes': int} or None if skipped.
    """
    global _interrupted
    n = data.shape[0]
    group_labels = make_group_labels(n, n_groups)

    out = {"n_samples": n, "n_groups": n_groups}

    # --- No CL (baseline) ---
    if _interrupted:
        return out
    times, mems = [], []
    for _ in range(n_repeats):
        t, m = run_hdbscan_timed(data)
        times.append(t)
        mems.append(m)
    out["no_cl"] = {
        "time_sec": round(float(np.median(times)), 6),
        "peak_mem_bytes": int(np.max(mems)),
    }

    # --- Group-label CL ---
    if _interrupted:
        return out
    times, mems = [], []
    for _ in range(n_repeats):
        t, m = run_hdbscan_timed(data, cannot_link_groups=group_labels)
        times.append(t)
        mems.append(m)
    out["group_cl"] = {
        "time_sec": round(float(np.median(times)), 6),
        "peak_mem_bytes": int(np.max(mems)),
    }

    # --- Sparse-matrix CL ---
    if _interrupted:
        out["sparse_cl"] = None
        out["sparse_cl_skip_reason"] = "interrupted"
        return out

    if skip_sparse:
        out["sparse_cl"] = None
        out["sparse_cl_skip_reason"] = "estimated_too_large"
        return out

    cl_sparse = make_sparse_cl_from_groups(group_labels, n_groups)
    out["sparse_cl_nnz"] = int(cl_sparse.nnz)

    times, mems = [], []
    for _ in range(n_repeats):
        t, m = run_hdbscan_timed(data, cannot_link=cl_sparse)
        times.append(t)
        mems.append(m)
    out["sparse_cl"] = {
        "time_sec": round(float(np.median(times)), 6),
        "peak_mem_bytes": int(np.max(mems)),
    }

    return out


# ---------------------------------------------------------------------------
# Benchmark suites
# ---------------------------------------------------------------------------

def bench_n_samples_scaling(quick=False):
    """
    Axis A: vary n_samples with fixed n_groups=10.

    Tests how each CL approach scales with data size.
    """
    sizes = [500, 1000, 2000] if quick else [500, 1000, 2000, 5000, 10000, 20000]
    n_groups = 10
    rng = np.random.RandomState(SEED)

    results = []
    for n in sizes:
        if _interrupted:
            break

        # Decide whether sparse CL is feasible
        est_mem = estimate_sparse_cl_mem(n, n_groups)
        skip_sparse = est_mem > MAX_MEM_BYTES

        data = make_data(n, rng)
        label = f"n={n:>6d}, n_groups={n_groups}"
        print(f"  {label} ...", end=" ", flush=True)

        try:
            result = run_config(data, n_groups, skip_sparse=skip_sparse)
            t_no = result.get("no_cl", {}).get("time_sec", "?")
            t_gr = result.get("group_cl", {}).get("time_sec", "?")
            t_sp = result.get("sparse_cl", {}).get("time_sec", "skip") \
                if result.get("sparse_cl") else "skip"
            print(f"no_cl={t_no}s  group={t_gr}s  sparse={t_sp}s")
        except Exception as e:
            result = {"n_samples": n, "n_groups": n_groups, "error": str(e)}
            print(f"ERROR: {e}")

        results.append(result)

    return results


def bench_n_groups_scaling(quick=False):
    """
    Axis B: vary n_groups with fixed n=5000.

    Tests how the group-label approach scales with number of groups
    (bitmask width) and how the sparse-matrix approach scales with
    sparsity (fewer groups = denser blocks = more CL pairs).
    """
    n = 5000
    group_counts = [2, 5, 10, 50] if quick else [2, 5, 10, 20, 50, 100, 200]
    rng = np.random.RandomState(SEED)

    data = make_data(n, rng)
    results = []

    for ng in group_counts:
        if _interrupted:
            break

        est_mem = estimate_sparse_cl_mem(n, ng)
        skip_sparse = est_mem > MAX_MEM_BYTES

        label = f"n={n}, n_groups={ng:>4d}"
        print(f"  {label} ...", end=" ", flush=True)

        try:
            result = run_config(data, ng, skip_sparse=skip_sparse)
            t_no = result.get("no_cl", {}).get("time_sec", "?")
            t_gr = result.get("group_cl", {}).get("time_sec", "?")
            t_sp = result.get("sparse_cl", {}).get("time_sec", "skip") \
                if result.get("sparse_cl") else "skip"
            print(f"no_cl={t_no}s  group={t_gr}s  sparse={t_sp}s")
        except Exception as e:
            result = {"n_samples": n, "n_groups": ng, "error": str(e)}
            print(f"ERROR: {e}")

        results.append(result)

    return results


def bench_cl_density_scaling(quick=False):
    """
    Axis C: vary CL density by changing the number of groups.

    With n=5000 and round-robin assignment, fewer groups = larger per-group
    blocks = more CL pairs (denser CL matrix).  This axis highlights the
    O(CL_nnz) cost of the sparse-matrix path vs the O(1) bitmask check.

    n_groups:   CL pairs per group:  total CL pairs (approx):
       2         2500*2499 = 6.2M     12.5M
       5         1000*999  = 999k     5.0M
      10          500*499  = 249k     2.5M
      50          100*99   = 9.9k     495k
     100           50*49   = 2.4k     245k
     500           10*9    = 90       45k
    """
    n = 5000
    # From few groups (dense CL) to many groups (sparse CL)
    group_counts = [5, 10, 50, 200] if quick else [2, 5, 10, 20, 50, 100, 200, 500]
    rng = np.random.RandomState(SEED)

    data = make_data(n, rng)
    results = []

    for ng in group_counts:
        if _interrupted:
            break

        est_nnz = estimate_sparse_cl_nnz(n, ng)
        est_mem = estimate_sparse_cl_mem(n, ng)
        skip_sparse = est_mem > MAX_MEM_BYTES

        label = f"n={n}, n_groups={ng:>4d} (~{est_nnz:>10,d} CL pairs)"
        print(f"  {label} ...", end=" ", flush=True)

        try:
            result = run_config(data, ng, skip_sparse=skip_sparse)
            result["est_cl_pairs"] = est_nnz
            t_no = result.get("no_cl", {}).get("time_sec", "?")
            t_gr = result.get("group_cl", {}).get("time_sec", "?")
            t_sp = result.get("sparse_cl", {}).get("time_sec", "skip") \
                if result.get("sparse_cl") else "skip"
            print(f"no_cl={t_no}s  group={t_gr}s  sparse={t_sp}s")
        except Exception as e:
            result = {"n_samples": n, "n_groups": ng, "error": str(e)}
            print(f"ERROR: {e}")

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _fmt_time(t):
    """Format time in seconds with appropriate precision."""
    if t is None or t == "skip":
        return "skip"
    if isinstance(t, str):
        return t
    if t < 0.01:
        return f"{t*1000:.2f}ms"
    return f"{t:.4f}s"


def _fmt_mem(mem_bytes):
    """Format memory in human-readable units."""
    if mem_bytes is None:
        return "skip"
    if mem_bytes < 1024:
        return f"{mem_bytes}B"
    if mem_bytes < 1024 ** 2:
        return f"{mem_bytes / 1024:.1f}KB"
    if mem_bytes < 1024 ** 3:
        return f"{mem_bytes / (1024**2):.1f}MB"
    return f"{mem_bytes / (1024**3):.2f}GB"


def _speedup_str(baseline_t, comparison_t):
    """Return speedup string like '2.3x' or 'N/A'."""
    if baseline_t is None or comparison_t is None:
        return "N/A"
    if comparison_t == 0:
        return "inf"
    ratio = baseline_t / comparison_t
    return f"{ratio:.2f}x"


def print_table(title, results, axis_key="n_samples"):
    """
    Print a formatted comparison table.

    Parameters
    ----------
    title    : str, section header
    results  : list of dicts from run_config
    axis_key : str, key for the primary axis column
    """
    print()
    print(f"{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    # Header
    header = (
        f"{'Axis':>12s} | "
        f"{'No CL':>10s} {'Mem':>8s} | "
        f"{'Group CL':>10s} {'Mem':>8s} {'vs base':>8s} | "
        f"{'Sparse CL':>10s} {'Mem':>8s} {'vs base':>8s} {'vs grp':>8s}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        if "error" in r:
            axis_val = r.get(axis_key, "?")
            print(f"{str(axis_val):>12s} | ERROR: {r['error']}")
            continue

        axis_val = r.get(axis_key, r.get("n_groups", "?"))

        no_cl = r.get("no_cl")
        group_cl = r.get("group_cl")
        sparse_cl = r.get("sparse_cl")

        t_no = no_cl["time_sec"] if no_cl else None
        m_no = no_cl["peak_mem_bytes"] if no_cl else None
        t_gr = group_cl["time_sec"] if group_cl else None
        m_gr = group_cl["peak_mem_bytes"] if group_cl else None
        t_sp = sparse_cl["time_sec"] if sparse_cl else None
        m_sp = sparse_cl["peak_mem_bytes"] if sparse_cl else None

        # Speedup: sparse_time / group_time (how much faster group is vs sparse)
        sp_vs_base = _speedup_str(t_sp, t_no) if t_sp else "N/A"
        gr_vs_base = _speedup_str(t_gr, t_no) if t_gr else "N/A"
        sp_vs_grp = _speedup_str(t_sp, t_gr) if (t_sp and t_gr) else "N/A"

        row = (
            f"{str(axis_val):>12s} | "
            f"{_fmt_time(t_no):>10s} {_fmt_mem(m_no):>8s} | "
            f"{_fmt_time(t_gr):>10s} {_fmt_mem(m_gr):>8s} {gr_vs_base:>8s} | "
            f"{_fmt_time(t_sp):>10s} {_fmt_mem(m_sp):>8s} {sp_vs_base:>8s} {sp_vs_grp:>8s}"
        )
        print(row)

    print()


def print_density_table(results):
    """
    Print density-scaling table with CL pair counts.
    """
    print()
    print(f"{'=' * 95}")
    print(f"  CL Density Scaling (n={results[0].get('n_samples', '?')}, "
          f"varying n_groups)")
    print(f"{'=' * 95}")

    header = (
        f"{'n_groups':>8s} {'CL pairs':>12s} | "
        f"{'No CL':>10s} | "
        f"{'Group CL':>10s} {'overhead':>8s} | "
        f"{'Sparse CL':>10s} {'overhead':>8s} | "
        f"{'grp/sparse':>10s}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        if "error" in r:
            ng = r.get("n_groups", "?")
            print(f"{str(ng):>8s} {'':>12s} | ERROR: {r['error']}")
            continue

        ng = r.get("n_groups", "?")
        est_pairs = r.get("est_cl_pairs", r.get("sparse_cl_nnz", "?"))
        if isinstance(est_pairs, int):
            pairs_str = f"{est_pairs:>12,d}"
        else:
            pairs_str = f"{str(est_pairs):>12s}"

        no_cl = r.get("no_cl")
        group_cl = r.get("group_cl")
        sparse_cl = r.get("sparse_cl")

        t_no = no_cl["time_sec"] if no_cl else None
        t_gr = group_cl["time_sec"] if group_cl else None
        t_sp = sparse_cl["time_sec"] if sparse_cl else None

        # Overhead: how much slower than no_cl baseline
        gr_overhead = f"{t_gr / t_no:.2f}x" if (t_gr and t_no and t_no > 0) else "N/A"
        sp_overhead = f"{t_sp / t_no:.2f}x" if (t_sp and t_no and t_no > 0) else "N/A"

        # Speedup: sparse / group (>1 means group is faster)
        grp_vs_sp = _speedup_str(t_sp, t_gr) if (t_sp and t_gr) else "N/A"

        row = (
            f"{str(ng):>8s} {pairs_str} | "
            f"{_fmt_time(t_no):>10s} | "
            f"{_fmt_time(t_gr):>10s} {gr_overhead:>8s} | "
            f"{_fmt_time(t_sp):>10s} {sp_overhead:>8s} | "
            f"{grp_vs_sp:>10s}"
        )
        print(row)

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cannot-link constraint benchmarks for fast_hdbscan"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run smaller sizes for smoke-testing",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: benchmarking/cl_results.json)",
    )
    args = parser.parse_args()

    if args.output is None:
        out_dir = Path(__file__).resolve().parent
        args.output = str(out_dir / "cl_results.json")

    mode = "quick" if args.quick else "full"
    print(f"Cannot-link constraint benchmarks")
    print(f"Mode: {mode}")
    print(f"Output: {args.output}")
    print(f"Config: min_samples={MIN_SAMPLES}, min_cluster_size={MIN_CLUSTER_SIZE}, "
          f"knn_k={KNN_K}, d={N_FEATURES}, n_repeats={N_REPEATS}")
    print()

    # --- JIT warmup ---
    print("Warming up JIT (all CL paths) ...", end=" ", flush=True)
    t_warmup = time.perf_counter()
    warmup_jit()
    print(f"done ({time.perf_counter() - t_warmup:.1f}s)")
    print()

    all_results = {}
    t_start = time.perf_counter()

    # --- Axis A: n_samples scaling ---
    print("Axis A: n_samples scaling (n_groups=10)")
    print("-" * 50)
    results_a = bench_n_samples_scaling(quick=args.quick)
    all_results["n_samples_scaling"] = results_a
    print_table("n_samples Scaling (n_groups=10)", results_a, axis_key="n_samples")

    if not _interrupted:
        # --- Axis B: n_groups scaling ---
        print("Axis B: n_groups scaling (n=5000)")
        print("-" * 50)
        results_b = bench_n_groups_scaling(quick=args.quick)
        all_results["n_groups_scaling"] = results_b
        print_table("n_groups Scaling (n=5000)", results_b, axis_key="n_groups")

    if not _interrupted:
        # --- Axis C: CL density scaling ---
        print("Axis C: CL density scaling (n=5000)")
        print("-" * 50)
        results_c = bench_cl_density_scaling(quick=args.quick)
        all_results["cl_density_scaling"] = results_c
        print_density_table(results_c)

    t_total = time.perf_counter() - t_start

    # --- Save JSON ---
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "config": {
            "min_samples": MIN_SAMPLES,
            "min_cluster_size": MIN_CLUSTER_SIZE,
            "knn_k": KNN_K,
            "n_features": N_FEATURES,
            "n_centers": N_CENTERS,
            "n_repeats": N_REPEATS,
            "seed": SEED,
        },
        "total_elapsed_sec": round(t_total, 2),
        "results": all_results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    status = "INTERRUPTED (partial)" if _interrupted else "complete"
    n_total = sum(len(v) for v in all_results.values())
    print(f"Done ({status}). {n_total} configurations in {t_total:.1f}s. "
          f"Saved to {args.output}")


if __name__ == "__main__":
    main()
