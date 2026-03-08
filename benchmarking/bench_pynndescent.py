"""
Benchmark: pynndescent KNN-graph MST vs native parallel Boruvka for euclidean data.

Compares wall-clock time and clustering quality (ARI) between:
  1. Native Boruvka (KD-tree, euclidean only)
  2. PyNNDescent + Kruskal MST

Usage:
    python benchmarking/bench_pynndescent.py [--n_samples 5000] [--n_features 10] [--min_samples 10] [--n_clusters 5] [--repeats 3]
"""

import argparse
import time
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score


def run_native_boruvka(X, min_samples, min_cluster_size):
    """Native euclidean parallel Boruvka path."""
    from fast_hdbscan import HDBSCAN

    t0 = time.perf_counter()
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        algorithm="boruvka",
    ).fit(X)
    elapsed = time.perf_counter() - t0
    return model.labels_, elapsed


def run_native_kruskal(X, min_samples, min_cluster_size, knn_k):
    """Native euclidean Kruskal path (exact pairwise if knn_k is None)."""
    from fast_hdbscan import HDBSCAN

    t0 = time.perf_counter()
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        algorithm="kruskal",
        knn_k=knn_k,
    ).fit(X)
    elapsed = time.perf_counter() - t0
    return model.labels_, elapsed


def run_pynndescent_kruskal(X, min_samples, min_cluster_size, knn_k=None):
    """PyNNDescent KNN graph + Kruskal MST (arbitrary metric path, but euclidean here)."""
    from fast_hdbscan.nndescent import compute_mst_from_knn_graph
    from fast_hdbscan.hdbscan import clusters_from_spanning_tree

    t0 = time.perf_counter()
    mst, neighbors, core_dists = compute_mst_from_knn_graph(
        X,
        min_samples=min_samples,
        metric="euclidean",
        knn_k=knn_k,
        random_state=42,
    )
    labels, probs, _, _, _ = clusters_from_spanning_tree(
        mst,
        min_cluster_size=min_cluster_size,
    )
    elapsed = time.perf_counter() - t0
    return labels, elapsed


def run_pynndescent_cosine(X, min_samples, min_cluster_size, knn_k=None):
    """PyNNDescent + cosine metric (non-euclidean showcase)."""
    from fast_hdbscan import HDBSCAN

    # Normalise for cosine
    from sklearn.preprocessing import normalize

    X_norm = normalize(X, norm="l2")

    t0 = time.perf_counter()
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="cosine",
        knn_k=knn_k,
    ).fit(X_norm)
    elapsed = time.perf_counter() - t0
    return model.labels_, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark pynndescent vs native Boruvka"
    )
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--n_features", type=int, default=10)
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--min_samples", type=int, default=10)
    parser.add_argument("--min_cluster_size", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--knn_k", type=int, default=None)
    args = parser.parse_args()

    print(
        f"Generating data: n_samples={args.n_samples}, n_features={args.n_features}, "
        f"n_clusters={args.n_clusters}"
    )
    X, y_true = make_blobs(
        n_samples=args.n_samples,
        n_features=args.n_features,
        centers=args.n_clusters,
        random_state=42,
        cluster_std=1.0,
    )
    X = StandardScaler().fit_transform(X)

    knn_k = args.knn_k

    methods = {
        "Native Boruvka (euclidean)": lambda: run_native_boruvka(
            X, args.min_samples, args.min_cluster_size
        ),
        "Native Kruskal (euclidean)": lambda: run_native_kruskal(
            X, args.min_samples, args.min_cluster_size, knn_k
        ),
        "PyNNDescent + Kruskal (euclidean)": lambda: run_pynndescent_kruskal(
            X, args.min_samples, args.min_cluster_size, knn_k
        ),
        "PyNNDescent + Kruskal (cosine)": lambda: run_pynndescent_cosine(
            X, args.min_samples, args.min_cluster_size, knn_k
        ),
    }

    print(
        f"\nBenchmarking with min_samples={args.min_samples}, min_cluster_size={args.min_cluster_size}, "
        f"knn_k={knn_k}, repeats={args.repeats}"
    )
    print("=" * 85)

    # Warmup run (JIT compilation)
    print("\nWarmup run (JIT compilation)...")
    for name, fn in methods.items():
        try:
            fn()
            print(f"  {name}: OK")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    print("\n" + "=" * 85)
    print(f"{'Method':<42} {'Time (s)':>10} {'Std':>8} {'ARI':>8} {'#Clusters':>10}")
    print("-" * 85)

    reference_labels = None

    for name, fn in methods.items():
        times = []
        labels_last = None
        for _ in range(args.repeats):
            try:
                labels, elapsed = fn()
                times.append(elapsed)
                labels_last = labels
            except Exception as e:
                print(f"  {name}: FAILED ({e})")
                break

        if not times:
            continue

        mean_t = np.mean(times)
        std_t = np.std(times)
        n_clusters_found = len(set(labels_last)) - (1 if -1 in labels_last else 0)

        if reference_labels is None:
            reference_labels = labels_last
            ari = 1.0
        else:
            ari = adjusted_rand_score(reference_labels, labels_last)

        print(
            f"  {name:<40} {mean_t:>10.4f} {std_t:>8.4f} {ari:>8.3f} {n_clusters_found:>10}"
        )

    print("=" * 85)
    print("\nARI is computed against the first method (Native Boruvka).")


if __name__ == "__main__":
    main()
