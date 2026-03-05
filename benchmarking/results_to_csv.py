"""
Convert benchmarking results.json to a CSV spreadsheet for easy viewing.

Usage:
    python benchmarking/results_to_csv.py [INPUT_JSON] [OUTPUT_CSV]

Defaults:
    INPUT_JSON  = benchmarking/results.json
    OUTPUT_CSV  = benchmarking/results.csv
"""

import csv
import json
import sys
from pathlib import Path


def flatten_result(r):
    """Flatten a single result dict + nested config into a flat row dict."""
    row = {}
    cfg = r.get("config", {})

    # Config fields first
    row["group"] = cfg.get("group", "")
    row["name"] = cfg.get("name", "")
    row["kind"] = cfg.get("kind", "")
    row["data_gen"] = cfg.get("data_gen", "")
    row["n_samples"] = cfg.get("n_samples", "")
    row["n_features"] = cfg.get("n_features", "")
    row["algorithm"] = cfg.get("algorithm", "")
    row["metric"] = cfg.get("metric", "")
    row["min_samples"] = cfg.get("min_samples", "")
    row["min_cluster_size"] = cfg.get("min_cluster_size", "")
    row["knn_k"] = cfg.get("knn_k", "")
    row["k_graph"] = cfg.get("k_graph", "")
    row["density"] = cfg.get("density", "")
    row["n_repeats"] = cfg.get("n_repeats", "")

    # Result fields
    if "error" in r:
        row["error"] = r["error"]
        return row

    row["elapsed_sec"] = r.get("elapsed_sec", "")
    row["peak_mem_bytes"] = r.get("peak_mem_bytes", "")
    row["peak_mem_MB"] = round(r["peak_mem_bytes"] / 1e6, 2) if "peak_mem_bytes" in r else ""

    # MST fields (present for kind=mst and mst_density)
    row["mst_n_edges"] = r.get("mst_n_edges", "")
    row["mst_n_inf_edges"] = r.get("mst_n_inf_edges", "")
    row["mst_total_weight"] = r.get("mst_total_weight", "")
    row["mst_max_weight"] = r.get("mst_max_weight", "")
    row["core_dist_mean"] = r.get("core_dist_mean", "")
    row["neighbors_shape"] = str(r["neighbors_shape"]) if "neighbors_shape" in r else ""

    # HDBSCAN fields (present for kind=hdbscan)
    row["n_clusters"] = r.get("n_clusters", "")
    row["n_noise"] = r.get("n_noise", "")
    row["noise_fraction"] = r.get("noise_fraction", "")

    # Density sweep fields
    row["graph_nnz"] = r.get("graph_nnz", "")
    row["graph_density"] = r.get("graph_density", "")

    # Repeat timing fields
    row["all_timings_sec"] = str(r["all_timings_sec"]) if "all_timings_sec" in r else ""
    row["timing_std_sec"] = r.get("timing_std_sec", "")

    row["error"] = ""
    return row


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).resolve().parent / "results.json"
    )
    output_path = sys.argv[2] if len(sys.argv) > 2 else str(
        Path(__file__).resolve().parent / "results.csv"
    )

    with open(input_path) as f:
        data = json.load(f)

    rows = [flatten_result(r) for r in data["results"]]

    if not rows:
        print("No results to write.")
        return

    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"  Groups: {sorted(set(r['group'] for r in rows))}")


if __name__ == "__main__":
    main()
