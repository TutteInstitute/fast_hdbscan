"""
Test for the core_graph module.
"""

import pytest
import numba as nb
import numpy as np
from numba import set_num_threads
from sklearn.datasets import make_blobs

from fast_hdbscan.hdbscan import compute_minimum_spanning_tree
from fast_hdbscan.core_graph import (
    knn_mst_union,
    sort_by_lens,
    flatten_to_csr,
    core_graph_to_rec_array,
    core_graph_to_edge_list,
)

set_num_threads(1)

np.random.seed(10)
X, y = make_blobs(n_samples=50, random_state=10)
lens = np.random.normal(size=y.shape)
mst, indices, core_distances = compute_minimum_spanning_tree(X, min_samples=5)


@nb.njit()
def create_core_graph(mst, indices, core_distances, lens):
    # Remain in numba context to avoid warnings
    return flatten_to_csr(
        sort_by_lens(knn_mst_union(indices, core_distances, mst, lens))
    )


def test_create_core_graph():
    csr = create_core_graph(mst, indices, core_distances, lens)
    assert np.all(csr.distances > 0)
    for idx in range(len(csr.indptr) - 1):
        assert np.all(np.diff(csr.weights[csr.indptr[idx] : csr.indptr[idx + 1]]) >= 0)
        ws = np.maximum(
            lens[idx], lens[csr.indices[csr.indptr[idx] : csr.indptr[idx + 1]]]
        )
        assert np.allclose(ws, csr.weights[csr.indptr[idx] : csr.indptr[idx + 1]])


def test_core_graph_to_rec_array():
    csr = create_core_graph(mst, indices, core_distances, lens)
    rec_array = core_graph_to_rec_array(csr)
    assert rec_array.shape[0] == len(csr.distances)
    assert np.all(
        rec_array["parent"]
        == np.repeat(np.arange(len(csr.indptr) - 1), np.diff(csr.indptr))
    )
    assert np.all(rec_array["child"] == csr.indices)
    assert np.all(rec_array["weight"] == csr.weights)
    assert np.all(rec_array["distance"] == csr.distances)


def test_core_graph_to_edge_list():
    csr = create_core_graph(mst, indices, core_distances, lens)
    edgelist = core_graph_to_edge_list(csr)
    assert edgelist.shape[0] == len(csr.distances)
    assert np.all(
        edgelist[:, 0] == np.repeat(np.arange(len(csr.indptr) - 1), np.diff(csr.indptr))
    )
    assert np.all(edgelist[:, 1] == csr.indices)
    assert np.all(edgelist[:, 2] == csr.weights)
    assert np.all(edgelist[:, 3] == csr.distances)
