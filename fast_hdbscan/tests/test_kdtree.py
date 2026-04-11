import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.neighbors import KDTree

from fast_hdbscan.numba_kdtree import NumbaKDTree, kdtree_to_numba


def test_kdtree_to_numba_basic():
    X, _ = load_iris(return_X_y=True)
    sklearn_tree = KDTree(X)
    numba_tree = kdtree_to_numba(sklearn_tree)
    assert isinstance(numba_tree, NumbaKDTree)
    assert numba_tree.data.shape == X.shape
    assert len(numba_tree.idx_array) == X.shape[0]
    assert numba_tree.idx_start.dtype == np.intp
    assert numba_tree.idx_end.dtype == np.intp
    assert numba_tree.radius.dtype == np.float32
    assert numba_tree.is_leaf.dtype == np.bool_


def test_kdtree_to_numba_small():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    sklearn_tree = KDTree(X)
    numba_tree = kdtree_to_numba(sklearn_tree)
    assert numba_tree.data.shape == (4, 2)
    assert np.all(numba_tree.idx_start >= 0)
    assert np.all(numba_tree.idx_end >= numba_tree.idx_start)


def test_kdtree_to_numba_attribute_style():
    """Test fallback when node_data has attribute-style access (namedtuple-like)."""
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    sklearn_tree = KDTree(X)
    data, idx_array, node_data, node_bounds = sklearn_tree.get_arrays()

    class AttrNodeData:
        def __init__(self, nd):
            self.idx_start = nd["idx_start"]
            self.idx_end = nd["idx_end"]
            self.is_leaf = nd["is_leaf"]
            self.radius = nd["radius"]
            self.dtype = type("dtype", (), {"names": None})()

    attr_node_data = AttrNodeData(node_data)

    class FakeTree:
        def get_arrays(self):
            return data, idx_array, attr_node_data, node_bounds

    numba_tree = kdtree_to_numba(FakeTree())
    assert isinstance(numba_tree, NumbaKDTree)
    assert numba_tree.data.shape == (4, 2)
