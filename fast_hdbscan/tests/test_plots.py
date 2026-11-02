"""Tests for the plotting classes ported from hdbscan.plots."""

import ast
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import fast_hdbscan.branches
import fast_hdbscan.hdbscan
import fast_hdbscan.plots
import fast_hdbscan.sub_clusters
from fast_hdbscan import HDBSCAN
from fast_hdbscan.plots import (
    CondensedTree,
    MinimumSpanningTree,
    SingleLinkageTree,
    _get_leaves,
)

X, _ = make_blobs(n_samples=200, random_state=10)
X = StandardScaler().fit_transform(X)


@pytest.fixture(scope="module")
def model():
    return HDBSCAN(min_cluster_size=10).fit(X)


def test_tree_attributes_use_fast_hdbscan_classes(model):
    assert isinstance(model.condensed_tree_, CondensedTree)
    assert isinstance(model.single_linkage_tree_, SingleLinkageTree)
    assert isinstance(model.minimum_spanning_tree_, MinimumSpanningTree)


@pytest.mark.parametrize(
    "module",
    [
        fast_hdbscan.plots,
        fast_hdbscan.hdbscan,
        fast_hdbscan.sub_clusters,
        fast_hdbscan.branches,
    ],
)
def test_module_does_not_import_hdbscan(module):
    tree = ast.parse(Path(module.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.add(node.module)
    assert not any(name.split(".")[0] == "hdbscan" for name in imported)


def same_partition(a, b):
    label_pairs = set(zip(a.tolist(), b.tolist()))
    labels_match_one_to_one = (
        len(label_pairs) == len(set(a.tolist())) == len(set(b.tolist()))
    )
    noise_matches = all((x == -1) == (y == -1) for x, y in label_pairs)
    return labels_match_one_to_one and noise_matches


@pytest.mark.parametrize("cut", [0.2, 0.5, 1.0])
@pytest.mark.parametrize("min_cluster_size", [2, 5, 20])
def test_get_clusters_matches_scipy_components(model, cut, min_cluster_size):
    from scipy.cluster.hierarchy import fcluster

    linkage = model._single_linkage_tree
    components = fcluster(linkage, t=cut, criterion="distance")
    sizes = np.bincount(components)
    expected = np.where(sizes[components] >= min_cluster_size, components, -1)

    labels = model.single_linkage_tree_.get_clusters(cut, min_cluster_size)
    assert np.issubdtype(labels.dtype, np.integer)
    assert same_partition(labels, expected)


def test_get_clusters_above_root_is_one_cluster(model):
    labels = model.single_linkage_tree_.get_clusters(np.inf, min_cluster_size=5)
    np.testing.assert_array_equal(labels, np.zeros(X.shape[0]))


def test_numpy_output_formats(model):
    assert model.condensed_tree_.to_numpy().shape[0] > 0
    assert model.single_linkage_tree_.to_numpy().shape == (X.shape[0] - 1, 4)
    assert model.minimum_spanning_tree_.to_numpy().shape == (X.shape[0] - 1, 3)


def test_pandas_output_formats(model):
    pytest.importorskip("pandas")
    assert len(model.condensed_tree_.to_pandas()) > 0
    assert len(model.single_linkage_tree_.to_pandas()) > 0
    assert len(model.minimum_spanning_tree_.to_pandas()) == X.shape[0] - 1


def test_networkx_output_formats(model):
    pytest.importorskip("networkx")
    assert model.condensed_tree_.to_networkx().number_of_nodes() > 0
    assert model.single_linkage_tree_.to_networkx().number_of_nodes() > 0
    assert model.minimum_spanning_tree_.to_networkx().number_of_nodes() == X.shape[0]


def test_condensed_tree_get_plot_data_selected_clusters(model):
    data = model.condensed_tree_.get_plot_data()
    assert set(data) == {
        "bar_centers",
        "bar_tops",
        "bar_bottoms",
        "bar_widths",
        "line_xs",
        "line_ys",
        "cluster_bounds",
    }
    selected = model.condensed_tree_._select_clusters()
    assert len(selected) == model.labels_.max() + 1


def test_condensed_tree_plot(model, plt):
    model.condensed_tree_.plot(
        select_clusters=True,
        label_clusters=True,
        selection_palette=("r", "g", "b"),
        cmap="Reds",
    )
    model.condensed_tree_.plot(log_size=True, colorbar=False, cmap="none")


def test_single_linkage_tree_plot(model, plt):
    model.single_linkage_tree_.plot(cmap="Reds")
    model.single_linkage_tree_.plot(
        vary_line_width=False, truncate_mode="lastp", p=10, cmap="none", colorbar=False
    )


def test_minimum_spanning_tree_plot(model, plt):
    model.minimum_spanning_tree_.plot(edge_cmap="Reds")


def test_minimum_spanning_tree_plot_projects_high_dimensional_data(plt):
    H, _ = make_blobs(n_samples=50, random_state=0, n_features=10)
    HDBSCAN(min_cluster_size=5).fit(H).minimum_spanning_tree_.plot(
        edge_cmap="Reds", vary_line_width=False, colorbar=False
    )


def test_project_to_2d_prefers_umap_when_installed(monkeypatch):
    class FakeUMAP:
        def fit_transform(self, data):
            return np.full((data.shape[0], 2), 7.0)

    fake_umap = type(sys)("umap")
    fake_umap.UMAP = FakeUMAP
    monkeypatch.setitem(sys.modules, "umap", fake_umap)

    projection = fast_hdbscan.plots._project_to_2d(np.zeros((5, 10)))
    assert np.all(projection == 7.0)


def test_project_to_2d_falls_back_to_tsne_without_umap(monkeypatch):
    monkeypatch.setitem(sys.modules, "umap", None)  # makes `import umap` raise ImportError
    H, _ = make_blobs(n_samples=50, random_state=0, n_features=40)
    assert fast_hdbscan.plots._project_to_2d(H).shape == (50, 2)


def recursive_leaves(cluster_tree, node):
    children = cluster_tree[cluster_tree["parent"] == node]["child"]
    if len(children) == 0:
        return [node]
    return [leaf for child in children for leaf in recursive_leaves(cluster_tree, child)]


def test_get_leaves_matches_recursive_order(model):
    dtype = model._condensed_tree.dtype
    #        10
    #      /    \
    #    11      12
    #   /  \    /  \
    #  13  14  15  16
    #     /  \
    #    17  18
    edges = [(10, 11), (10, 12), (11, 13), (11, 14), (12, 15), (12, 16), (14, 17), (14, 18)]
    tree = np.array([(p, c, 1.0, 5) for p, c in edges], dtype=dtype)
    assert _get_leaves(tree) == [13, 17, 18, 15, 16]

    cluster_tree = model._condensed_tree[model._condensed_tree["child_size"] > 1]
    root = cluster_tree["parent"].min()
    assert _get_leaves(model._condensed_tree) == recursive_leaves(cluster_tree, root)


# --- Parity with hdbscan on the same tree arrays (skipped without hdbscan)


def test_condensed_tree_plot_data_matches_hdbscan(model):
    hdbscan_plots = pytest.importorskip("hdbscan.plots")
    ours = model.condensed_tree_
    theirs = hdbscan_plots.CondensedTree(model._condensed_tree, model.labels_)

    ours_data = ours.get_plot_data()
    theirs_data = theirs.get_plot_data()
    for key in theirs_data:
        if key == "cluster_bounds":
            assert ours_data[key] == theirs_data[key]
        else:
            np.testing.assert_allclose(ours_data[key], theirs_data[key])
    assert ours._select_clusters() == theirs._select_clusters()


@pytest.mark.parametrize("cut", [0.2, 0.5, 1.0])
@pytest.mark.parametrize("min_cluster_size", [2, 5, 20])
def test_get_clusters_matches_hdbscan(model, cut, min_cluster_size):
    hdbscan_plots = pytest.importorskip("hdbscan.plots")
    theirs = hdbscan_plots.SingleLinkageTree(model._single_linkage_tree)
    np.testing.assert_array_equal(
        model.single_linkage_tree_.get_clusters(cut, min_cluster_size),
        theirs.get_clusters(cut, min_cluster_size),
    )
