import numba
import numpy as np
from collections import namedtuple

from .disjoint_set import ds_rank_create
from .hdbscan import clusters_from_spanning_tree
from .cluster_trees import empty_condensed_tree
from .boruvka import merge_components, update_point_components

CoreGraph = namedtuple("CoreGraph", ["weights", "distances", "indices", "indptr"])


@numba.njit(parallel=True)
def knn_mst_union(neighbors, core_distances, min_spanning_tree, lens_values):
    # List of dictionaries of child: (weight, distance)
    graph = [
        {np.int32(0): (np.float64(0.0), np.float64(0.0)) for _ in range(0)}
        for _ in range(neighbors.shape[0])
    ]

    # Add knn edges
    for point in numba.prange(len(core_distances)):
        children = graph[point]
        parent_lens = lens_values[point]
        parent_dist = core_distances[point]
        for child in neighbors[point]:
            if child < 0:
                continue
            children[child] = (
                max(parent_lens, lens_values[child]),
                max(parent_dist, core_distances[child]),
            )

    # Add non-knn mst edges
    for parent, child, distance in min_spanning_tree:
        parent = np.int32(parent)
        child = np.int32(child)
        children = graph[parent]
        if child in children:
            continue
        children[child] = (max(lens_values[parent], lens_values[child]), distance)

    return graph


@numba.njit(parallel=True)
def flatten_to_csr(graph):
    # Count children to form indptr
    num_points = len(graph)
    indptr = np.empty(num_points + 1, dtype=np.int32)
    indptr[0] = 0
    for i, children in enumerate(graph):
        indptr[i + 1] = indptr[i] + len(children)

    # Flatten children to form indices, weights, and distances
    weights = np.empty(indptr[-1], dtype=np.float32)
    distances = np.empty(indptr[-1], dtype=np.float32)
    indices = np.empty(indptr[-1], dtype=np.int32)
    for point in numba.prange(num_points):
        start = indptr[point]
        children = graph[point]
        for j, (child, (weight, distance)) in enumerate(children.items()):
            weights[start + j] = weight
            distances[start + j] = distance
            indices[start + j] = child

    # Return as named csr tuple
    return CoreGraph(weights, distances, indices, indptr)


@numba.njit(parallel=True)
def sort_by_lens(graph):
    new_weights = np.empty_like(graph.weights)
    new_distances = np.empty_like(graph.distances)
    new_indices = np.empty_like(graph.indices)
    for point in numba.prange(len(graph.indptr) - 1):
        start = graph.indptr[point]
        end = graph.indptr[point + 1]
        
        row_weights = graph.weights[start:end]
        row_distances = graph.distances[start:end]
        row_indices = graph.indices[start:end]

        order = np.argsort(row_weights)
        new_weights[start:end] = row_weights[order]
        new_distances[start:end] = row_distances[order]
        new_indices[start:end] = row_indices[order]
    return CoreGraph(new_weights, new_distances, new_indices, graph.indptr)


@numba.njit(parallel=True)
def apply_lens(core_graph, lens_values):
    # Apply new lens to the graph
    for point in numba.prange(len(lens_values)):
        point_lens = lens_values[point]
        start = core_graph.indptr[point]
        end = core_graph.indptr[point + 1]
        for idx, child in enumerate(core_graph.indices[start:end]):
            core_graph.weights[start + idx] = max(point_lens, lens_values[child])
    return sort_by_lens(core_graph)


@numba.njit(locals={"parent": numba.types.int32})
def select_components(distances, indices, indptr, point_components):
    component_edges = {
        np.int64(0): (np.int32(0), np.int32(1), np.float32(0.0)) for _ in range(0)
    }

    # Find the best edges from each component
    for parent, from_component in enumerate(point_components):
        start = indptr[parent]
        if start == len(indices) or indices[start] == -1:
            continue

        neighbor = indices[start]
        distance = distances[start]
        if from_component in component_edges:
            if distance < component_edges[from_component][2]:
                component_edges[from_component] = (parent, neighbor, distance)
        else:
            component_edges[from_component] = (parent, neighbor, distance)

    return component_edges


@numba.njit(parallel=True)
def update_graph_components(distances, indices, indptr, point_components):
    for point in numba.prange(len(point_components)):
        counter = 0
        start = indptr[point]
        end = indptr[point + 1]
        for idx in range(start, end):
            child = indices[idx]
            if child == -1:
                break
            if point_components[child] != point_components[point]:
                indices[start + counter] = indices[idx]
                distances[start + counter] = distances[idx]
                counter += 1
        indices[start + counter : end] = -1
        distances[start + counter : end] = np.inf


@numba.njit()
def minimum_spanning_tree(graph, overwrite=False):
    """
    Implements Boruvka on lod-style graph with multiple connected components.
    """
    distances = graph.weights
    indices = graph.indices
    indptr = graph.indptr
    if not overwrite:
        indices = indices.copy()
        distances = distances.copy()

    disjoint_set = ds_rank_create(len(indptr) - 1)
    point_components = np.arange(len(indptr) - 1)
    n_components = len(point_components)

    edges_list = [np.empty((0, 3), dtype=np.float64) for _ in range(0)]
    while n_components > 1:
        new_edges = merge_components(
            disjoint_set,
            select_components(distances, indices, indptr, point_components),
        )
        if new_edges.shape[0] == 0:
            break

        edges_list.append(new_edges)
        update_point_components(disjoint_set, point_components)
        update_graph_components(distances, indices, indptr, point_components)
        n_components -= new_edges.shape[0]

    counter = 0
    num_edges = sum([edges.shape[0] for edges in edges_list])
    result = np.empty((num_edges, 3), dtype=np.float64)
    for edges in edges_list:
        result[counter : counter + edges.shape[0]] = edges
        counter += edges.shape[0]
    return n_components, point_components, result


@numba.njit()
def core_graph_spanning_tree(neighbors, core_distances, min_spanning_tree, lens):
    graph = sort_by_lens(
        flatten_to_csr(
            knn_mst_union(neighbors, core_distances, min_spanning_tree, lens)
        )
    )
    return (*minimum_spanning_tree(graph), graph)


def core_graph_clusters(
    lens,
    neighbors,
    core_distances,
    min_spanning_tree,
    **kwargs,
):
    num_components, component_labels, lensed_mst, graph = core_graph_spanning_tree(
        neighbors, core_distances, min_spanning_tree, lens
    )
    if num_components > 1:
        for i, label in enumerate(np.unique(component_labels)):
            component_labels[component_labels == label] = i
        return (
            component_labels,
            np.ones(len(component_labels), dtype=np.float32),
            np.empty((0, 4)),
            empty_condensed_tree(),
            lensed_mst,
            graph,
        )

    return (
        *clusters_from_spanning_tree(lensed_mst, **kwargs),
        graph,
    )


def core_graph_to_rec_array(graph):
    result = np.empty(
        graph.indptr[-1],
        dtype=[
            ("parent", np.int32),
            ("child", np.int32),
            ("weight", np.float32),
            ("distance", np.float32),
        ],
    )
    result["parent"] = np.repeat(
        np.arange(len(graph.indptr) - 1), np.diff(graph.indptr)
    )
    result["child"] = graph.indices
    result["weight"] = graph.weights
    result["distance"] = graph.distances
    return result


def core_graph_to_edge_list(graph):
    result = np.empty((graph.indptr[-1], 4), dtype=np.float64)
    result[:, 0] = np.repeat(np.arange(len(graph.indptr) - 1), np.diff(graph.indptr))
    result[:, 1] = graph.indices
    result[:, 2] = graph.weights
    result[:, 3] = graph.distances
    return result
