import numba
import numpy as np

from collections import namedtuple

from .disjoint_set import ds_rank_create, ds_find, ds_union_by_rank

from numba.typed import Dict, List
from numba.types import int64, ListType

int64_list_type = ListType(int64)

LinkageMergeData = namedtuple("LinkageMergeData", ["parent", "size", "next"])


@numba.njit()
def create_linkage_merge_data(base_size):
    parent = np.full(2 * base_size - 1, -1, dtype=np.intp)
    size = np.concatenate((np.ones(base_size, dtype=np.intp), np.zeros(base_size - 1, dtype=np.intp)))
    next_parent = np.array([base_size], dtype=np.intp)

    return LinkageMergeData(parent, size, next_parent)


@numba.njit()
def create_linkage_merge_data_w_sample_weights(sample_weights):
    base_size = sample_weights.shape[0]
    parent = np.full(2 * base_size - 1, -1, dtype=np.intp)
    size = np.concatenate((sample_weights, np.zeros(base_size - 1, dtype=np.float32)))
    next_parent = np.array([base_size], dtype=np.intp)

    return LinkageMergeData(parent, size, next_parent)


@numba.njit()
def linkage_merge_find(linkage_merge, node):
    relabel = node
    while linkage_merge.parent[node] != -1 and linkage_merge.parent[node] != node:
        node = linkage_merge.parent[node]

    linkage_merge.parent[node] = node

    # label up to the root
    while linkage_merge.parent[relabel] != node:
        next_relabel = linkage_merge.parent[relabel]
        linkage_merge.parent[relabel] = node
        relabel = next_relabel

    return node


@numba.njit()
def linkage_merge_join(linkage_merge, left, right):
    linkage_merge.size[linkage_merge.next[0]] = linkage_merge.size[left] + linkage_merge.size[right]
    linkage_merge.parent[left] = linkage_merge.next[0]
    linkage_merge.parent[right] = linkage_merge.next[0]
    linkage_merge.next[0] += 1


@numba.njit()
def mst_to_linkage_tree(sorted_mst):
    result = np.empty((sorted_mst.shape[0], sorted_mst.shape[1] + 1))

    n_samples = sorted_mst.shape[0] + 1
    linkage_merge = create_linkage_merge_data(n_samples)

    for index in range(sorted_mst.shape[0]):

        left = np.intp(sorted_mst[index, 0])
        right = np.intp(sorted_mst[index, 1])
        delta = sorted_mst[index, 2]

        left_component = linkage_merge_find(linkage_merge, left)
        right_component = linkage_merge_find(linkage_merge, right)

        if left_component > right_component:
            result[index][0] = left_component
            result[index][1] = right_component
        else:
            result[index][1] = left_component
            result[index][0] = right_component

        result[index][2] = delta
        result[index][3] = linkage_merge.size[left_component] + linkage_merge.size[right_component]

        linkage_merge_join(linkage_merge, left_component, right_component)

    return result


@numba.njit()
def mst_to_linkage_tree_w_sample_weights(sorted_mst, sample_weights):
    result = np.empty((sorted_mst.shape[0], sorted_mst.shape[1] + 1))

    linkage_merge = create_linkage_merge_data_w_sample_weights(sample_weights)

    for index in range(sorted_mst.shape[0]):

        left = np.intp(sorted_mst[index, 0])
        right = np.intp(sorted_mst[index, 1])
        delta = sorted_mst[index, 2]

        left_component = linkage_merge_find(linkage_merge, left)
        right_component = linkage_merge_find(linkage_merge, right)

        if left_component > right_component:
            result[index][0] = left_component
            result[index][1] = right_component
        else:
            result[index][1] = left_component
            result[index][0] = right_component

        result[index][2] = delta
        result[index][3] = linkage_merge.size[left_component] + linkage_merge.size[right_component]

        linkage_merge_join(linkage_merge, left_component, right_component)

    return result


@numba.njit()
def bfs_from_hierarchy(hierarchy, bfs_root, num_points):
    to_process = [bfs_root]
    result = []

    while to_process:
        result.extend(to_process)
        next_to_process = []
        for n in to_process:
            if n >= num_points:
                i = n - num_points
                next_to_process.append(int(hierarchy[i, 0]))
                next_to_process.append(int(hierarchy[i, 1]))
        to_process = next_to_process

    return result


@numba.njit()
def eliminate_branch(branch_node, parent_node, lambda_value, parents, children, lambdas, sizes, idx, ignore, hierarchy,
                     num_points):
    if branch_node < num_points:
        parents[idx] = parent_node
        children[idx] = branch_node
        lambdas[idx] = lambda_value
        idx += 1
    else:
        for sub_node in bfs_from_hierarchy(hierarchy, branch_node, num_points):
            if sub_node < num_points:
                children[idx] = sub_node
                parents[idx] = parent_node
                lambdas[idx] = lambda_value
                idx += 1
            else:
                ignore[sub_node] = True

    return idx


CondensedTree = namedtuple('CondensedTree', ['parent', 'child', 'lambda_val', 'child_size'])


@numba.njit()
def empty_condensed_tree():
    parents = np.empty(shape=0, dtype=np.intp)
    others = np.empty(shape=0, dtype=np.float32)
    return CondensedTree(parents, parents, others, others)


@numba.njit(fastmath=True)
def condense_tree(hierarchy, min_cluster_size=10, max_cluster_size=np.inf, sample_weights=None):
    root = 2 * hierarchy.shape[0]
    num_points = hierarchy.shape[0] + 1
    next_label = num_points + 1

    node_list = bfs_from_hierarchy(hierarchy, root, num_points)

    relabel = np.zeros(root + 1, dtype=np.int64)
    relabel[root] = num_points

    parents = np.ones(root, dtype=np.int64)
    children = np.empty(root, dtype=np.int64)
    lambdas = np.empty(root, dtype=np.float32)
    sizes = np.ones(root, dtype=np.float32)

    ignore = np.zeros(root + 1, dtype=np.bool_) # 'bool' is no longer an attribute of 'numpy'

    if sample_weights is None:
        sample_weights = np.ones(num_points, dtype=np.float32)

    idx = 0

    for node in node_list:
        if ignore[node] or node < num_points:
            continue

        parent_node = relabel[node]
        l, r, d, _ = hierarchy[node - num_points]
        left = np.int64(l)
        right = np.int64(r)
        if d > 0.0:
            lambda_value = 1.0 / d
        else:
            lambda_value = np.inf

        left_count = np.float32(hierarchy[left - num_points, 3]) if left >= num_points else sample_weights[left]
        right_count = np.float32(hierarchy[right - num_points, 3]) if right >= num_points else sample_weights[right]

        # The logic here is in a strange order, but it has non-trivial performance gains ...
        # The most common case by far is a singleton on the left; and cluster on the right take care of this separately
        if left < num_points and right_count >= min_cluster_size:
            relabel[right] = parent_node
            parents[idx] = parent_node
            children[idx] = left
            lambdas[idx] = lambda_value
            idx += 1
        # Next most common is a small left cluster and a large right cluster: relabel the right node; eliminate the left branch
        elif left_count < min_cluster_size and right_count >= min_cluster_size:
            relabel[right] = parent_node
            idx = eliminate_branch(left, parent_node, lambda_value, parents, children, lambdas, sizes, idx, ignore,
                                   hierarchy, num_points)
        # Then we have a large left cluster and a small right cluster: relabel the left node; eliminate the right branch
        elif left_count >= min_cluster_size and right_count < min_cluster_size:
            relabel[left] = parent_node
            idx = eliminate_branch(right, parent_node, lambda_value, parents, children, lambdas, sizes, idx, ignore,
                                   hierarchy, num_points)
        # If both clusters are small then eliminate all branches
        elif left_count < min_cluster_size and right_count < min_cluster_size:
            idx = eliminate_branch(left, parent_node, lambda_value, parents, children, lambdas, sizes, idx, ignore,
                                   hierarchy, num_points)
            idx = eliminate_branch(right, parent_node, lambda_value, parents, children, lambdas, sizes, idx, ignore,
                                   hierarchy, num_points)
        # If both clusters are too large then relabel both
        elif left_count > max_cluster_size and right_count > max_cluster_size:
            relabel[left] = parent_node
            relabel[right] = parent_node
        else:
            relabel[left] = next_label

            parents[idx] = parent_node
            children[idx] = next_label
            lambdas[idx] = lambda_value
            sizes[idx] = left_count
            next_label += 1
            idx += 1

            relabel[right] = next_label

            parents[idx] = parent_node
            children[idx] = next_label
            lambdas[idx] = lambda_value
            sizes[idx] = right_count
            next_label += 1
            idx += 1

    return CondensedTree(parents[:idx], children[:idx], lambdas[:idx], sizes[:idx])


@numba.njit()
def extract_leaves(condensed_tree, allow_single_cluster=True):
    n_nodes = condensed_tree.parent.max() + 1
    n_points = condensed_tree.parent.min()
    leaf_indicator = np.ones(n_nodes, dtype=np.bool_)
    leaf_indicator[:n_points] = False

    for parent, child_size in zip(condensed_tree.parent, condensed_tree.child_size):
        if child_size > 1:
            leaf_indicator[parent] = False

    return np.nonzero(leaf_indicator)[0]


@numba.njit()
def cluster_tree_leaves(cluster_tree, n_points):
    n_nodes = cluster_tree.child.max() + 1
    leaf_indicator = np.ones(n_nodes - n_points, dtype=np.bool_)
    leaf_indicator[cluster_tree.parent - n_points] = False
    return np.nonzero(leaf_indicator)[0] + n_points


# The *_bcubed functions below implement the (semi-supervised) HDBSCAN*(BC) algorithm presented
# in Castro Gertrudes, J., Zimek, A., Sander, J. et al. A unified view of density-based methods 
# for semi-supervised clustering and classification. Data Min Knowl Disc 33, 1894–1952 (2019).

@numba.njit()
def cluster_tree_from_condensed_tree_bcubed(condensed_tree, cluster_tree, label_indices):
    # This functions returns a cluster_tree with virtual nodes (if applicable).

    label_indices_list = list(label_indices.keys())
    cluster_tree_parents = list(cluster_tree.parent)

    # A labeled node that has no children and who's parent is not a leaf cluster, then it must be 
    # a noisy node (virtual node). 

    mask1 = condensed_tree.child_size > 1
    mask2 = condensed_tree.child_size == 1
    mask3 = np.array([child in label_indices_list for child in condensed_tree.child])
    mask4 = np.array([parent in cluster_tree_parents for parent in condensed_tree.parent]) # check that it's not a leaf cluster

    mask = (mask1 | (mask2 & mask3 & mask4)) 

    return CondensedTree(condensed_tree.parent[mask], condensed_tree.child[mask], condensed_tree.lambda_val[mask],
                         condensed_tree.child_size[mask])


@numba.njit()
def get_condensed_tree_clusters_bcubed(condensed_tree, label_indices, cluster_tree=None, cluster_tree_bcubed=None, allow_virtual_nodes=False):

    cluster_elements = Dict.empty(
        key_type=int64,
        value_type=int64_list_type,
        )
    
    virtual_nodes = [0 for x in range(0)] 
    labeled_points = set(label_indices.keys())

    parents_set = set(list(condensed_tree.parent))
    for i in range(len(condensed_tree.child) - 1, -1, -1): # Traverse tree bottom up
        parent = condensed_tree.parent[i]
        child = condensed_tree.child[i]
        if child in parents_set:
            if parent in cluster_elements:
                cluster_elements[parent].extend(cluster_elements[child])
            else:
                cluster_labeled_points = list(set(cluster_elements[child]) & labeled_points)
                cluster_elements[parent] = List(cluster_labeled_points)
        elif parent in cluster_elements:
            if child in labeled_points:
                cluster_elements[parent].append(child)
        else:
            cluster_elements[parent] = List.empty_list(int64)
            if child in labeled_points:
                cluster_elements[parent].append(child)

    if allow_virtual_nodes and (cluster_tree is not None) and (cluster_tree_bcubed is not None):
        for node in list(set(cluster_tree_bcubed.child).difference(set(cluster_tree.child))):
            virtual_nodes.append(node)
            cluster_elements[node] = List.empty_list(int64)
            cluster_elements[node].append(node)

    return cluster_elements, np.array(virtual_nodes)


@numba.njit()
def eom_recursion_bcubed(node, cluster_tree, stability_node_scores, bcubed_node_scores, selected_clusters, unselected_nodes):

    current_score_stability_bcubed = np.array([stability_node_scores[node], bcubed_node_scores[node]], dtype=np.float32)

    children = cluster_tree.child[cluster_tree.parent == node]
    child_score_total_stability_bcubed = np.array([0.0, 0.0], dtype=np.float32)

    for child_node in children:
        child_score_total_stability_bcubed += eom_recursion_bcubed(child_node, cluster_tree, stability_node_scores, bcubed_node_scores, selected_clusters, unselected_nodes)

    if child_score_total_stability_bcubed[1] > current_score_stability_bcubed[1]:
        return child_score_total_stability_bcubed

    elif child_score_total_stability_bcubed[1] < current_score_stability_bcubed[1]:
        selected_clusters[node] = True
        unselect_below_node_bcubed(node, cluster_tree, selected_clusters, unselected_nodes)
        return current_score_stability_bcubed   

    # Stability scores used to resolve ties.
    elif child_score_total_stability_bcubed[1] == current_score_stability_bcubed[1]:
        
        if child_score_total_stability_bcubed[0] > current_score_stability_bcubed[0]:
            return child_score_total_stability_bcubed
        else:
            selected_clusters[node] = True
            unselect_below_node_bcubed(node, cluster_tree, selected_clusters, unselected_nodes)
            return current_score_stability_bcubed


@numba.njit()
def score_condensed_tree_nodes_bcubed(cluster_elements, label_indices): 

    label_values = label_indices.values()
    label_counts = {0: 0 for i in range(0)}

    for label in label_values:
        if label in label_counts:
            label_counts[label] +=1
        else:
            label_counts[label] = 1

    label_counts_values = list(label_counts.values())
    total_num_of_labeled_points = sum(label_counts_values)
    bcubed = {0: 0.0 for i in range(0)}

    for cluster, cluster_labeled_points in cluster_elements.items():

        cluster_labeled_points_dict = {0: 0 for i in range(0)}
        bcubed[cluster] = 0.0

        if len(cluster_labeled_points) > 0:
            
            for p in cluster_labeled_points:
                p_label = label_indices[p]
                if p_label in cluster_labeled_points_dict:
                    cluster_labeled_points_dict[p_label] += 1
                else:
                    cluster_labeled_points_dict[p_label] = 1
    
            for label, num_points in cluster_labeled_points_dict.items():

                total_num_of_class_label = label_counts[label]
                num_labeled_in_node = len(cluster_labeled_points)

                precision_point = (num_points/num_labeled_in_node)/total_num_of_labeled_points
                recall_point = (num_points/total_num_of_class_label)/total_num_of_labeled_points

                # Bcubed F-measure 
                bcubed[cluster] += num_points*(2.0/(1.0/precision_point + 1.0/recall_point))
    return bcubed

@numba.njit()
def unselect_below_node_bcubed(node, cluster_tree, selected_clusters, unselected_nodes):
    
    for child in cluster_tree.child[cluster_tree.parent == node]:
        if not unselected_nodes[child]:
            unselect_below_node_bcubed(child, cluster_tree, selected_clusters, unselected_nodes)
            selected_clusters[child] = False
            unselected_nodes[child] = True

@numba.njit()
def extract_clusters_bcubed(condensed_tree, cluster_tree, data_labels, allow_virtual_nodes=False, allow_single_cluster=False):
    label_indices = Dict()
    for index in np.flatnonzero(data_labels > -1):
        label_indices[index] = data_labels[index]

    if allow_virtual_nodes:

        cluster_tree_bcubed = cluster_tree_from_condensed_tree_bcubed(condensed_tree, cluster_tree, label_indices)
        cluster_elements, virtual_nodes = get_condensed_tree_clusters_bcubed(condensed_tree, label_indices, cluster_tree, cluster_tree_bcubed, allow_virtual_nodes)
        stability_node_scores = score_condensed_tree_nodes(condensed_tree)
        for node in virtual_nodes:
            stability_node_scores[node] = np.float32(0.0)
        bcubed_node_scores = score_condensed_tree_nodes_bcubed(cluster_elements, label_indices)
              
    else:

        cluster_tree_bcubed = cluster_tree
        cluster_elements, virtual_nodes = get_condensed_tree_clusters_bcubed(condensed_tree, label_indices)
        stability_node_scores = score_condensed_tree_nodes(condensed_tree) 
        bcubed_node_scores = score_condensed_tree_nodes_bcubed(cluster_elements, label_indices)

    selected_clusters = {node: False for node in bcubed_node_scores}

    if len(cluster_tree_bcubed.parent) == 0:
        return np.zeros(0, dtype=np.int64)

    cluster_tree_root = cluster_tree_bcubed.parent.min()
    unselected_nodes = {node: False for node in bcubed_node_scores}

    if allow_single_cluster:
        eom_recursion_bcubed(cluster_tree_root, cluster_tree_bcubed, stability_node_scores, bcubed_node_scores, selected_clusters, unselected_nodes)
    elif len(bcubed_node_scores) > 1:
        root_children = cluster_tree_bcubed.child[cluster_tree_bcubed.parent == cluster_tree_root]
        for child_node in root_children:
            eom_recursion_bcubed(child_node, cluster_tree_bcubed, stability_node_scores, bcubed_node_scores, selected_clusters, unselected_nodes)

    return np.asarray([node for node, selected in selected_clusters.items() if (selected and (node not in virtual_nodes))])


@numba.njit()
def score_condensed_tree_nodes(condensed_tree):
    root = condensed_tree.parent[0]
    result = {root: np.float32(0.0)}

    for i in range(condensed_tree.parent.shape[0]):
        if condensed_tree.child_size[i] > 1:
            child = condensed_tree.child[i]
            result[child] = -condensed_tree.lambda_val[i] * condensed_tree.child_size[i]

        parent = condensed_tree.parent[i]
        result[parent] += condensed_tree.lambda_val[i] * condensed_tree.child_size[i]

    return result


@numba.njit()
def cluster_tree_from_condensed_tree(condensed_tree):
    return mask_condensed_tree(condensed_tree, condensed_tree.child_size > 1)


@numba.njit()
def mask_condensed_tree(condensed_tree, mask):
    return CondensedTree(
        condensed_tree.parent[mask], 
        condensed_tree.child[mask], 
        condensed_tree.lambda_val[mask],
        condensed_tree.child_size[mask]
    )


@numba.njit()
def unselect_below_node(node, cluster_tree, selected_clusters):
    for child in cluster_tree.child[cluster_tree.parent == node]:
        unselect_below_node(child, cluster_tree, selected_clusters)
        selected_clusters[child] = False


@numba.njit(fastmath=True)
def eom_recursion(node, cluster_tree, node_scores, node_sizes, selected_clusters, max_cluster_size):
    current_score = max(node_scores[node], 0.0) # floating point errors can make score negative!
    current_size = node_sizes[node]

    children = cluster_tree.child[cluster_tree.parent == node]
    child_score_total = 0.0

    for child_node in children:
        child_score_total += eom_recursion(child_node, cluster_tree, node_scores, node_sizes, selected_clusters, max_cluster_size)

    if child_score_total > current_score or current_size > max_cluster_size:
        return child_score_total
    else:
        selected_clusters[node] = True
        unselect_below_node(node, cluster_tree, selected_clusters)
        return current_score


@numba.njit()
def extract_eom_clusters(condensed_tree, cluster_tree, max_cluster_size=np.inf, allow_single_cluster=False):
    node_scores = score_condensed_tree_nodes(condensed_tree)
    if len(cluster_tree.parent) > 0:
        node_sizes = {node: size for node, size in zip(cluster_tree.child, cluster_tree.child_size.astype(np.float32))}
        node_sizes[cluster_tree.parent.min()] = np.float32(cluster_tree.parent.min() - 1)
    else:
        node_sizes = {-1: np.float32(0.0)}
    selected_clusters = {node: False for node in node_scores}

    if len(cluster_tree.parent) == 0:
        return np.zeros(0, dtype=np.int64)

    cluster_tree_root = cluster_tree.parent.min()

    if allow_single_cluster:
        eom_recursion(cluster_tree_root, cluster_tree, node_scores, node_sizes, selected_clusters, max_cluster_size)
    elif len(node_scores) > 1:
        root_children = cluster_tree.child[cluster_tree.parent == cluster_tree_root]
        for child_node in root_children:
            eom_recursion(child_node, cluster_tree, node_scores, node_sizes, selected_clusters, max_cluster_size)

    return np.asarray([node for node, selected in selected_clusters.items() if selected])


@numba.njit()
def simplify_hierarchy(condensed_tree, n_points, persistence_threshold):
    keep_mask = np.ones(condensed_tree.parent.shape[0], dtype=np.bool_)
    cluster_tree = cluster_tree_from_condensed_tree(condensed_tree)

    processed = {np.int64(0)}
    processed.clear()
    while cluster_tree.parent.shape[0] > 0:
        leaves = set(cluster_tree_leaves(cluster_tree, n_points))
        births = max_lambdas(condensed_tree, leaves)
        deaths = min_lambdas(cluster_tree, leaves)

        cluster_mask = np.ones(cluster_tree.parent.shape[0], dtype=np.bool_)
        for leaf in sorted(leaves, reverse=True):
            if leaf in processed or (births[leaf] - deaths[leaf]) >= persistence_threshold:
                continue
            
            # Find rows for leaf and sibling
            leaf_idx = np.searchsorted(cluster_tree.child, leaf)
            parent = cluster_tree.parent[leaf_idx]
            if leaf_idx > 0 and cluster_tree.parent[leaf_idx - 1] == parent:
                sibling_idx = leaf_idx - 1 
            else:
                sibling_idx = leaf_idx + 1
            sibling = cluster_tree.child[sibling_idx]
                        
            # Update parent values to the new parent
            for idx, row in enumerate(cluster_tree.parent):
                if row in [leaf, sibling]:
                    cluster_tree.parent[idx] = parent
            for idx, row in enumerate(condensed_tree.parent):
                if row in [leaf, sibling]:
                    condensed_tree.parent[idx] = parent
                    condensed_tree.lambda_val[idx] = deaths[leaf]
            
            # Mark visited rows
            processed.add(leaf)
            processed.add(sibling)
            cluster_mask[leaf_idx] = False
            cluster_mask[sibling_idx] = False
            for idx, child in enumerate(condensed_tree.child):
                if child in [leaf, sibling]:
                    keep_mask[idx] = False

        if np.all(cluster_mask):
            break
        cluster_tree = mask_condensed_tree(cluster_tree, cluster_mask)

    condensed_tree = mask_condensed_tree(condensed_tree, keep_mask)
    return remap_cluster_ids(condensed_tree, n_points)


@numba.njit()
def remap_cluster_ids(condensed_tree, n_points):
    n_nodes = condensed_tree.parent.max() + 1
    remaining_parents = np.unique(condensed_tree.parent)
    id_map = np.empty(n_nodes - n_points, dtype=np.int64)
    id_map[remaining_parents - n_points] = np.arange(
        n_points, n_points + remaining_parents.shape[0]
    )
    for column in [condensed_tree.parent, condensed_tree.child]:
        for idx, node in enumerate(column):
            if node >= n_points:
                column[idx] = id_map[node - n_points]
    return condensed_tree


@numba.njit()
def cluster_epsilon_search(clusters, cluster_tree, min_epsilon=0.0):
    selected = list()
    # only way to create a typed empty set
    processed = {np.int64(0)}
    processed.clear()

    # cluster_tree is sorted with increasing children
    # prepare to use binary search on parent in segment_in_branches
    parent_order = np.argsort(cluster_tree.parent)
    parents = cluster_tree.parent[parent_order]
    children = cluster_tree.child[parent_order]

    root = cluster_tree.parent.min()
    for cluster in clusters:
        idx = np.searchsorted(cluster_tree.child, cluster)
        death_eps = 1 / cluster_tree.lambda_val[idx]
        if death_eps < min_epsilon:
            if cluster not in processed:
                parent = traverse_upwards(cluster_tree, min_epsilon, root, cluster)
                selected.append(parent)
                processed |= segments_in_branch(parents, children, parent)
        else:
            selected.append(cluster)
    return np.asarray(selected)


@numba.njit()
def traverse_upwards(cluster_tree, min_epsilon, root, segment):
    parent = cluster_tree.parent[cluster_tree.child == segment][0]
    if parent == root:
        return root
    parent_death_eps = 1 / cluster_tree.lambda_val[cluster_tree.child == parent][0]
    if parent_death_eps >= min_epsilon:
        return parent
    else:
        return traverse_upwards(cluster_tree, min_epsilon, root, parent)


@numba.njit()
def segments_in_branch(parents, children, segment):
    # only way to create a typed empty set
    child_set = {np.int64(0)}
    result = {np.int64(0)}
    result.clear()
    to_process = {segment}

    while len(to_process) > 0:
        result |= to_process
   
        child_set.clear()
        for segment in to_process:
            idx = np.searchsorted(parents, segment)
            if idx >= len(parents):
                continue
            child_set.add(children[idx])
            child_set.add(children[idx + 1])
        
        to_process.clear()
        to_process |= child_set

    return result


@numba.njit(parallel=True)
def get_cluster_labelling_at_cut(linkage_tree, cut, min_cluster_size):

    root = 2 * linkage_tree.shape[0]
    num_points = linkage_tree.shape[0] + 1
    result = np.empty(num_points, dtype=np.intp)
    disjoint_set = ds_rank_create(root + 1)

    cluster = num_points
    for i in range(linkage_tree.shape[0]):
        if linkage_tree[i, 2] < cut:
            ds_union_by_rank(disjoint_set, np.intp(linkage_tree[i, 0]), cluster)
            ds_union_by_rank(disjoint_set, np.intp(linkage_tree[i, 1]), cluster)
        cluster += 1

    cluster_size = np.zeros(cluster, dtype=np.intp)
    for n in range(num_points):
        cluster = ds_find(disjoint_set, n)
        cluster_size[cluster] += 1
        result[n] = cluster

    cluster_label_map = {-1: -1}
    cluster_label = 0
    unique_labels = np.unique(result)

    for cluster in unique_labels:
        if cluster_size[cluster] < min_cluster_size:
            cluster_label_map[cluster] = -1
        else:
            cluster_label_map[cluster] = cluster_label
            cluster_label += 1

    for n in numba.prange(num_points):
        result[n] = cluster_label_map[result[n]]

    return result

@numba.njit()
def get_cluster_label_vector(
        tree,
        clusters,
        cluster_selection_epsilon,
        n_samples,
):
    if len(tree.parent) == 0 or len(clusters) == 0:
        return np.full(n_samples, -1, dtype=np.intp)
    root_cluster = tree.parent.min()
    result = np.full(n_samples, -1, dtype=np.intp)
    cluster_label_map = {c: n for n, c in enumerate(np.sort(clusters))}

    disjoint_set = ds_rank_create(max(tree.parent.max() + 1, tree.child.max() + 1))
    clusters = set(clusters)

    for n in range(tree.parent.shape[0]):
        child = tree.child[n]
        parent = tree.parent[n]
        if child not in clusters:
            ds_union_by_rank(disjoint_set, parent, child)

    for n in range(root_cluster):
        cluster = ds_find(disjoint_set, n)
        if cluster < root_cluster:
            result[n] = -1
        elif cluster == root_cluster:
            if len(clusters) == 1:
                max_lambda = tree.lambda_val[tree.parent == cluster].max()
                cur_lambda = tree.lambda_val[tree.child == n]
                if cluster_selection_epsilon > 0.0:
                    if cur_lambda >= 1 / cluster_selection_epsilon:
                        result[n] = cluster_label_map[cluster]
                    else:
                        result[n] = -1
                elif cur_lambda >= max_lambda:
                    result[n] = cluster_label_map[cluster]
                else:
                    result[n] = -1
            else:
                result[n] = -1
        else:
            result[n] = cluster_label_map[cluster]

    return result


@numba.njit()
def max_lambdas(tree, clusters):
    result = {c: 0.0 for c in clusters}

    for n in range(tree.parent.shape[0]):
        cluster = tree.parent[n]
        if cluster in clusters and tree.child_size[n] == 1:
            result[cluster] = max(result[cluster], tree.lambda_val[n])

    return result


@numba.njit()
def min_lambdas(cluster_tree, clusters):
    return {
        c: cluster_tree.lambda_val[np.searchsorted(cluster_tree.child, c)] 
        for c in clusters
    }


@numba.njit()
def get_point_membership_strength_vector(tree, clusters, labels):
    result = np.zeros(labels.shape[0], dtype=np.float32)
    deaths = max_lambdas(tree, set(clusters))
    root_cluster = tree.parent.min()
    cluster_index_map = {n: c for n, c in enumerate(np.sort(clusters))}

    for n in range(tree.child.shape[0]):
        point = tree.child[n]
        if point >= root_cluster or labels[point] < 0:
            continue

        cluster = cluster_index_map[labels[point]]
        max_lambda = deaths[cluster]
        if max_lambda == 0.0 or not np.isfinite(tree.lambda_val[n]):
            result[point] = 1.0
        else:
            lambda_val = min(tree.lambda_val[n], max_lambda)
            result[point] = lambda_val / max_lambda

    return result
