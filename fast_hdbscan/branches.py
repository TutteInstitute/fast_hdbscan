import numpy as np
from .sub_clusters import SubClusterDetector, find_sub_clusters


def compute_centrality(data, probabilities, *args):
    points = args[-1]
    cluster_data = data[points, :]
    centroid = np.average(cluster_data, weights=probabilities[points], axis=0)
    with np.errstate(divide="ignore"):
        return 1 / np.linalg.norm(cluster_data - centroid[None, :], axis=1)


def apply_branch_threshold(
    labels,
    branch_labels,
    probabilities,
    cluster_probabilities,
    cluster_points,
    linkage_trees,
    label_sides_as_branches=False,
):
    running_id = 0
    min_branch_count = 1 if label_sides_as_branches else 2
    for pts, tree in zip(cluster_points, linkage_trees):
        unique_branch_labels = np.unique(branch_labels[pts])
        has_noise = int(unique_branch_labels[0] == -1)
        num_branches = len(unique_branch_labels) - has_noise
        if num_branches <= min_branch_count and tree is not None:
            labels[pts] = running_id
            probabilities[pts] = cluster_probabilities[pts]
            running_id += 1
        else:
            labels[pts] = branch_labels[pts] + has_noise + running_id
            running_id += num_branches + has_noise


def find_branch_sub_clusters(
    clusterer,
    cluster_labels=None,
    cluster_probabilities=None,
    label_sides_as_branches=False,
    min_cluster_size=None,
    max_cluster_size=None,
    allow_single_cluster=None,
    cluster_selection_method=None,
    cluster_selection_epsilon=0.0,
    cluster_selection_persistence=0.0,
):
    result = find_sub_clusters(
        clusterer,
        cluster_labels,
        cluster_probabilities,
        lens_callback=compute_centrality,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        allow_single_cluster=allow_single_cluster,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_persistence=cluster_selection_persistence,
    )
    apply_branch_threshold(
        result[0],
        result[4],
        result[1],
        result[3],
        result[-1],
        label_sides_as_branches=label_sides_as_branches,
    )
    return result


class BranchDetector(SubClusterDetector):
    """
    Performs a flare-detection post-processing step to detect branches within
    clusters [1]_.

    For each cluster, a graph is constructed connecting the data points based on
    their mutual reachability distances. Each edge is given a centrality value
    based on how far it lies from the cluster's center. Then, the edges are
    clustered as if that centrality was a distance, progressively removing the
    'center' of each cluster and seeing how many branches remain.

    References
    ----------
    .. [1] Bot, D. M., Peeters, J., Liesenborgs J., & Aerts, J. (2023, November).
       FLASC: A Flare-Sensitive Clustering Algorithm: Extending HDBSCAN* for
       Detecting Branches in Clusters. arXiv:2311.15887.
    """

    def __init__(
        self,
        min_cluster_size=None,
        max_cluster_size=None,
        allow_single_cluster=None,
        cluster_selection_method=None,
        cluster_selection_epsilon=0.0,
        cluster_selection_persistence=0.0,
        propagate_labels=False,
        label_sides_as_branches=False,
    ):
        super().__init__(
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            allow_single_cluster=allow_single_cluster,
            cluster_selection_method=cluster_selection_method,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_persistence=cluster_selection_persistence,
            propagate_labels=propagate_labels,
        )
        self.label_sides_as_branches = label_sides_as_branches

    def fit(self, clusterer, labels=None, probabilities=None, sample_weight=None):
        super().fit(clusterer, labels, probabilities, sample_weight, compute_centrality)
        apply_branch_threshold(
            self.labels_,
            self.sub_cluster_labels_,
            self.probabilities_,
            self.cluster_probabilities_,
            self.cluster_points_,
            self._linkage_trees,
            label_sides_as_branches=self.label_sides_as_branches,
        )
        self.branch_labels_ = self.sub_cluster_labels_
        self.branch_probabilities_ = self.sub_cluster_probabilities_
        self.centralities_ = self.lens_values_
        return self

    def propagated_labels(self, label_sides_as_branches=None):
        if label_sides_as_branches is None:
            label_sides_as_branches = self.label_sides_as_branches

        labels, branch_labels = super().propagated_labels()
        apply_branch_threshold(
            labels,
            branch_labels,
            np.zeros_like(self.probabilities_),
            np.zeros_like(self.probabilities_),
            self.cluster_points_,
            self._linkage_trees,
            label_sides_as_branches=label_sides_as_branches,
        )
        return labels, branch_labels

    @property
    def approximation_graph_(self):
        """See :class:`~hdbscan.plots.ApproximationGraph` for documentation."""
        return super()._make_approximation_graph(
            lens_name="centrality", sub_cluster_name="branch"
        )
