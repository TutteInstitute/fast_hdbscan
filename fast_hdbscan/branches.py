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
    """Detect branch sub-clusters within clusters using flare detection.

    This function identifies branches (elongated sub-structures) within clusters
    using the flare-sensitive clustering approach from FLASC. It analyzes the
    centrality (distance from cluster center) of edges in the reachability graph
    to detect branch-like structures.

    Parameters
    ----------
    clusterer : HDBSCAN
        A fitted HDBSCAN instance containing the clustering results and
        necessary internal structures (single linkage tree, etc.).

    cluster_labels : ndarray of shape (n_samples,), optional
        Initial cluster labels. If None, uses labels from `clusterer.labels_`.

    cluster_probabilities : ndarray of shape (n_samples,), optional
        Probability values for each point. If None, uses
        `clusterer.probabilities_`.

    label_sides_as_branches : bool, default=False
        If True, treats the two sides of a branch split as separate branches.
        If False, requires at least 2 clear branches to separate them.

    min_cluster_size : int, optional
        Minimum size for sub-clusters. If None, uses clusterer's value.

    max_cluster_size : int, optional
        Maximum size for sub-clusters. If None, uses clusterer's value.

    allow_single_cluster : bool, optional
        Whether to allow a single cluster. If None, uses clusterer's value.

    cluster_selection_method : {'eom', 'leaf'}, optional
        Method for selecting clusters. If None, uses clusterer's method.

    cluster_selection_epsilon : float, default=0.0
        Epsilon for cluster selection.

    cluster_selection_persistence : float, default=0.0
        Persistence threshold for cluster selection.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Final cluster labels with branches identified. Branch-like structures
        within clusters may have different labels than the parent cluster.

    probabilities : ndarray of shape (n_samples,)
        Updated membership strength probabilities.

    sub_cluster_labels : ndarray of shape (n_samples,)
        Labels of identified sub-clusters (branches) within each main cluster.

    cluster_probabilities : ndarray of shape (n_samples,)
        Membership strengths for the original clusters.

    branch_labels : ndarray of shape (n_samples,)
        Labels specifically for branches; equivalent to sub_cluster_labels.

    lens_values : ndarray of shape (n_samples,)
        Centrality values (inverse distance from center) for each point.
        Higher values indicate points closer to the cluster center.

    Notes
    -----
    This function is a convenient wrapper around the :class:`BranchDetector`
    class for functional API usage. For object-oriented usage, use
    :class:`BranchDetector` directly.

    Examples
    --------
    >>> from fast_hdbscan import HDBSCAN, find_branch_sub_clusters
    >>> import numpy as np
    >>> X = np.random.random((200, 2))
    >>> clusterer = HDBSCAN(min_cluster_size=10).fit(X)
    >>> (labels, probs, sub_labels, cluster_probs, branch_labels,
    ...  lens_vals) = find_branch_sub_clusters(clusterer)

    References
    ----------
    .. [1] Bot D.M., Peeters J., Liesenborgs J., Aerts J. 2025. FLASC: a
       flare-sensitive clustering algorithm. PeerJ Computer Science 11:e2792
       https://doi.org/10.7717/peerj-cs.2792.
    """
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
    """Performs flare-detection post-processing to identify branches in clusters.

    This class detects branches (elongated sub-structures or flares) within
    clusters using a graph-based approach. For each cluster, a graph is
    constructed from the k-nearest neighbors based on mutual reachability
    distances. Each edge is assigned a centrality value measuring how far it
    lies from the cluster's center. The edges are then recursively clustered
    using these centrality values, progressively removing the 'center' of the
    cluster to identify branch-like structures.

    Branches are persistent sub-structures within clusters that would not be
    detected by standard HDBSCAN clustering. This is particularly useful for
    identifying elongated or multi-modal structures within single clusters.

    Parameters
    ----------
    min_cluster_size : int, optional
        Minimum size for branch sub-clusters. If None, inherits from the
        parent clusterer.

    max_cluster_size : int, optional
        Maximum size for branch sub-clusters. If None, inherits from the
        parent clusterer.

    allow_single_cluster : bool, optional
        Whether to allow branch detection to return a single cluster.
        If None, inherits from parent clusterer.

    cluster_selection_method : {'eom', 'leaf'}, optional
        Method for selecting branch clusters from the hierarchy.
        If None, inherits from parent clusterer.

    cluster_selection_epsilon : float, default=0.0
        Epsilon for branch cluster selection.

    cluster_selection_persistence : float, default=0.0
        Persistence threshold for branch cluster selection.

    propagate_labels : bool, default=False
        Whether to propagate branch labels to noise points in clusters.

    label_sides_as_branches : bool, default=False
        If True, treats the two sides of a branch split as separate branches.
        If False, requires at least 2 clear branches for separation.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Final cluster labels with branches identified. Points in branches
        may have different labels than the cluster's core.

    probabilities_ : ndarray of shape (n_samples,)
        Membership strength probabilities after branch detection.

    branch_labels_ : ndarray of shape (n_samples,)
        Labels specifically for identified branches within clusters.

    branch_probabilities_ : ndarray of shape (n_samples,)
        Membership strengths for branch assignments.

    centralities_ : ndarray of shape (n_samples,)
        Centrality values for each point in its cluster (inverse distance
        from center). Higher values indicate points closer to the cluster
        center and less likely to be in a branch.

    cluster_probabilities_ : ndarray of shape (n_samples,)
        Original membership strengths from the parent HDBSCAN clustering.

    cluster_points_ : list of ndarray
        Lists of point indices for each cluster from the parent clustering.

    sub_cluster_labels_ : ndarray of shape (n_samples,)
        Labels for identified sub-clusters (branches).

    sub_cluster_probabilities_ : ndarray of shape (n_samples,)
        Membership strengths for sub-cluster assignments.

    Examples
    --------
    >>> from fast_hdbscan import HDBSCAN, BranchDetector
    >>> import numpy as np
    >>> X = np.random.random((300, 2))
    >>> clusterer = HDBSCAN(min_cluster_size=15).fit(X)
    >>> branch_detector = BranchDetector(min_cluster_size=10)
    >>> branch_detector.fit(clusterer)
    >>> print(f"Original clusters: {len(np.unique(clusterer.labels_)) - 1}")
    >>> print(f"Clusters with branches: {len(np.unique(branch_detector.labels_)) - 1}")

    Methods
    -------
    fit(clusterer, labels=None, probabilities=None, sample_weight=None)
        Detect branches in a fitted HDBSCAN clusterer.

    propagated_labels(label_sides_as_branches=None)
        Get labels with noise points assigned to nearest branch.

    approximation_graph_
        Property returning a graph visualization of branch structure.

    References
    ----------
    .. [1] Bot D.M., Peeters J., Liesenborgs J., Aerts J. 2025. FLASC: a
       flare-sensitive clustering algorithm. PeerJ Computer Science 11:e2792
       https://doi.org/10.7717/peerj-cs.2792.
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
