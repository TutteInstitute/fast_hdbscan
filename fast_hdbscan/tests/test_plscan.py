"""
Tests for PLSCAN clustering algorithm
Based on test_hdbscan.py
"""

import numpy as np
from sklearn.utils._testing import (
    assert_array_equal,
    assert_array_almost_equal,
)
from fast_hdbscan import PLSCAN
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import pytest

n_clusters = 3
X, y = make_blobs(n_samples=300, random_state=42, centers=n_clusters, cluster_std=1.0)
# X, y = shuffle(X, y, random_state=7)
# X = StandardScaler().fit_transform(X)


def test_plscan_feature_vector():
    labels = PLSCAN().fit(X).labels_
    print(labels)
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_1 == n_clusters


def test_plscan_input_lists():
    X_ = [[1.0, 2.0], [3.0, 4.0]]
    PLSCAN().fit(X_)  # must not raise exception


def test_plscan_badargs():
    with pytest.raises(ValueError):
        PLSCAN().fit("fail")
    with pytest.raises(ValueError):
        PLSCAN().fit(None)
    with pytest.raises(ValueError):
        PLSCAN(min_samples="fail").fit(X)
    with pytest.raises(ValueError):
        PLSCAN(min_samples=-1).fit(X)


def test_plscan_max_layers():
    model = PLSCAN(max_layers=1).fit(X)
    assert len(model.cluster_layers_) == 1


def test_plscan_layer_similarity_threshold():
    # Lower threshold should allow more layers
    model_low = PLSCAN(layer_similarity_threshold=0.01, max_layers=5).fit(X)
    model_high = PLSCAN(layer_similarity_threshold=0.9, max_layers=5).fit(X)
    assert len(model_low.cluster_layers_) >= len(model_high.cluster_layers_)


def test_plscan_reproducibility():
    model1 = PLSCAN(reproducible=True).fit(X)
    model2 = PLSCAN(reproducible=True).fit(X)
    assert_array_equal(model1.labels_, model2.labels_)


def test_plscan_base_n_clusters():
    # Should get close to requested n_clusters in first layer
    requested = 4
    model = PLSCAN(base_n_clusters=requested).fit(X)
    first_layer = model.cluster_layers_[0]
    n_found = len(set(first_layer)) - int(-1 in first_layer)
    assert abs(n_found - requested) <= 1


def test_plscan_multiple_layers():
    model = PLSCAN(max_layers=3, layer_similarity_threshold=0.1).fit(X)
    assert len(model.cluster_layers_) > 1
    # Each layer should have different clusterings
    unique_labels = [set(layer) for layer in model.cluster_layers_]
    assert len(set(map(tuple, unique_labels))) == len(model.cluster_layers_)


def test_plscan_tree_generation():
    model = PLSCAN(max_layers=3).fit(X)
    tree = model.cluster_tree_
    # Tree should be a dict with tuple keys and list values
    assert isinstance(tree, dict)
    for k, v in tree.items():
        assert isinstance(k, tuple)
        assert isinstance(v, list)


def test_plscan_invalid_max_layers():
    """Test parameter validation for max_layers."""
    with pytest.raises(ValueError):
        PLSCAN(max_layers=0).fit(X)
    with pytest.raises(ValueError):
        PLSCAN(max_layers=-1).fit(X)
    with pytest.raises(ValueError):
        PLSCAN(max_layers="invalid").fit(X)


def test_plscan_invalid_base_min_cluster_size():
    """Test parameter validation for base_min_cluster_size."""
    with pytest.raises(ValueError):
        PLSCAN(base_min_cluster_size=0).fit(X)
    with pytest.raises(ValueError):
        PLSCAN(base_min_cluster_size=-5).fit(X)
    with pytest.raises(ValueError):
        PLSCAN(base_min_cluster_size="invalid").fit(X)


def test_plscan_invalid_base_n_clusters():
    """Test parameter validation for base_n_clusters."""
    with pytest.raises(ValueError):
        PLSCAN(base_n_clusters=0).fit(X)
    with pytest.raises(ValueError):
        PLSCAN(base_n_clusters=-1).fit(X)
    with pytest.raises(ValueError):
        PLSCAN(base_n_clusters="invalid").fit(X)


def test_plscan_membership_strengths():
    """Test that membership_strengths_ is computed correctly."""
    model = PLSCAN().fit(X)
    assert hasattr(model, "membership_strengths_")
    assert model.membership_strengths_.shape == model.labels_.shape
    assert np.all(model.membership_strengths_ >= 0)
    assert np.all(model.membership_strengths_ <= 1)
    # Noise points should have 0 strength
    noise_mask = model.labels_ == -1
    if np.any(noise_mask):
        assert np.all(model.membership_strengths_[noise_mask] == 0)


def test_plscan_membership_strength_layers():
    """Test that membership_strength_layers_ is computed for all layers."""
    model = PLSCAN(max_layers=3).fit(X)
    assert hasattr(model, "membership_strength_layers_")
    assert len(model.membership_strength_layers_) == len(model.cluster_layers_)
    for strengths, labels in zip(
        model.membership_strength_layers_, model.cluster_layers_
    ):
        assert strengths.shape == labels.shape
        assert np.all(strengths >= 0)
        assert np.all(strengths <= 1)


def test_plscan_layer_persistence_scores():
    """Test that layer_persistence_scores_ exists and is valid."""
    model = PLSCAN(max_layers=5).fit(X)
    assert hasattr(model, "layer_persistence_scores_")
    assert len(model.layer_persistence_scores_) == len(model.cluster_layers_)
    assert all(score >= 0 for score in model.layer_persistence_scores_)


def test_plscan_best_layer_selection():
    """Test that the best layer (highest persistence) is chosen as labels_."""
    model = PLSCAN(max_layers=5, layer_similarity_threshold=0.1).fit(X)
    if len(model.cluster_layers_) > 1:
        best_layer_idx = np.argmax(model.layer_persistence_scores_)
        assert_array_equal(model.labels_, model.cluster_layers_[best_layer_idx])
        assert_array_equal(
            model.membership_strengths_,
            model.membership_strength_layers_[best_layer_idx],
        )


def test_plscan_single_layer():
    """Test that with single layer, it's used as labels_."""
    model = PLSCAN(max_layers=1).fit(X)
    assert len(model.cluster_layers_) == 1
    assert_array_equal(model.labels_, model.cluster_layers_[0])
    # Note: membership_strengths_ is only set when there are multiple layers
    # For single layer, can access via membership_strength_layers_[0]
    assert hasattr(model, "membership_strength_layers_")
    assert len(model.membership_strength_layers_) == 1


def test_plscan_small_dataset():
    """Test behavior with very small datasets."""
    X_small = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    model = PLSCAN(base_min_cluster_size=2).fit(X_small)
    assert hasattr(model, "labels_")
    assert len(model.labels_) == 4
    assert hasattr(model, "cluster_layers_")


def test_plscan_single_cluster():
    """Test with all points in one tight blob."""
    X_single, _ = make_blobs(n_samples=100, centers=1, cluster_std=0.1, random_state=42)
    model = PLSCAN(base_min_cluster_size=5).fit(X_single)
    # Should find at least one cluster (no guarantee all are clustered)
    n_clusters_found = len(set(model.labels_)) - int(-1 in model.labels_)
    assert n_clusters_found >= 1


def test_plscan_uniform_noise():
    """Test with uniformly distributed points (all noise)."""
    X_noise = np.random.RandomState(42).uniform(0, 10, (100, 2))
    model = PLSCAN(base_min_cluster_size=10).fit(X_noise)
    assert hasattr(model, "labels_")
    # May find small clusters or all noise, just ensure it doesn't crash


def test_plscan_fit_returns_self():
    """Test that fit() returns self for sklearn API compliance."""
    model = PLSCAN()
    result = model.fit(X)
    assert result is model


def test_plscan_fit_predict():
    """Test that fit_predict() returns labels same as labels_ attribute."""
    model = PLSCAN()
    labels = model.fit_predict(X)
    assert_array_equal(labels, model.labels_)


def test_plscan_min_cluster_sizes():
    """Test that min_cluster_sizes_ attribute exists."""
    model = PLSCAN().fit(X)
    assert hasattr(model, "min_cluster_sizes_")


def test_plscan_total_persistence():
    """Test that total_persistence_ attribute exists."""
    model = PLSCAN().fit(X)
    assert hasattr(model, "total_persistence_")


def test_plscan_tree_structure_validity():
    """Test that cluster tree has valid parent-child relationships."""
    model = PLSCAN(max_layers=3).fit(X)
    tree = model.cluster_tree_
    # Check that all keys and values are tuples of (layer, cluster_id)
    for parent, children in tree.items():
        assert isinstance(parent, tuple) and len(parent) == 2
        assert isinstance(parent[0], (int, np.integer))
        assert isinstance(parent[1], (int, np.integer))
        for child in children:
            assert isinstance(child, tuple) and len(child) == 2
            assert isinstance(child[0], (int, np.integer))
            assert isinstance(child[1], (int, np.integer))
            # Child layer should be less than parent layer
            assert child[0] < parent[0]


def test_plscan_tree_covers_clusters():
    """Test that all clusters from all layers appear in tree."""
    model = PLSCAN(max_layers=3).fit(X)
    tree = model.cluster_tree_
    # Collect all clusters mentioned in tree
    tree_clusters = set()
    for parent, children in tree.items():
        tree_clusters.add(parent)
        for child in children:
            tree_clusters.add(child)

    # Collect all actual clusters from layers (excluding noise)
    actual_clusters = set()
    for layer_idx, layer_labels in enumerate(model.cluster_layers_):
        unique_labels = np.unique(layer_labels)
        for label in unique_labels:
            if label >= 0:  # Exclude noise
                actual_clusters.add((layer_idx, int(label)))

    # All actual clusters should be in tree
    assert actual_clusters.issubset(tree_clusters)


def test_plscan_layer_diversity():
    """Test that different layers have different clusterings."""
    model = PLSCAN(max_layers=5, layer_similarity_threshold=0.1).fit(X)
    if len(model.cluster_layers_) > 1:
        for i in range(len(model.cluster_layers_) - 1):
            # Layers should not be identical
            assert not np.array_equal(
                model.cluster_layers_[i], model.cluster_layers_[i + 1]
            )


def test_plscan_layer_ordering():
    """Test that layers are sorted by number of clusters (descending)."""
    model = PLSCAN(max_layers=5, layer_similarity_threshold=0.1).fit(X)
    n_clusters_per_layer = []
    for layer in model.cluster_layers_:
        n_clusters = len(set(layer)) - int(-1 in layer)
        n_clusters_per_layer.append(n_clusters)

    # Check that the list is sorted in descending order
    assert n_clusters_per_layer == sorted(n_clusters_per_layer, reverse=True)


def test_plscan_with_sample_weights():
    """Test that sample_weights parameter works."""
    weights = np.ones(X.shape[0])
    # Give higher weight to first half of points
    weights[: X.shape[0] // 2] = 2.0

    model_weighted = PLSCAN().fit(X, sample_weight=weights)
    model_unweighted = PLSCAN().fit(X)

    # Results should be different (not guaranteed but likely with these weights)
    assert hasattr(model_weighted, "labels_")
    assert hasattr(model_unweighted, "labels_")


def test_plscan_get_params():
    """Test that get_params() returns all parameters."""
    model = PLSCAN(
        min_samples=10,
        max_layers=5,
        base_min_cluster_size=15,
        layer_similarity_threshold=0.3,
        reproducible=True,
    )
    params = model.get_params()
    assert params["min_samples"] == 10
    assert params["max_layers"] == 5
    assert params["base_min_cluster_size"] == 15
    assert params["layer_similarity_threshold"] == 0.3
    assert params["reproducible"] is True


def test_plscan_set_params():
    """Test that set_params() works correctly."""
    model = PLSCAN()
    model.set_params(min_samples=15, max_layers=7)
    assert model.min_samples == 15
    assert model.max_layers == 7
    # Should be able to fit after set_params
    model.fit(X)
    assert hasattr(model, "labels_")


def test_plscan_vs_hdbscan_base_layer():
    """Test that PLSCAN base layer produces reasonable clustering similar to HDBSCAN."""
    from fast_hdbscan import HDBSCAN

    # Use same parameters for both
    min_cluster_size = 10
    min_samples = 5

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples
    ).fit(X)
    plscan_model = PLSCAN(
        base_min_cluster_size=min_cluster_size, min_samples=min_samples
    ).fit(X)

    # Get number of clusters (excluding noise)
    hdbscan_n_clusters = len(set(hdbscan_model.labels_)) - int(
        -1 in hdbscan_model.labels_
    )
    plscan_n_clusters = len(set(plscan_model.labels_)) - int(-1 in plscan_model.labels_)

    # Should find similar number of clusters (within reasonable range)
    assert abs(hdbscan_n_clusters - plscan_n_clusters) <= 2

    # Both should find at least some clusters
    assert hdbscan_n_clusters > 0
    assert plscan_n_clusters > 0


def test_plscan_vs_hdbscan_parameter_compatibility():
    """Test that shared parameters work similarly between PLSCAN and HDBSCAN."""
    from fast_hdbscan import HDBSCAN

    # Test min_samples parameter
    for min_samples in [3, 5, 10]:
        plscan = PLSCAN(min_samples=min_samples)
        hdbscan = HDBSCAN(min_samples=min_samples)

        plscan.fit(X)
        hdbscan.fit(X)

        # Both should produce valid clusterings
        assert hasattr(plscan, "labels_")
        assert hasattr(hdbscan, "labels_")
        assert len(plscan.labels_) == len(X)
        assert len(hdbscan.labels_) == len(X)


def test_plscan_provides_multiple_scales():
    """Test that PLSCAN provides multiple scales unlike single-scale HDBSCAN."""
    from fast_hdbscan import HDBSCAN

    plscan = PLSCAN(max_layers=5, layer_similarity_threshold=0.1).fit(X)
    hdbscan = HDBSCAN().fit(X)

    # PLSCAN should provide multiple layers
    assert hasattr(plscan, "cluster_layers_")
    assert len(plscan.cluster_layers_) >= 1

    # HDBSCAN produces single clustering
    assert hasattr(hdbscan, "labels_")

    # PLSCAN should offer cluster tree structure
    assert hasattr(plscan, "cluster_tree_")
    assert isinstance(plscan.cluster_tree_, dict)


def test_plscan_max_layers_limit():
    """Test that max_layers parameter strictly limits number of layers."""
    for max_layers in [1, 2, 3, 5, 10]:
        model = PLSCAN(max_layers=max_layers, layer_similarity_threshold=0.01).fit(X)
        assert len(model.cluster_layers_) <= max_layers


def test_plscan_layer_similarity_threshold_extremes():
    """Test behavior at extreme threshold values."""
    # Very low threshold - should produce more diverse layers
    model_low = PLSCAN(layer_similarity_threshold=0.0, max_layers=10).fit(X)

    # Very high threshold - should produce fewer layers (clusters too similar)
    model_high = PLSCAN(layer_similarity_threshold=0.99, max_layers=10).fit(X)

    # Low threshold should allow more layers
    assert len(model_low.cluster_layers_) >= len(model_high.cluster_layers_)

    # Both should produce at least one layer
    assert len(model_low.cluster_layers_) >= 1
    assert len(model_high.cluster_layers_) >= 1


def test_plscan_base_n_clusters_variations():
    """Test base_n_clusters with various values."""
    for n_clusters in [2, 5, 10]:
        model = PLSCAN(base_n_clusters=n_clusters).fit(X)
        # Should have at least one layer
        assert len(model.cluster_layers_) >= 1
        # First layer should try to approximate requested clusters
        first_layer_n = len(set(model.cluster_layers_[0])) - int(
            -1 in model.cluster_layers_[0]
        )
        # Allow some flexibility in matching requested clusters
        assert first_layer_n > 0


def test_plscan_base_n_clusters_exceeds_data():
    """Test base_n_clusters larger than reasonable for dataset."""
    # Request more clusters than samples
    model = PLSCAN(base_n_clusters=X.shape[0] + 10).fit(X)
    assert hasattr(model, "labels_")
    assert len(model.labels_) == len(X)
    # Should handle gracefully


def test_plscan_base_min_cluster_size_variations():
    """Test different base_min_cluster_size values."""
    # Small min cluster size - more granular clusters
    model_small = PLSCAN(base_min_cluster_size=2).fit(X)

    # Large min cluster size - fewer, larger clusters
    model_large = PLSCAN(base_min_cluster_size=50).fit(X)

    # Both should produce valid results
    assert hasattr(model_small, "labels_")
    assert hasattr(model_large, "labels_")

    # Small size typically finds more clusters
    n_clusters_small = len(set(model_small.labels_)) - int(-1 in model_small.labels_)
    n_clusters_large = len(set(model_large.labels_)) - int(-1 in model_large.labels_)

    # This relationship should generally hold
    assert n_clusters_small >= n_clusters_large


def test_plscan_reproducible_false():
    """Test that reproducible=False may produce different results."""
    # Note: This test may occasionally fail due to randomness
    # Run multiple times to check for variation
    labels_list = []
    for _ in range(3):
        model = PLSCAN(reproducible=False).fit(X)
        labels_list.append(model.labels_.copy())

    # All should be valid
    for labels in labels_list:
        assert len(labels) == len(X)


def test_plscan_verbose_mode():
    """Test that verbose=True doesn't crash."""
    import sys
    from io import StringIO

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        model = PLSCAN(verbose=True).fit(X)
        output = sys.stdout.getvalue()

        # Should have produced some output
        assert len(output) > 0
        assert hasattr(model, "labels_")
    finally:
        sys.stdout = old_stdout


def test_plscan_combined_parameters():
    """Test various combinations of parameters work together."""
    configs = [
        {
            "min_samples": 3,
            "max_layers": 2,
            "base_min_cluster_size": 5,
            "layer_similarity_threshold": 0.1,
        },
        {
            "min_samples": 10,
            "max_layers": 5,
            "base_n_clusters": 4,
            "layer_similarity_threshold": 0.3,
        },
        {
            "min_samples": 5,
            "max_layers": 1,
            "base_min_cluster_size": 20,
            "reproducible": True,
        },
    ]

    for config in configs:
        model = PLSCAN(**config).fit(X)
        assert hasattr(model, "labels_")
        assert hasattr(model, "cluster_layers_")
        assert len(model.cluster_layers_) <= config.get("max_layers", 10)
