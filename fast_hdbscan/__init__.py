from .hdbscan import HDBSCAN, fast_hdbscan, PLSCAN
from .branches import BranchDetector, find_branch_sub_clusters

# Force JIT compilation on import
import numpy as np

random_state = np.random.RandomState(42)
random_data = random_state.random(size=(15, 3))
HDBSCAN(allow_single_cluster=True).fit(random_data)
HDBSCAN(cluster_selection_method="leaf").fit(random_data)
PLSCAN().fit(random_data)

__all__ = [
    "HDBSCAN",
    "fast_hdbscan",
    "PLSCAN",
    "BranchDetector",
    "find_branch_sub_clusters",
]
