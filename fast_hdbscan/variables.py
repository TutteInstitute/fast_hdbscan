import os


# Controls whether or not numba functions in the library can use cached values
NUMBA_CACHE = os.getenv("FAST_HDBSCAN_NUMBA_CACHE", "true").lower() in {"1", "true"}
