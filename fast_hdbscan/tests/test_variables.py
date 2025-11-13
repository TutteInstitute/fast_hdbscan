def test_variable_default():
    from fast_hdbscan.variables import NUMBA_CACHE

    assert NUMBA_CACHE is True
