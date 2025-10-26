"""Unit-tests for 'cache-to-nebula-storage' util."""

import pytest

from nlsn.nebula import nebula_storage as ns
from nlsn.nebula.pipelines.pipelines import (
    _FAIL_CACHE,
    PipelineError,
    _cache_to_nebula_storage,
    _update_fail_name,
)


@pytest.mark.parametrize("n", [None, 0, 1, 2, 3, 5, 1000])
def test_update_fail_name(n):
    """Unit-test for '_update_fail_name' function."""

    def _make_name(_s: str, _i: int):
        return f"{_s}_{_i}"

    ns.clear()
    name = "test"

    if n is None:
        chk = _update_fail_name(name)
        assert chk == name
        return

    ns.set(name, True)
    # n is integer from now on
    for i in range(n):
        ns.set(_make_name(name, i), True)

    if n > 100:
        with pytest.raises(PipelineError):
            _update_fail_name(name)
    else:
        chk = _update_fail_name(name)
        exp = _make_name(name, n)
        assert chk == exp

    ns.clear()


@pytest.mark.parametrize(
    "list_k1, list_k2, err",
    [
        (["split", "split"], ["a", "b"], False),
        (["transformer"], ["a"], False),
        (["split", "wrong"], ["a", "b"], True),
        (["split", "transformer"], ["a", "b"], True),
        (["transformer", "transformer"], ["a", "b"], True),
    ],
)
def test_cache_to_nebula_storage(list_k1, list_k2, err):
    """Unit-test for '_cache_to_nebula_storage' function."""
    ns.clear()
    _FAIL_CACHE.clear()

    for k1, k2 in zip(list_k1, list_k2):
        _FAIL_CACHE[(k1, k2)] = "x"

    if err:
        with pytest.raises(AssertionError):
            _cache_to_nebula_storage()
    else:
        t_out, _msg, li_keys = _cache_to_nebula_storage()
        if "transformer" in list_k1:
            assert t_out == "transformer"
        else:
            assert t_out == "split"

        for key in li_keys:
            assert ns.get(key) == "x"

    ns.clear()
    _FAIL_CACHE.clear()
