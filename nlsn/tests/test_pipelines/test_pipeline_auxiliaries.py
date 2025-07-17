"""Unit-tests for pipeline auxiliaries."""

import pytest

from nlsn.nebula.pipelines.pipelines import (
    _remove_last_transformers,
    is_storage_request,
)
from nlsn.nebula.shared_transformers import Count


class TestStorageRequest:
    """Test 'is_storage_request' function."""

    @staticmethod
    @pytest.mark.parametrize(
        "o",
        [
            Count(),
            [Count()],
            {"x": "y"},
            {"store": "a", "store2": "b"},
            "no",
            0,
            1,
        ],
    )
    def test_wrong_request(o):
        """Test 'is_storage_request' function and expect 0."""
        assert is_storage_request(o).value == 0

    @staticmethod
    @pytest.mark.parametrize(
        "o",
        [
            {"store": 1},
            {"store_debug": 1},
            {"storage_debug_mode": "a"},
            {"replace_with_stored_df": 1},
        ],
    )
    def test_wrong_value_type(o):
        """Wrong value type."""
        with pytest.raises(TypeError):
            is_storage_request(o)

    @staticmethod
    @pytest.mark.parametrize(
        "key",
        [
            {"store": "a"},
            {"store_debug": "a"},
            {"storage_debug_mode": True},
            {"storage_debug_mode": False},
            {"replace_with_stored_df": "another_df"},
        ],
    )
    def test_valid_key(key):
        """Valid requests."""
        assert is_storage_request(key).value > 0


@pytest.mark.parametrize(
    "li, n",
    [
        ([1, 2, 3], 3),
        ([1, 2, 3], 1),
        ([Count(), Count(), 1], 1),
        ([Count(), {"store": "a"}, 1], 1),
    ],
)
def test__remove_last_transformers_error(li, n):
    """Test '_remove_last_transformers' function with wrong arguments."""
    with pytest.raises(AssertionError):
        _remove_last_transformers(li, n)


# Do not duplicate references in the next test
c1 = Count()
c2 = Count()
store_1 = {"store": "a"}
store_2 = {"store": "b"}


@pytest.mark.parametrize(
    "li_input, n, exp",
    [
        ([c1, c2], 1, [c1]),
        ([c1, c2, {"store": "a"}, c1], 2, [c1, {"store": "a"}]),
        ([c1, c2, store_1, store_2, c1], 2, [c1, store_1, store_2]),
        ([], 1, []),  # empty lists are allowed
    ],
)
def test__remove_last_transformers(li_input, n, exp):
    """Test '_remove_last_transformers' function."""
    _remove_last_transformers(li_input, n)
    assert li_input == exp
