"""Unit-tests for pipeline auxiliaries."""

import pytest

from nebula.pipelines.pipe_aux import parse_storage_request
from nebula.transformers import AssertNotEmpty


class TestStorageRequest:
    """Test 'is_storage_request' function."""

    @staticmethod
    @pytest.mark.parametrize(
        "o",
        [
            AssertNotEmpty(),
            [AssertNotEmpty()],
            {"x": "y"},
            {"store": "a", "store2": "b"},
            "no",
            0,
            1,
        ],
    )
    def test_wrong_request(o):
        """Test 'is_storage_request' function and expect 0."""
        assert parse_storage_request(o).value == 0

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
            parse_storage_request(o)

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
        assert parse_storage_request(key).value > 0
