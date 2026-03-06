"""Unit-tests for nebula storage."""

import pytest

from nebula.storage import assert_is_hashable
from nebula.storage import nebula_storage as ns


@pytest.fixture(autouse=True)
def _reset_storage():
    ns.clear()
    ns.allow_overwriting()
    ns.allow_debug(False)
    yield
    ns.clear()
    ns.allow_overwriting()
    ns.allow_debug(False)


def test_hashable_object():
    """Test the 'assert_is_hashable' function with hashable objects."""
    assert_is_hashable(0)
    assert_is_hashable(1)
    assert_is_hashable((1, 2, 3))
    assert_is_hashable("x")


def test_non_hashable_object():
    """Test the 'assert_is_hashable' function with non-hashable objects."""
    with pytest.raises(AssertionError):
        assert_is_hashable([1, 2, 3])


def test_overwriting_status():
    """Test 'overwriting' status."""
    assert ns.is_overwriting_allowed is True
    ns.disallow_overwriting()
    assert ns.is_overwriting_allowed is False


def test_set_debug():
    """Test 'debug' status."""
    ns.allow_debug(True)
    assert ns.is_debug_mode

    with pytest.raises(TypeError):
        ns.allow_debug("wrong")


def test_set_get_isin():
    """Test 'get' and 'isin' functionalities."""
    ns.set("key1", "value1")
    assert ns.isin("key1")
    assert ns.get("key1") == "value1"


def test_clear():
    """Test 'clear' functionality."""
    assert ns.count_objects() == 0

    ns.set("key1", "value1")
    ns.set("key2", "value2")
    assert ns.count_objects() == 2

    ns.clear()
    assert ns.count_objects() == 0


def test_clear_specific_keys():
    """Test 'clear' functionality specifying a single key."""
    ns.set("key1", "value1")
    ns.set("key2", "value2")
    assert ns.count_objects() == 2

    ns.clear("key1")
    assert ns.count_objects() == 1

    assert ns.isin("key1") is False
    assert ns.isin("key2") is True


def test_clear_user_defined_keys():
    """Test 'clear' functionality specifying a list of keys."""
    ns.set("key1", "value1")
    ns.set("key2", "value2")
    ns.set("key3", "value3")
    assert ns.count_objects() == 3

    ns.clear(["key1", "key3"])
    assert ns.count_objects() == 1

    assert ns.isin("key1") is False
    assert ns.isin("key2") is True
    assert ns.isin("key3") is False


def test_list_keys():
    """Test 'list_keys' functionality specifying a list of unsorted keys."""
    ns.set("b", "value1")
    ns.set("c", "value2")
    ns.set("a", "value3")
    chk = ns.list_keys()
    assert chk == ["a", "b", "c"]


def test_set_no_overwriting():
    """Test 'set' method twice with the same key and disallowed overwriting."""
    ns.set("key1", "value1")
    ns.disallow_overwriting()

    with pytest.raises(KeyError):
        ns.set("key1", "value2")


def test_set_different_debug_modes():
    """Test 'set' method with different storage debug mode."""
    ns.allow_debug(False)
    ns.set("key1", "value1", debug=True)
    assert ns.isin("key1") is False

    ns.allow_debug(True)
    ns.set("key1", "value1", debug=True)
    assert ns.get("key1") == "value1"
