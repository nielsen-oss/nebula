"""Unit-test for 'select_all_columns' function."""

from nlsn.nebula.spark_transformers.filters import select_all_columns


def test_select_all_columns():
    """Test select_all_columns function."""
    assert select_all_columns(None, None, None)
    assert select_all_columns(None, None, "*")
    assert select_all_columns(["test"], "x", "*")
    assert not select_all_columns("test", "x", None)
    assert not select_all_columns(None, "x", None)
