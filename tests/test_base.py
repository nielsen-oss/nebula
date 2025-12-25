"""Unit-tests for base transformer functionalities."""

import pandas as pd
import pytest

from nebula.base import (
    LazyWrapper,
    extract_lazy_params,
    is_ns_lazy_request,
)
from nebula.storage import nebula_storage as ns
from nebula.transformers import AssertNotEmpty, SelectColumns


@pytest.mark.parametrize(
    "o, exp",
    [
        (None, False),
        (True, False),
        ([], False),
        ([1, 2], False),
        ([1, ns], False),
        ([ns, "key"], True),
    ],
)
def test_is_ns_lazy_request(o, exp: bool):
    """Test 'is_ns_lazy_request' function."""
    assert is_ns_lazy_request(o) is exp


def test_extract_lazy_params():
    """Test 'extract_lazy_params' function."""
    ns.clear()
    ns.set("key", 10)

    kws = {"a": (ns, "key"), "b": 100}

    try:
        chk = extract_lazy_params(kws)
        assert chk["a"] == 10
        assert chk["b"] == 100
    finally:
        ns.clear()


class TestLazyWrapper:
    """Test 'LazyWrapper' class."""

    @staticmethod
    @pytest.fixture(scope="class", name="df_input")
    def _get_df_input():
        data = [
            ["a", 0.0],
            ["b", 1.0],
        ]
        return pd.DataFrame(data, columns=["c1", "c2"])

    def test_with_params(self, df_input):
        """Test using a transformer with parameters."""
        t = LazyWrapper(SelectColumns, **{"columns": "c1"})
        df_chk = t.transform(df_input)
        pd.testing.assert_frame_equal(df_chk, df_input[["c1"]])

    def test_with_params_error(self, df_input):
        """Test using a transformer with (wrong) parameter."""
        t = LazyWrapper(SelectColumns, **{"columns": "c1"})
        df_chk = t.transform(df_input)
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(df_chk, df_input[["c2"]])

    def test_without_params(self, df_input):
        """Test using a transformer without parameters."""
        t = LazyWrapper(AssertNotEmpty)
        df_chk = t.transform(df_input)
        pd.testing.assert_frame_equal(df_chk, df_input)
