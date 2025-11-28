"""Unit-tests for base transformer functionalities."""

import pandas as pd
import pytest

from nlsn.nebula.base import (
    LazyWrapper,
    extract_lazy_params,
    is_lazy_function,
    is_ns_lazy_request,
    nlazy,
)
from nlsn.nebula.transformers import AssertNotEmpty, SelectColumns
from nlsn.nebula.storage import nebula_storage as ns


@nlazy
def _lazy():
    return "ok"


def _not_lazy():
    return "ok"


@pytest.mark.parametrize("func, exp", [(_lazy, True), (_not_lazy, False)])
def test_is_lazy_function(func, exp: bool):
    """Test 'is_lazy_function' function."""
    assert is_lazy_function(func) is exp


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

    kws = {
        "a": _lazy,
        "b": _not_lazy,
        "c": (ns, "key"),
        "d": 100,
    }

    try:
        chk = extract_lazy_params(kws)
        assert chk["a"] == "ok"
        assert chk["b"] is _not_lazy
        assert chk["c"] == 10
        assert chk["d"] == 100
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
