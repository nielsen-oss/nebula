"""Unit-Test for 'Head', 'Tail' and 'PrintSchema' transformers."""

import pandas as pd
import polars as pl
import pytest

from nlsn.nebula.pandas_polars_transformers import Head, PrintSchema, Tail
from nlsn.tests.auxiliaries import assert_pandas_polars_frame_equal


def _get_input_df(df_type: str):
    data = {
        "c1": ["a", "b", "c"],
        "c2": ["aa", "bb", "cc"],
    }
    if df_type == "pandas":
        return pd.DataFrame(data)
    else:
        return pl.DataFrame(data)


def _test_head_tail(df_type: str, df, t, n: int, meth: str):
    df_chk = t.transform(df)
    df_exp = getattr(df, meth)(n) if n >= 0 else df
    assert_pandas_polars_frame_equal(df_type, df_exp, df_chk)


@pytest.mark.parametrize("df_type", ["pandas", "polars"])
@pytest.mark.parametrize("n", [-1, 0, 1])
def test_head(df_type: str, n):
    """Test Head transformer."""
    df_input = _get_input_df(df_type)
    t = Head(n=n)
    _test_head_tail(df_type, df_input, t, n, "head")


@pytest.mark.parametrize("df_type", ["pandas", "polars"])
@pytest.mark.parametrize("columns", ["c1", None])
def test_print_schema(df_type: str, columns):
    """Test PrintSchema transformer."""
    df_input = _get_input_df(df_type)
    t = PrintSchema(columns=columns)
    df_chk = t.transform(df_input)
    assert_pandas_polars_frame_equal(df_type, df_input, df_chk)


@pytest.mark.parametrize("df_type", ["pandas", "polars"])
@pytest.mark.parametrize("n", [-1, 0, 1])
def test_tail(df_type: str, n):
    """Test Tail transformer."""
    df_input = _get_input_df(df_type)
    t = Tail(n=n)
    _test_head_tail(df_type, df_input, t, n, "tail")
