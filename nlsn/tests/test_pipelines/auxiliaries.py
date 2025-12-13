"""Auxiliaries for testing pipelines."""

import narwhals as nw
from polars.testing import assert_frame_equal

__all__ = [
    "pl_assert_equal",
]


def pl_assert_equal(df_chk, df_exp) -> None:
    if isinstance(df_chk, (nw.DataFrame, nw.LazyFrame)):
        df_chk = nw.to_native(df_chk)
    if isinstance(df_exp, (nw.DataFrame, nw.LazyFrame)):
        df_exp = nw.to_native(df_exp)
    assert_frame_equal(df_chk, df_exp)
