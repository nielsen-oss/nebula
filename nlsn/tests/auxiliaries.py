"""Auxiliary module for unit-testing."""

import re
from fnmatch import fnmatch

import narwhals as nw
import polars as pl
from pandas.testing import assert_frame_equal as pd_assert_frame_equal
from polars.testing import assert_frame_equal as pl_assert_frame_equal

from nlsn.nebula.auxiliaries import ensure_list

__all__ = [
    "assert_pandas_polars_frame_equal",
    "from_pandas",
    "get_expected_columns",
]


def assert_pandas_polars_frame_equal(
        df_type: str,
        df_exp,
        df_chk,
        *,
        check_row_order=False,
        check_dtype: bool = True,
) -> None:
    """Assert that 2 Pandas / Polars dataframes are equal.

    Args:
        df_type (str):
            Type of dataframe.
        df_exp (pd.DataFrame | pl.DataFrame):
            Expected dataframe.
        df_chk (pd.DataFrame | pl.DataFrame):
            DataFrame to check for equality.
        check_row_order (bool):
            Whether to check row order. Used only for Polars.
            Defaults to False.
        check_dtype (bool):
            Whether to check row order. Used only for Polars.
            Defaults TO True
    """
    if df_type == "pandas":
        pd_assert_frame_equal(df_exp, df_chk)
    elif df_type == "polars":
        pl_assert_frame_equal(
            df_exp, df_chk, check_row_order=check_row_order, check_dtypes=check_dtype
        )
    else:
        raise ValueError("df_type must be either 'pandas' or 'polars'")


def from_pandas(df_input, backend: str, to_nw: bool, spark=None):
    if backend == "polars":
        df = pl.from_pandas(df_input)
    elif backend == "spark":
        df = spark.createDataFrame(df_input)
    else:
        df = df_input

    if to_nw:
        df = nw.from_native(df)

    return df


def get_expected_columns(
        input_columns: list[str],
        *,
        columns: list | None = None,
        regex: str | None = None,
        glob: str | None = None
) -> list[str]:
    """Get the expected columns by using the nlsn.nebula.select_columns logic."""
    columns = ensure_list(columns)

    if not (regex or glob):
        return columns

    columns_seen: set[str] = set()

    ret: list[str] = columns[:]
    columns_seen.update(columns)

    if regex:
        pattern = re.compile(regex)
        for col in input_columns:
            if bool(pattern.search(col)) and col not in columns_seen:
                ret.append(col)
                columns_seen.add(col)

    if glob:
        for col in input_columns:
            if fnmatch(col, glob) and col not in columns_seen:
                ret.append(col)
                columns_seen.add(col)

    return ret

# def assert_frame_equal(
#         df_chk_input,
#         df_exp_input,
#         ignore_order: bool = False,
# ):
#     df_chk: pd.DataFrame = _to_pd = (df_chk_input)
#     df_exp: pd.DataFrame = _to_pd(df_exp_input)
