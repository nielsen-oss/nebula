"""Auxiliary module for unit-testing."""

import re
from fnmatch import fnmatch
from typing import List, Optional, Set, Union

import pandas as pd
import polars as pl
from pandas.testing import assert_frame_equal as pd_assert_frame_equal
from polars.testing import assert_frame_equal as pl_assert_frame_equal

from nlsn.nebula.auxiliaries import ensure_list


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
            df_exp, df_chk, check_row_order=check_row_order, check_dtype=check_dtype
        )
    else:
        raise ValueError("df_type must be either 'pandas' or 'polars'")


def get_expected_columns(
    df, columns: Optional[list], regex: Optional[str], glob: Optional[str]
) -> List[str]:
    """Get the expected columns by using the nlsn.nebula.select_columns logic."""
    columns = ensure_list(columns)

    if not (regex or glob):
        return columns

    columns_seen: Set[str] = set()

    ret: List[str] = columns[:]
    columns_seen.update(columns)

    input_columns: List[str] = df.columns

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


def pandas_to_polars(backend: str, df) -> Union[pd.DataFrame, pl.DataFrame]:
    """Convert pandas dataframe to polars dataframe if requested."""
    if backend == "polars":
        return pl.from_pandas(df)
    return df
