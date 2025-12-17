"""Auxiliary module for unit-testing."""

import re
from fnmatch import fnmatch

import narwhals as nw
import pandas as pd
import polars as pl
import pyspark.sql
from polars.testing import assert_frame_equal

from nebula.auxiliaries import ensure_list

__all__ = [
    "from_pandas",
    "get_expected_columns",
    "pd_sort_assert",
    "pl_assert_equal",
    "to_pandas",
]


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


def pd_sort_assert(
        chk: pd.DataFrame,
        exp: pd.DataFrame,
        na_position: str = 'first',
        **kws
):
    cols = list(chk.columns)
    if list(exp.columns) != cols:
        raise ValueError(f"{exp.columns=} != {chk.columns=}")

    chk = chk.sort_values(cols, na_position=na_position).reset_index(drop=True)
    exp = exp.sort_values(cols, na_position=na_position).reset_index(drop=True)
    pd.testing.assert_frame_equal(chk, exp, **kws)


def pl_assert_equal(df_chk, df_exp, sort: list[str] | None = None) -> None:
    if isinstance(df_chk, (nw.DataFrame, nw.LazyFrame)):
        df_chk = nw.to_native(df_chk)
    if isinstance(df_exp, (nw.DataFrame, nw.LazyFrame)):
        df_exp = nw.to_native(df_exp)
    if sort:
        df_chk = df_chk.sort(sort)
        df_exp = df_exp.sort(sort)
    assert_frame_equal(df_chk, df_exp)


def to_pandas(df_input) -> pd.DataFrame:
    # from narwhals to native ...
    if isinstance(df_input, (nw.DataFrame, nw.LazyFrame)):
        df = nw.to_native(df_input)
    else:
        df = df_input

    # ... to pandas
    if isinstance(df, pl.DataFrame):  # polars
        return df.to_pandas()
    if isinstance(df, pl.LazyFrame):  # polars lazy
        return df.collect().to_pandas()
    if isinstance(df, pyspark.sql.DataFrame):  # spark
        return df.toPandas()
    elif isinstance(df, pd.DataFrame):
        return df

    raise TypeError(f"Unknown df type: {type(df)}")
