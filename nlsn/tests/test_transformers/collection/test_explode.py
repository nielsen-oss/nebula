"""Unit-test for Explode."""

from typing import List

import pytest
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    MapType,
    StringType,
    StructField,
    StructType,
)

from nlsn.nebula.spark_transformers import Explode

_DATA = [
    ("row_1", [0, 1, None, 2], {"a": 0, "b": None, "c": 1}),
    ("row_2", [0, None, None], {"a": 0, "b": None}),
    ("row_3", [0], {"c": 3}),
    ("row_4", [None], {"a": 0, "b": None}),
    ("row_5", [], {}),
    ("row_6", None, {"a": 0, "c": 6}),
]

_COL_STRING: str = "row_name"
_COL_ARRAY: str = "arrays"
_COL_MAP: str = "mapping"
_COL_OUTPUT_ARRAY: str = "output"
_COL_OUTPUT_MAP: List[str] = ["key", "value"]


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    """Creates initial DataFrame."""
    fields = [
        StructField(_COL_STRING, StringType()),
        StructField(_COL_ARRAY, ArrayType(IntegerType())),
        StructField(_COL_MAP, MapType(StringType(), IntegerType())),
    ]
    schema: StructType = StructType(fields)
    return spark.createDataFrame(_DATA, schema).cache()


def _check_columns_array_type(cols_chk, kwg):
    """Check output columns."""
    # If "output_col" is not provided, the output is stored in the input
    # column (_COL_ARRAY).
    out_col: str = kwg.get("output_cols", _COL_ARRAY)
    drop_after: bool = kwg["drop_after"]

    # Create the list of expected columns.
    cols_exp = [_COL_STRING, _COL_ARRAY, _COL_MAP]

    # If output column != input column, add it to the expected columns.
    if out_col != _COL_ARRAY:
        cols_exp.append(_COL_OUTPUT_ARRAY)

    # Remove the input column if "drop_after" = True and
    # output column != input column.
    if drop_after and (_COL_ARRAY != out_col):
        cols_exp.remove(_COL_ARRAY)
    assert cols_chk == cols_exp, kwg


@pytest.mark.parametrize("output_cols", [["c1"], ["c1", 1], ["c1", "c2", "c3"]])
def test_explode_wrong_output_columns(output_cols):
    """Test Explode transformer with wrong 'output_cols'."""
    with pytest.raises(AssertionError):
        Explode(input_col=_COL_MAP, output_cols=output_cols)


@pytest.mark.parametrize("output_cols", [None, "wrong"])
def test_explode_map_wrong_output_columns(df_input, output_cols):
    """Test Explode transformer with wrong 'output_cols' for MapType."""
    t = Explode(input_col=_COL_MAP, output_cols=output_cols)
    with pytest.raises(AssertionError):
        t.transform(df_input)


def test_explode_wrong_input(df_input):
    """Test Explode transformer with <StringType> as input column."""
    t = Explode(input_col=_COL_STRING, output_cols="x")
    with pytest.raises(AssertionError):
        t.transform(df_input)


def _get_expected_len(idx_data):
    # List of iterables.
    data_col = [i[idx_data] for i in _DATA]

    # Expected len when "outer" = False, thus null values and empty
    # arrays are discarded.
    base_len = sum(len(i) for i in data_col if i)

    # Number of rows where data is null or the array is empty.
    # This number sum up with base_len when "outer" = True
    null_len = len([True for i in data_col if not i])

    return base_len, null_len


def test_explode_array(df_input):
    """Test Explode transformer with <ArrayType> as input column."""
    inputs = [
        {"output_cols": _COL_OUTPUT_ARRAY, "outer": True, "drop_after": True},
        {"output_cols": _COL_OUTPUT_ARRAY, "outer": True, "drop_after": False},
        {"output_cols": _COL_OUTPUT_ARRAY, "outer": False, "drop_after": True},
        {"output_cols": _COL_OUTPUT_ARRAY, "outer": False, "drop_after": False},
        {"outer": True, "drop_after": True},
        {"outer": True, "drop_after": False},
        {"outer": False, "drop_after": True},
        {"outer": False, "drop_after": False},
    ]
    # 1 -> array, 2 -> map
    base_len, null_len = _get_expected_len(1)

    for kwg in inputs:
        t = Explode(input_col=_COL_ARRAY, **kwg)
        df_out = t.transform(df_input)

        # Check columns.
        _check_columns_array_type(df_out.columns, kwg)

        len_chk = df_out.count()
        len_exp = base_len
        if kwg["outer"]:
            len_exp += null_len
        assert len_chk == len_exp, kwg


def test_explode_mapping(df_input):
    """Test Explode transformer with <MapType> as input column."""
    inputs = [
        {"outer": True, "drop_after": True},
        {"outer": True, "drop_after": False},
        {"outer": False, "drop_after": True},
        {"outer": False, "drop_after": False},
    ]
    # 1 -> array, 2 -> map
    base_len, null_len = _get_expected_len(2)

    cols_exp = set(df_input.columns).union(set(_COL_OUTPUT_MAP))
    cols_exp.copy()

    for kwg in inputs:
        t = Explode(input_col=_COL_MAP, output_cols=_COL_OUTPUT_MAP, **kwg)
        df_out = t.transform(df_input)

        set_cols_chk = set(df_out.columns)
        # Check columns.
        if kwg["drop_after"]:
            assert set_cols_chk == (cols_exp - {_COL_MAP})
        else:
            assert set_cols_chk == cols_exp

        len_chk = df_out.count()
        len_exp = base_len
        if kwg["outer"]:
            len_exp += null_len
        assert len_chk == len_exp, kwg
