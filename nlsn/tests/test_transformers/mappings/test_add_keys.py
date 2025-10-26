"""Unit-test for AddKeys."""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    MapType,
    StringType,
    StructField,
    StructType,
)

from nlsn.nebula.auxiliaries import ensure_flat_list
from nlsn.nebula.spark_transformers import AddKeys


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    data = [
        ({}, {}, {}),
        ({"a": 1}, {"a": 2}, {"a": 3.0}),
        ({"a": 1, "b": 1}, {"a": 2, "b": 2}, {"a": 3.0, "b": 3.0}),
        ({}, {}, {"a": 3.0}),
        ({"a": 1}, {"a": 2}, {}),
        ({"a": 1, "b": 1}, {"a": 2, "b": 2}, {"a": 3.0, "b": 3.0, "c": 3.0}),
        ({"a": 1, "b": 1}, {"a": 2, "b": 2}, {"a": 3.0, "c": 3.0}),
        ({}, {"b": 2}, {"a": 3.0}),
        ({"a": 1}, {"b": 2}, {}),
        ({"b": 1}, {"a": 2, "b": 2}, {"a": 3.0, "b": 3.0, "c": 3.0}),
        ({"a": 1}, {"b": 2}, {"c": 3.0}),
        ({}, {}, None),
        ({"a": 1}, {"a": 2}, None),
        ({"a": 1, "b": 1}, {"a": 2, "b": 2}, None),
        ({}, None, None),
        (None, None, None),
    ]

    fields = [
        StructField("c1", MapType(StringType(), IntegerType()), True),
        StructField("c2", MapType(StringType(), LongType()), True),
        StructField("c3", MapType(StringType(), DoubleType()), True),
    ]
    schema = StructType(fields)
    ret = spark.createDataFrame(data, schema)
    ret = ret.select("*", *[F.col(i).alias(f"copy_{i}") for i in ret.columns])
    return ret.persist()


def _get_expected(input_dict: dict, to_add: dict):
    if not input_dict:
        return to_add
    return {**to_add, **input_dict}


def _assert_results(rows: list, col_input: str, col_output: str, to_add: dict):
    for row in rows:
        chk: dict = getattr(row, col_input)
        input_dict: dict = getattr(row, col_output)
        if to_add:
            exp: dict = _get_expected(input_dict, to_add)
            assert chk == exp
        else:  # skip if 'to_add' is empty
            assert chk == input_dict


def test_add_keys(df_input):
    """Test AddKeys."""
    input_columns = ["c1", "c2"]
    to_add = {"a": 10, "e": 10}
    t = AddKeys(to_add=to_add, input_columns=input_columns)

    df_out = t.transform(df_input)
    rows = df_out.collect()

    cols_to_check = ensure_flat_list(input_columns)
    for c_base in cols_to_check:
        _assert_results(rows, c_base, f"copy_{c_base}", to_add)


def test_add_keys_empty(df_input):
    """Test AddKeys with an empty dictionary to add."""
    t = AddKeys(to_add={}, input_columns="c1")
    df_out = t.transform(df_input)
    rows = df_out.collect()
    _assert_results(rows, "c1", "copy_c1", {})


def test_add_keys_wrong_type():
    """Test AddKeys with a wrong type to add."""
    with pytest.raises(AssertionError):
        _ = AddKeys(to_add=[], input_columns="c1")


@pytest.mark.parametrize("input_columns", ["NO", ["c1", "NO"]])
def test_add_keys_wrong(df_input, input_columns):
    """Test AddKeys passing wrong parameters."""
    with pytest.raises(AssertionError):
        t = AddKeys(
            to_add={},
            input_columns=input_columns,
        )
        _ = t.transform(df_input)
