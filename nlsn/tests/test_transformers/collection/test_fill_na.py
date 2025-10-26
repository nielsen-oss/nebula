"""Unit-test for FillNa."""

import pytest
from chispa import assert_df_equality
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    FloatType,
    IntegerType,
    MapType,
    StringType,
    StructField,
    StructType,
)

from nlsn.nebula.spark_transformers import FillNa


@pytest.fixture(scope="module", name="df_input")
def _get_input_data(spark):
    fields = [
        StructField("c_float", FloatType(), True),
        StructField("c_int", IntegerType(), True),
        StructField("c_str", StringType(), True),
        StructField("c_bool", BooleanType(), True),
        StructField("c_dict", MapType(StringType(), IntegerType(), True), True),
        StructField("c_array", ArrayType(IntegerType(), True), True),
    ]
    schema = StructType(fields)

    data = [
        [1.0, 1, "a", True, {"a": 1}, [1, 2]],
        [None, 1, "a", True, {"a": 1}, [1, 2]],
        [float("nan"), 1, "a", True, {"a": 1}, [1, 2]],
        [1.0, None, "a", True, {"a": 1}, [1, 2]],
        [1.0, 1, None, True, {"a": 1}, [1, 2]],
        [1.0, 1, "a", None, {"a": 1}, [1, 2]],
        [1.0, 1, "a", True, None, [1, 2]],
        [1.0, 1, "a", True, {"a": 1}, None],
    ]
    return spark.createDataFrame(data, schema).persist()


def _assert_equality(df_chk, df_exp):
    c2d = ["c_dict", "c_array"]
    assert_df_equality(
        df_chk.drop(*c2d),
        df_exp.drop(*c2d),
        ignore_row_order=True,
        allow_nan_equality=True,
    )


def test_fill_na_error():
    """Test FillNa with wrong parameters."""
    with pytest.raises(AssertionError):
        FillNa(value={"a": 1}, columns=["x"])


@pytest.mark.parametrize(
    "value",
    [
        {"c_float": 1.1, "c_int": 3},
        {"c_bool": False, "c_str": "X"},
    ],
)
def test_fill_na_dict_value(df_input, value: dict):
    """Test FillNa using 'value' as <dict>."""
    t = FillNa(value=value)
    df_chk = t.transform(df_input)
    df_exp = df_input.fillna(value=value)
    _assert_equality(df_chk, df_exp)


def test_fill_na_subset(df_input):
    """Test FillNa using a subset."""
    t = FillNa(value=1, columns=["c_int"])
    df_chk = t.transform(df_input)
    df_exp = df_input.fillna(value=1, subset=["c_int"])
    _assert_equality(df_chk, df_exp)
