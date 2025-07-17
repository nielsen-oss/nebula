"""Unit-test for FloorOrCeil."""

import pytest
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, StructField, StructType

from nlsn.nebula.spark_transformers import FloorOrCeil


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    fields = [
        StructField("col1", FloatType(), True),
        StructField("col2", FloatType(), True),
    ]
    data = [
        [1.1, 3.2],
        [2.3, 5.4],
        [4.5, 0.6],
        [0.7, 4.8],
        [0.9, 0.0],
        [None, float("nan")],
        [float("nan"), None],
        [float("nan"), float("nan")],
        [None, None],
        [None, 0.1],
        [0.2, None],
        [None, 3.6],
        [4.7, None],
    ]
    schema = StructType(fields)
    return spark.createDataFrame(data, schema=schema).persist()


@pytest.mark.parametrize(
    "columns, regex, glob, input_output_columns",
    [
        (None, None, None, None),
        (["a", "b"], None, "a*", None),
        (None, None, "*", {"a": "b"}),
        ("a", None, None, {"a": "b"}),
    ],
)
def test_floor_or_ceil_wrong_columns(columns, regex, glob, input_output_columns):
    """Test FloorOrCeil with wrong columns selection strategy."""
    with pytest.raises(AssertionError):
        FloorOrCeil(
            operation="ceil",
            columns=columns,
            regex=regex,
            glob=glob,
            input_output_columns=input_output_columns,
        )


@pytest.mark.parametrize(
    "input_output_columns, err",
    [
        (["a", "b"], TypeError),
        ({1: "1", "b": "2"}, TypeError),
        ({"a": "1", "b": 2}, TypeError),
        ({"a": "1", "b": "1"}, ValueError),
    ],
)
def test_floor_or_ceil_wrong_input_output_columns(input_output_columns, err):
    """Test FloorOrCeil with wrong 'input_output_columns'."""
    with pytest.raises(err):
        FloorOrCeil(
            operation="ceil",
            input_output_columns=input_output_columns,
        )


def test_floor_or_ceil_column_override(df_input):
    """Test FloorOrCeil overriding 'col2'."""
    t = FloorOrCeil(
        operation="ceil",
        input_output_columns={"col1": "new_col", "col2": "col2"},
    )
    df_chk = t.transform(df_input)
    df_exp = df_input.withColumn("new_col", F.ceil("col1")).withColumn(
        "col2", F.ceil("col2")
    )
    assert_df_equality(
        df_chk,
        df_exp,
        ignore_row_order=True,
        ignore_column_order=True,
        allow_nan_equality=True,
    )


@pytest.mark.parametrize(
    "op_str, columns",
    [
        ("ceil", ["col1"]),
        ("ceil", ["col2"]),
        ("floor", ["col1", "col2"]),
    ],
)
def test_floor_or_ceil(df_input, op_str, columns):
    """Test FloorOrCeil."""
    t = FloorOrCeil(
        operation=op_str,
        columns=columns,
    )
    df_chk = t.transform(df_input)

    op = getattr(F, op_str)
    df_exp = df_input
    for c in columns:
        df_exp = df_exp.withColumn(c, op(c))

    assert_df_equality(
        df_chk,
        df_exp,
        ignore_row_order=True,
        ignore_column_order=True,
        allow_nan_equality=True,
    )
