"""Unit-test for LogicalOperator."""

from decimal import Decimal

import pytest
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from nlsn.nebula.spark_transformers import RowWiseGreatestOrLeast


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    data = [
        (1, 2, 3, 1, 3),
        (4, 5, None, 4, 5),
        (None, -8, 9, -8, 9),
        (7, None, None, 7, 7),
        (None, None, None, None, None),
    ]
    fields = [
        StructField("col1", IntegerType(), nullable=True),
        StructField("col2", IntegerType(), nullable=True),
        StructField("col3", IntegerType(), nullable=True),
        StructField("least", IntegerType(), nullable=True),
        StructField("greatest", IntegerType(), nullable=True),
    ]
    return spark.createDataFrame(data, schema=StructType(fields)).persist()


@pytest.mark.parametrize("columns", [("c1",), "c1", ("c1", "c1")])
def test_row_wise_greatest_or_least_wrong_columns(columns):
    """Test RowWiseGreatestOrLeast with wrong columns."""
    with pytest.raises(AssertionError):
        RowWiseGreatestOrLeast(
            columns=columns,
            output_col="max_value",
            operation="greatest",
        )


@pytest.mark.parametrize(
    "columns, glob",
    [
        (["c_wrong_1", "c_wrong_2"], None),
        (None, "c_wrong_*"),
    ],
)
def test_row_wise_greatest_or_least_no_matching_fields(df_input, columns, glob):
    """Test RowWiseGreatestOrLeast with no matching columns."""
    t = RowWiseGreatestOrLeast(
        columns=columns,
        glob=glob,
        output_col="max_value",
        operation="greatest",
    )
    with pytest.raises((ValueError, AssertionError)):
        t.transform(df_input)


@pytest.mark.parametrize("operation", ["least", "greatest"])
def test_row_wise_greatest_or_least(df_input, operation: str):
    """Test RowWiseGreatestOrLeast."""
    t = RowWiseGreatestOrLeast(
        glob="col*",
        output_col="result",
        operation=operation,
    )
    df_chk = t.transform(df_input)
    n_missed = df_chk.filter(F.col("result") != F.col(operation)).count()
    assert n_missed == 0


@pytest.fixture(scope="module", name="df_input_non_numerical")
def _get_df_non_numerical(spark):
    data = [
        ("a", "b", "c"),
        ("d", "e", None),
        (None, "f", "g"),
        ("h", None, None),
        (None, None, None),
    ]
    fields = [
        StructField("col1", StringType(), nullable=True),
        StructField("col2", StringType(), nullable=True),
        StructField("col3", StringType(), nullable=True),
    ]
    return spark.createDataFrame(data, StructType(fields)).cache()


@pytest.mark.parametrize(
    "columns,operation,exp_data",
    [
        (
            ["col1", "col2"],
            "greatest",
            [
                ("a", "b", "c", "b"),
                ("d", "e", None, "e"),
                (None, "f", "g", "f"),
                ("h", None, None, "h"),
                (None, None, None, None),
            ],
        ),
        (
            ["col2", "col3"],
            "least",
            [
                ("a", "b", "c", "b"),
                ("d", "e", None, "e"),
                (None, "f", "g", "f"),
                ("h", None, None, None),
                (None, None, None, None),
            ],
        ),
    ],
)
def test_row_wise_greatest_or_least_non_numeric(
    spark, df_input_non_numerical, columns, operation, exp_data
):
    """Test RowWiseGreatestOrLeast with non-numerical data."""
    t = RowWiseGreatestOrLeast(
        columns=columns, output_col="result", operation=operation
    )
    df_chk = t.transform(df_input_non_numerical)

    fields = [
        StructField("col1", StringType(), nullable=True),
        StructField("col2", StringType(), nullable=True),
        StructField("col3", StringType(), nullable=True),
        StructField("result", StringType(), nullable=True),
    ]
    df_exp = spark.createDataFrame(exp_data, schema=StructType(fields))

    assert_df_equality(df_chk, df_exp, ignore_row_order=True)


def test_row_wise_greatest_or_least_mixed_column_types(spark):
    """Test RowWiseGreatestOrLeast with mixed data types."""
    data = [
        (-3.14, -42, Decimal("-123.45"), -2.718, -9223372036854775808, "Hello"),
        (0.0, 5, Decimal("0.00"), 0.0, -9223372036854775808, "World"),
        (3.14, 42, Decimal("123.45"), 2.718, 9223372036854775807, None),
        (None, None, None, None, None, "NullString"),
    ]

    fields = [
        StructField("col_1", FloatType(), nullable=True),
        StructField("col_2", IntegerType(), nullable=True),
        StructField("col_3", DecimalType(8, 2), nullable=True),
        StructField("col_4", DoubleType(), nullable=True),
        StructField("col_5", LongType(), nullable=True),
        StructField("col_6", StringType(), nullable=True),
    ]
    df_input = spark.createDataFrame(data, schema=StructType(fields))

    t = RowWiseGreatestOrLeast(
        glob="*",
        output_col="max_value",
        operation="greatest",
    )
    with pytest.raises(TypeError):
        t.transform(df_input)


@pytest.mark.parametrize("operation", ["least", "greatest"])
def test_row_wise_greatest_or_least_with_nan(spark, operation: str):
    """Test RowWiseGreatestOrLeast with NaN."""
    data = [
        (1.0, float("nan"), 1.0),
        (float("nan"), 0.0, 0.0),
        (float("nan"), float("nan"), None),
    ]
    fields = [
        StructField("col_1", FloatType(), nullable=True),
        StructField("col_2", FloatType(), nullable=True),
        StructField("expected", FloatType(), nullable=True),
    ]
    df = spark.createDataFrame(data, schema=StructType(fields))

    t = RowWiseGreatestOrLeast(
        glob="*",
        output_col="result",
        operation=operation,
    )

    df_chk = t.transform(df)
    n_missed = df_chk.filter(F.col("result") != F.col("expected")).count()
    assert n_missed == 0
