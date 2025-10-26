"""Unit-test for ColumnMethod."""

import pytest
from chispa import assert_df_equality
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType
from pyspark.sql.utils import AnalysisException

from nlsn.nebula.spark_transformers import ColumnMethod


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    """Creates initial DataFrame."""
    fields = [StructField("name", StringType(), True)]

    data = [
        ["house"],
        ["cat"],
        ["secondary"],
        [None],
    ]
    return spark.createDataFrame(data, StructType(fields)).persist()


def test_column_method_invalid_method(df_input):
    """Test ColumnMethod with a wrong method name."""
    with pytest.raises(ValueError):
        t = ColumnMethod(input_column="name", method="invalid")
        t.transform(df_input)


def test_column_method_invalid_column(df_input):
    """Test ColumnMethod with a wrong column name."""
    t = ColumnMethod(input_column="invalid", method="isNull")
    with pytest.raises(AnalysisException):
        t.transform(df_input)


def test_column_method(df_input):
    """Test ColumnMethod."""
    t = ColumnMethod(
        input_column="name", method="contains", output_column="result", args=["se"]
    )
    df_chk = t.transform(df_input)

    df_exp = df_input.withColumn("result", F.col("name").contains("se"))
    assert_df_equality(df_chk, df_exp, ignore_row_order=True)


def test_column_method_no_args(df_input):
    """Test ColumnMethod w/o any arguments and overriding the input column."""
    t = ColumnMethod(input_column="name", method="isNull")
    df_chk = t.transform(df_input)

    df_exp = df_input.withColumn("name", F.col("name").isNull())
    assert_df_equality(df_chk, df_exp, ignore_row_order=True)
