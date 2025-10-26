"""Unit-test for Substring."""

import pytest
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from nlsn.nebula.spark_transformers import Substring


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    fields = [
        StructField(name="id", dataType=IntegerType()),
        StructField(name="input_col", dataType=StringType()),
    ]

    data = [
        (1, "abcdef"),
        (2, "xyz"),
        (3, "pq"),
        (4, None),
        (5, ""),
        (6, "zxcvbnmlkj"),
    ]
    return spark.createDataFrame(data, schema=StructType(fields)).persist()


def test_substring_transformer_no_length():
    """Test Substring transformer with wrong 'length'."""
    with pytest.raises(ValueError):
        Substring(input_col="x", start=0, length=-1)


@pytest.mark.parametrize("output_col", ["substring", None])
def test_substring_transformer(df_input, output_col):
    """Test Substring transformer."""
    t = Substring(input_col="input_col", start=3, length=2, output_col=output_col)
    df_chk = t.transform(df_input)
    exp_col = output_col if output_col else "input_col"
    df_exp = df_input.withColumn(exp_col, F.substring("input_col", 3, 2))
    assert_df_equality(df_chk, df_exp, ignore_row_order=True)
