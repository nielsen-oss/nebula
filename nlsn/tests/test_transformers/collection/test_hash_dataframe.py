"""Unit-test for HashDataFrame."""

import pytest
from chispa import assert_df_equality
from pyspark.sql import functions as F

from nlsn.nebula.spark_transformers import HashDataFrame


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    data = [
        ["a", 1.0],
        ["b", 4.0],
        ["d", None],
        [None, float("nan")],
        ["e", float("nan")],
    ]

    schema = "c1: string, c2: float"
    return spark.createDataFrame(data, schema=schema).persist()


def test_hash_dataframe_transformer(df_input):
    """Test HashDataFrame transformer."""
    t1 = HashDataFrame(output_col="hashed")
    t2 = HashDataFrame(columns=["c2", "c1"], output_col="hashed")
    df_chk_1 = t1.transform(df_input)
    df_chk_2 = t2.transform(df_input)

    input_cols = F.concat_ws("-", "c1", "c2")
    df_exp = df_input.withColumn("hashed", F.md5(input_cols))

    assert_df_equality(df_chk_1, df_exp, ignore_row_order=True, allow_nan_equality=True)
    assert_df_equality(df_chk_2, df_exp, ignore_row_order=True, allow_nan_equality=True)
