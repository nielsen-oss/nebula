"""Unit-test for Aggregate."""

import pytest
from chispa import assert_df_equality
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from nlsn.nebula.spark_transformers import Aggregate


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark: SparkSession):
    data = [
        [1, "11"],
        [2, "22"],
        [3, "33"],
        [4, "44"],
        [5, "55"],
    ]
    df = spark.createDataFrame(data, schema="c1: int, c2: string")
    return df.cache()


def test_aggregate_single_op(df_input):
    """Test Aggregate transformer with a single aggregation."""
    aggregations = {"agg": "sum", "col": "c1", "alias": "c1_sum"}
    df_chk = Aggregate(aggregations=aggregations).transform(df_input)
    df_exp = df_input.agg(F.sum("c1").alias("c1_sum"))
    assert_df_equality(df_chk, df_exp, ignore_row_order=True)


def test_aggregate_double_ops(df_input):
    """Test Aggregate transformer with a double aggregation."""
    aggregations = [
        {"agg": "sum", "col": "c1", "alias": "c1_sum"},
        {"agg": "count", "col": "c2"},
    ]
    df_chk = Aggregate(aggregations=aggregations).transform(df_input)
    df_exp = df_input.agg(F.sum("c1").alias("c1_sum"), F.count("c2"))
    assert_df_equality(df_chk, df_exp, ignore_row_order=True)
