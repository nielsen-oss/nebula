"""Unit-test for ColumnBucketizer."""

import chispa
import numpy as np
import pytest
from py4j.protocol import Py4JJavaError
from pyspark.ml.feature import Bucketizer
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType, StructField, StructType

from nlsn.nebula.spark_transformers import ColumnBucketizer


def test_column_bucketizer_wrong_buckets():
    """Test ColumnBucketizer with empty 'buckets' parameter."""
    with pytest.raises(ValueError):
        _ = ColumnBucketizer(
            input_col="a",
            output_col="b",
            buckets=[],
            handle_invalid="keep",
            add_infinity_buckets=False,
        )


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    fields = [
        StructField("idx", IntegerType(), True),
        StructField("values", FloatType(), True),
    ]

    # fmt: off
    data = [
        [0, 3.0],
        [1, 4.5],
        [2, 5.0],
        [3, 5.1],
        [4, 6.0],
        [5, 6.5],
        [6, 7.1],
        [7, 10.0],
        [8, None],
        [9, np.nan],
        [10, -np.inf],
        [11, np.inf],
    ]
    # fmt: on

    return spark.createDataFrame(data, schema=StructType(fields)).persist()


@pytest.mark.parametrize("handle_invalid", ["keep", "error", None])
def test_column_bucketizer_with_infinity(df_input, handle_invalid):
    """Test ColumnBucketizer with 'add_infinity_buckets' set to True."""
    buckets = list(range(2, 11, 2))  # include 10

    kwargs = {
        "input_col": "values",
        "output_col": "output",
        "buckets": buckets,
        "add_infinity_buckets": True,
    }

    if handle_invalid != "keep":
        t = ColumnBucketizer(**kwargs)
        with pytest.raises(Py4JJavaError):
            # .count() doesn't trigger anything.
            t.transform(df_input).collect()
        return

    t = ColumnBucketizer(**kwargs, handle_invalid=handle_invalid)
    df_chk = t.transform(df_input).fillna(-1.0)

    # Extend buckets with +/- inf.
    extended_buckets = [float("-inf")] + buckets[:] + [float("inf")]
    df_exp = (
        Bucketizer(
            splits=extended_buckets,
            inputCol="values",
            outputCol="output",
            handleInvalid="keep",
        )
        .transform(df_input)
        .fillna(-1.0)
    )

    chispa.assert_df_equality(df_chk, df_exp, ignore_row_order=True)


def test_column_bucketizer_without_infinity(df_input):
    """Test ColumnBucketizer with 'add_infinity_buckets' set to False."""
    buckets = list(range(2, 11, 2))  # include 10
    df_input_rid = df_input.filter((F.col("values") >= 2) & (F.col("values") <= 10))

    kwargs = {
        "input_col": "values",
        "output_col": "output",
        "buckets": buckets,
        "add_infinity_buckets": False,
        "handle_invalid": "error",
    }

    t = ColumnBucketizer(**kwargs)
    df_chk = t.transform(df_input_rid)

    df_exp = Bucketizer(
        splits=buckets, inputCol="values", outputCol="output", handleInvalid="error"
    ).transform(df_input_rid)

    chispa.assert_df_equality(df_chk, df_exp, ignore_row_order=True)


def test_column_bucketizer_no_output_col(df_input):
    """Test ColumnBucketizer without setting the output column."""
    buckets = list(range(2, 11, 2))  # include 10
    df_input_rid = df_input.filter((F.col("values") >= 2) & (F.col("values") <= 10))

    kwargs = {
        "input_col": "values",
        "buckets": buckets,
        "add_infinity_buckets": False,
        "handle_invalid": "error",
    }

    t = ColumnBucketizer(**kwargs)
    df_chk = t.transform(df_input_rid)

    df_exp = (
        Bucketizer(
            splits=buckets,
            inputCol="values",
            outputCol="_output_bucket_col_",
            handleInvalid="error",
        )
        .transform(df_input_rid)
        .withColumnRenamed("_output_bucket_col_", "values")
    )

    chispa.assert_df_equality(df_chk, df_exp, ignore_row_order=True)
