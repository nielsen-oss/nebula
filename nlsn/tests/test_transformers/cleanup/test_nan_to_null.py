"""Unit-test for NanToNull."""

import functools
import operator

import pyspark.sql.functions as F
import pytest
from pyspark.sql.types import (
    DoubleType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from nlsn.nebula.spark_transformers import NanToNull

_nan = float("nan")


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    fields = [
        StructField("a", FloatType(), True),
        StructField("b", DoubleType(), True),
        StructField("c", StringType(), True),
    ]

    data = [
        [0.0, 0.0, "s1"],
        [1.0, 1.0, "s2"],
        [None, _nan, None],
        [_nan, None, None],
    ]

    df = spark.createDataFrame(data, schema=StructType(fields))
    # Add a new column "e", that is derived from a column containing a NaN.
    # Then cast it to integer, so the NaN should be automatically converted to null
    return df.withColumn("d", F.col("b").cast(IntegerType())).persist()


@pytest.mark.parametrize(
    "columns, glob, exp",
    [
        (None, "*", 0),  # Convert all NaN
        ("a", None, 1),  # remain 1 nan in column b, third row
    ],
)
def test_nan_to_null_all_columns(df_input, columns, glob, exp: int):
    """Test NaNToNull transformer applied to all columns."""
    t = NanToNull(columns=columns, glob=glob)
    df_check = t.transform(df_input)

    # Test all columns and the subset "a" + "b", the result should not change
    for nan_columns in [["a", "b"], df_input.columns]:
        f_nan = [F.isnan(c) for c in nan_columns]
        cond = functools.reduce(operator.or_, f_nan)

        n_nan_found = df_check.filter(cond).count()
        assert n_nan_found == exp, "Unexpected rows with NaNs"
