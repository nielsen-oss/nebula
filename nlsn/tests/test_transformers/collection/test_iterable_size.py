"""Unit-test for IterableSize."""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    MapType,
    StringType,
    StructField,
    StructType,
)
from pytest import fixture

from nlsn.nebula.spark_transformers import IterableSize


@fixture(scope="module", name="df_input")
def _get_input_data(spark):
    fields = [
        StructField("array", ArrayType(IntegerType(), True), True),
        StructField("map", MapType(StringType(), IntegerType(), True), True),
        StructField("expected", IntegerType(), True),
    ]
    schema = StructType(fields)

    data = [
        ([1, 2, 3], {"key1": 4, "key2": 5, "key3": 1}, 3),
        ([4, 5], {"key1": 3, "key2": 2}, 2),
        ([6], {"key1": 2}, 1),
        (None, None, -1),
    ]
    # If the value is null, F.size returns -1
    return spark.createDataFrame(data, schema).persist()


@pytest.mark.parametrize("input_col", ["array", "map"])
def test_array_size(df_input, input_col):
    """Test IterableSize."""
    t = IterableSize(input_col=input_col, output_col="size")
    df_chk = t.transform(df_input)
    n_diff = df_chk.filter(F.col("expected") != F.col("size")).count()
    assert n_diff == 0
