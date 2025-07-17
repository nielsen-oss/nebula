"""Unit-test for Count."""

import pytest
from chispa import assert_df_equality
from pyspark.sql.types import IntegerType, StructField, StructType

from nlsn.nebula.spark_transformers import Count
from nlsn.nebula.storage import nebula_storage as ns


def test_count_wrong_store_key():
    """Test Count transformer again a non-hashable store_key."""
    with pytest.raises(AssertionError):
        Count(store_key={})


def test_count(spark):
    """Test Count transformer."""
    ns.clear()
    ns.allow_overwriting()

    store_key = "test_count"

    n_rows = 10
    schema = StructType([StructField("c1", IntegerType(), True)])
    df = spark.createDataFrame([[i] for i in range(n_rows)], schema=schema)

    t = Count(persist=True, store_key=store_key)
    df_out = t.transform(df)

    assert_df_equality(df, df_out, ignore_row_order=True)

    count_chk = ns.get(store_key)
    ns.clear()
    assert count_chk == n_rows
