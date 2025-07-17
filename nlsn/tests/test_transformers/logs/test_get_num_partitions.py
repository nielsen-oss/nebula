"""Unit-test for GetNumPartitions."""

import pytest
from pyspark.sql.types import IntegerType, StructField, StructType

from nlsn.nebula.spark_transformers import GetNumPartitions
from nlsn.nebula.storage import nebula_storage as ns


def test_count_wrong_store_key():
    """Test Count transformer again a non-hashable store_key."""
    with pytest.raises(AssertionError):
        GetNumPartitions(store_key={})


def test_get_num_partitions(spark):
    """Test GetNumPartitions transformer."""
    ns.clear()
    ns.allow_overwriting()

    store_key = "test_num_partitions"

    num_partitions = 2
    schema = StructType([StructField("c1", IntegerType(), True)])
    data = [[i] for i in range(num_partitions * 10)]
    df = spark.createDataFrame(data, schema=schema).repartition(num_partitions)

    t = GetNumPartitions(store_key=store_key)
    df_out = t.transform(df)

    assert df is df_out

    n_chk = ns.get(store_key)
    ns.clear()
    assert n_chk == num_partitions
