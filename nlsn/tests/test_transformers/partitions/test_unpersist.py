"""Unit-test for Unpersist and ClearCache transformers."""

import pytest
from pyspark.sql.types import StringType, StructField, StructType

from nlsn.nebula.spark_transformers.partitions import ClearCache, UnPersist


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    fields = [
        StructField("a", StringType(), True),
        StructField("b", StringType(), True),
        StructField("c", StringType(), True),
    ]

    data = [[f"a{i}", f"b{i}", f"c{i}"] for i in range(10)]
    schema = StructType(fields)
    return spark.createDataFrame(data, schema=schema).persist()


@pytest.mark.parametrize("blocking", [True, False])
def test_un_persist(df_input, blocking: bool):
    """Test Unpersist."""
    t = UnPersist(blocking=blocking)
    df_out = t.transform(df_input.persist())
    assert not df_input.is_cached
    assert not df_out.is_cached


def test_clear_cache(spark, df_input):
    """Test ClearCache."""
    spark.sql("DROP TABLE IF EXISTS tbl1")
    spark.sql("CREATE TABLE tbl1 (name STRING, age INT) USING parquet")
    ClearCache().transform(df_input)

    tables = [i.name for i in spark.catalog.listTables()]

    for tbl in tables:
        assert not spark.catalog.isCached(tbl)
