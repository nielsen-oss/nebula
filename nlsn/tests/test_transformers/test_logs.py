"""Unit-test for Logs Transformers."""

from pyspark.sql.types import IntegerType, StructField, StructType

from nlsn.nebula.transformers.logs import *


def test_cpu_info(spark):
    """Test CpuInfo."""
    schema = StructType([StructField("c1", IntegerType(), True)])
    df = spark.createDataFrame([[1]], schema=schema)

    t = CpuInfo(n_partitions=10)
    t.transform(df)
