"""Unit-test for CpuInfo."""

from pyspark.sql.types import IntegerType, StructField, StructType

from nlsn.nebula.spark_transformers import CpuInfo


def test_cpu_info(spark):
    """Test CpuInfo transformer."""
    schema = StructType([StructField("c1", IntegerType(), True)])
    df = spark.createDataFrame([[1]], schema=schema)

    t = CpuInfo(n_partitions=10)
    t.transform(df)
