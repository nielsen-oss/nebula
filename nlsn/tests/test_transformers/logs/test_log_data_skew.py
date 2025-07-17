"""Unit-test for LogDataSkew."""

from pyspark.sql.types import IntegerType, StructField, StructType

from nlsn.nebula.spark_transformers import LogDataSkew


def test_log_data_skew(spark):
    """Test LogDataSkew transformer."""
    schema = StructType([StructField("c1", IntegerType(), True)])
    data = [[i] for i in range(100)]
    df = spark.createDataFrame(data, schema=schema)

    t = LogDataSkew()
    t.transform(df)
