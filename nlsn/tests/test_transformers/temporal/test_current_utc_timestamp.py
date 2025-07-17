"""Unit-test for CurrentUtcTimestamp."""

import pyspark.sql.functions as F
from pyspark.sql.types import StringType, StructField, StructType

from nlsn.nebula.spark_transformers import CurrentUtcTimestamp


def test_current_utc_timestamp(spark):
    """Test CurrentUtcTimestamp transformer."""
    fields = [
        StructField("c1", StringType(), True),
    ]
    df = spark.createDataFrame([["x"]], schema=StructType(fields))

    df = df.withColumn("start", F.current_timestamp())
    df = CurrentUtcTimestamp(column="chk").transform(df)
    df = df.withColumn("end", F.current_timestamp())

    cond = (F.col("chk") >= F.col("start")) & (F.col("chk") <= F.col("end"))
    n_rows = df.filter(cond).count()

    assert n_rows == 1
