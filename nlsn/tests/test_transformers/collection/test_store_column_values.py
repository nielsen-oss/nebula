"""Unit-test for StoreColumnValues in spark."""

from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from nlsn.nebula.shared_transformers import StoreColumnValues
from nlsn.nebula.storage import nebula_storage as ns


def test_pandas_polars_store_column_values(spark):
    """Test StoreColumnValues transformer in spark."""
    fields = [
        StructField("idx", IntegerType(), True),
        StructField("b", StringType(), True),
    ]

    data = [
        [0, "a"],
        [1, "b"],
        [2, "b"],
    ]

    df_input = spark.createDataFrame(data, schema=StructType(fields)).persist()

    ns.clear()
    try:
        StoreColumnValues(key="test", column="b", as_type="set").transform(df_input)
        chk = ns.get("test")
        exp = {"a", "b"}
        assert chk == exp

    finally:
        ns.clear()
