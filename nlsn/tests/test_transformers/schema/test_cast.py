"""Unit-test for Cast."""

from typing import List, Tuple

from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    MapType,
    StringType,
    StructField,
    StructType,
)

from nlsn.nebula.spark_transformers import Cast
from nlsn.nebula.spark_util import get_schema_as_str


def test_cast(spark):
    """Test Cast transformer."""
    data = [
        ["9", "11", 1.0, 1, "2022-01-01", 1, [], {"a": 0, "b": 1}],
        [None, "c", 2.0, 2, None, 0, [], None],
        ["10", "", None, None, "2022-02-02", 0, [], {"a": -1, "b": 0}],
        [None, None, 3.0, 3, "2022-03-03", None, [], None],
        [None, "12", 4.0, 4, "2022-03-03", 1, None, {"a": 1, "b": 2}],
    ]

    fields = [
        StructField("a", StringType(), True),
        StructField("b", StringType(), True),
        StructField("c", FloatType(), True),
        StructField("d", IntegerType(), True),
        StructField("e", StringType(), True),
        StructField("f", IntegerType(), True),
        StructField("g", ArrayType(IntegerType()), True),
        StructField("h", MapType(StringType(), IntegerType()), True),
    ]

    schema = StructType(fields)

    df = spark.createDataFrame(data, schema=schema)
    # int == integer in spark, but in this test keep int,
    # because <get_schema_as_str> returns "int".
    dict_cast = {
        "a": "int",  # "int" not "integer"
        "b": "float",
        "c": "string",
        "e": "date",
        "f": "boolean",
        "g": "string",
        "h": "string",
    }

    t = Cast(cast=dict_cast)

    df_out = t.transform(df)

    schema_chk_full: List[Tuple[str, str]] = get_schema_as_str(df_out, True)

    schema_chk = [tuple([k, v]) for k, v in schema_chk_full if k in dict_cast]
    assert dict(schema_chk) == dict_cast
