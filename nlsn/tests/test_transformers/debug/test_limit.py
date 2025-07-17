"""Unit-test for Limit."""

import pytest
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from nlsn.nebula.spark_transformers import Limit

_data = [
    (1, "a"),
    (2, "b"),
    (3, "c"),
    (4, "d"),
    (5, "e"),
]

_N_INPUT: int = len(_data)


@pytest.fixture(scope="module", name="df_input")
def _get_input_df(spark):
    fields = [
        StructField("col1", IntegerType(), True),
        StructField("col2", StringType(), True),
    ]
    return spark.createDataFrame(_data, schema=StructType(fields)).persist()


@pytest.mark.parametrize("n", [2, -1])
def test_limit(df_input, n: int):
    """Test Limit transformer."""
    t = Limit(n=n)

    df_chk = t.transform(df_input)
    count: int = df_chk.count()

    if n < 0:
        assert count == _N_INPUT
        return

    if n <= _N_INPUT:
        assert count == n
    else:
        assert count == _N_INPUT
