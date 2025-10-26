"""Unit-test for SqlFunction."""

import pytest
from chispa import assert_df_equality
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType, StructField, StructType

from nlsn.nebula.spark_transformers import SqlFunction


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    """Creates initial DataFrame."""
    fields = [StructField("data", ArrayType(IntegerType()))]
    data = [([2, 1, None, 3],), ([1],), ([],), (None,)]
    return spark.createDataFrame(data, StructType(fields)).persist()


@pytest.mark.parametrize("asc", [True, False])
def test_sql_function(df_input, asc: bool):
    """Test SqlFunction."""
    t = SqlFunction(
        column="result", function="sort_array", args=["data"], kwargs={"asc": asc}
    )
    df_chk = t.transform(df_input)

    df_exp = df_input.withColumn("result", F.sort_array("data", asc=asc))
    assert_df_equality(df_chk, df_exp, ignore_row_order=True)


def test_sql_function_no_args(df_input):
    """Test SqlFunction w/o any arguments."""
    t = SqlFunction(column="result", function="rand")
    df_chk = t.transform(df_input)

    n_null: int = df_chk.filter(F.col("result").isNull()).count()
    assert n_null == 0
