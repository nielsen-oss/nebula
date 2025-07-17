"""Unit-test for Join."""

import pytest
from chispa import assert_df_equality
from pyspark.sql.types import FloatType, StringType, StructField, StructType

from nlsn.nebula.spark_transformers import Join
from nlsn.nebula.storage import nebula_storage as ns


def test_join_error():
    """Test Join transformer with wrong 'how'."""
    with pytest.raises(ValueError):
        Join(table="x", on="x", how="wrong")


@pytest.fixture(scope="module", name="df_left")
def _get_df_left(spark):
    fields = [
        StructField("c1", StringType(), True),
        StructField("c2", StringType(), True),
        StructField("c_left", FloatType(), True),
    ]

    data = [
        ["a", "aa", 1.0],
        ["b", "bb", 2.0],
        ["c", "cc", 3.0],
    ]

    return spark.createDataFrame(data, schema=StructType(fields)).cache()


@pytest.fixture(scope="module", name="df_right")
def _get_df_right(spark):
    fields = [
        StructField("c1", StringType(), True),
        StructField("c2", StringType(), True),
        StructField("c_right", FloatType(), True),
    ]

    data = [
        ["a", "aa", 10.0],
        ["b", "bb", 20.0],
        ["d", "dd", 30.0],
    ]

    return spark.createDataFrame(data, schema=StructType(fields)).cache()


@pytest.mark.parametrize(
    "on, how, broadcast", [("c1", "left", True), (["c1", "c2"], "inner", False)]
)
def test_join(df_left, df_right, on, how: str, broadcast: bool):
    """Test Join."""
    # Drop "c2" from left df otherwise is duplicated and chispa crashes
    c2d = [] if "c2" in on else ["c2"]

    ns.clear()
    ns.set("df_right", df_right)

    t = Join(table="df_right", on=on, how=how, broadcast=broadcast)
    df_chk = t.transform(df_left.drop(*c2d))

    df_exp = df_left.drop(*c2d).join(df_right, on=on, how=how)

    assert_df_equality(df_chk, df_exp, ignore_row_order=True)

    ns.clear()
