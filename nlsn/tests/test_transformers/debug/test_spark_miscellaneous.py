"""Unit-Test for miscellaneous transformers."""

import pytest
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql.types import StringType, StructField, StructType

from nlsn.nebula.spark_transformers import LocalCheckpoint, PrintSchema, Show


@pytest.fixture(scope="module", name="df_input")
def _get_input_df(spark):
    fields = [
        StructField("c1", StringType(), True),
        StructField("c2", StringType(), True),
    ]
    data = [
        ["a", "aa"],
        ["b", "bb"],
        ["c", "cc"],
    ]
    schema = StructType(fields)
    return spark.createDataFrame(data, schema=schema).persist()


@pytest.mark.parametrize("eager", [True, False])
def test_local_check_point(df_input, eager: bool):
    """Test LocalCheckpoint transformer."""
    t = LocalCheckpoint(eager=eager)
    df_chk = t.transform(df_input)
    assert_df_equality(df_input, df_chk)


@pytest.mark.parametrize("columns", ["c1", None])
def test_show(df_input, columns):
    """Test Show transformer."""
    t = Show(columns=columns)
    df_chk = t.transform(df_input)
    assert_df_equality(df_input, df_chk)


@pytest.mark.parametrize("columns", ["c1", None])
def test_print_schema(df_input, columns):
    """Test PrintSchema transformer."""
    t = PrintSchema(columns=columns)
    df_chk = t.transform(df_input)
    assert df_input is df_chk  # Assert equality
