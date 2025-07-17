"""Unit-test for AssertNotEmpty."""

import pytest
from chispa import assert_df_equality
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from nlsn.nebula.spark_transformers import AssertNotEmpty


@pytest.fixture(scope="module", name="df_input")
def _get_input_df(spark):
    fields = [
        StructField("col1", IntegerType(), True),
        StructField("col2", StringType(), True),
    ]
    data = [
        (1, "a"),
        (2, "b"),
        (3, "c"),
        (4, "d"),
        (5, "e"),
    ]
    return spark.createDataFrame(data, schema=StructType(fields)).persist()


@pytest.mark.parametrize("error", [True, False])
def test_assert_not_empty(df_input, error: bool):
    """Test AssertNotEmpty transformer."""
    t = AssertNotEmpty()

    if error:
        with pytest.raises(AssertionError):
            t.transform(df_input.limit(0))
    else:
        df_chk = t.transform(df_input)
        assert_df_equality(df_chk, df_input, ignore_row_order=True)
