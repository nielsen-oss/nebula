"""Unit-test for ReplaceDotInColumnNames."""

import pytest
from pyspark.sql.types import FloatType, StringType, StructField, StructType

from nlsn.nebula.spark_transformers import ReplaceDotInColumnNames


def test_replace_dot_in_column_names_wrong_params():
    """Test ReplaceDotInColumnNames with wrong parameters."""
    with pytest.raises(AssertionError):
        ReplaceDotInColumnNames(replacement=".")
    with pytest.raises(AssertionError):
        ReplaceDotInColumnNames(replacement="hel.lo")


def test_replace_dot_in_column_names(spark):
    """Test ReplaceDotInColumnNames."""
    fields = [
        StructField("computer.use", StringType(), True),
        StructField("tab_use", FloatType(), True),
    ]
    schema = StructType(fields)
    df_input = spark.createDataFrame([["a", 0.0], ["b", 1.0]], schema=schema)

    expected_columns = ["computer_use", "tab_use"]
    t = ReplaceDotInColumnNames(replacement="_")
    df_out = t.transform(df_input)
    assert df_out.columns == expected_columns
