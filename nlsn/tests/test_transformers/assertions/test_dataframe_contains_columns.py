"""Unit-test for DataFrameContainsColumns."""

import pytest
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from nlsn.nebula.spark_transformers import DataFrameContainsColumns


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
    return spark.createDataFrame(data, schema=StructType(fields))


@pytest.mark.parametrize(
    "columns, error",
    [
        ([], False),
        ("col1", False),
        (["col1"], False),
        (["col1", "col2"], False),
        (["col1", "col2", "col3"], True),
        ("col3", True),
        (["col3"], True),
    ],
)
def test_dataframe_contains_columns(df_input, columns, error: bool):
    """Test DataFrameContainsColumns transformer."""
    t = DataFrameContainsColumns(columns=columns)
    if error:
        with pytest.raises(AssertionError):
            t.transform(df_input)
    else:
        t.transform(df_input)
