"""Unit-test for AssertRowsAreDistinct."""

import pytest
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql.types import IntegerType, StructField, StructType

from nlsn.nebula.spark_transformers import AssertRowsAreDistinct

_fields = [
    StructField("c1", IntegerType(), True),
    StructField("c2", IntegerType(), True),
]

_schema = StructType(_fields)


@pytest.mark.parametrize(
    "exception, columns, data, raise_error, msg",
    [
        # Duplicated. Select specific columns.
        (True, ["c1", "c2"], [[1, 2], [1, 2]], True, None),
        # Rows are duplicated, but do not raise any error
        (False, ["c1", "c2"], [[1, 2], [1, 2]], False, "duplicated"),
        # Duplicated. Select all columns.
        (True, None, [[1, 2], [1, 2]], True, ""),
        # Duplicated. Select all columns by default.
        (True, None, [[1, 2], [1, 2]], True, "duplicated"),
        # No error, rows are distinct.
        (False, ["c1", "c2"], [[1, 1], [2, 2]], True, ""),
        # Subset c1 is not distinct.
        (True, ["c1"], [[1, 1], [1, 2]], True, None),
    ],
)
def test_assert_rows_are_distinct(spark, exception, columns, data, raise_error, msg):
    """Test AssertRowsAreDistinct transformer."""
    df = spark.createDataFrame(data, schema=_schema)
    t = AssertRowsAreDistinct(
        columns=columns, raise_error=raise_error, log_level="info", log_message=msg
    )

    if exception:
        with pytest.raises(AssertionError):
            t.transform(df)
    else:
        df_chk = t.transform(df)
        assert_df_equality(df, df_chk, ignore_row_order=True)


def test_assert_rows_are_distinct_error():
    """Test AssertRowsAreDistinct transformer with wrong parameters."""
    with pytest.raises(AssertionError):
        AssertRowsAreDistinct(log_message="message")
