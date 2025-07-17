"""Unit-test for EmptyArrayToNull."""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from nlsn.nebula.spark_transformers.arrays import EmptyArrayToNull

_DATA = [
    ("row_1", [0, 1, None, 2]),
    ("row_2", [0, None, None]),
    ("row_3", [0]),
    ("row_4", [None]),
    ("row_5", []),
    ("row_6", None),
]


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    """Creates initial DataFrame."""
    fields = [
        StructField("col_string", StringType()),
        StructField("col_array", ArrayType(IntegerType())),
    ]
    schema: StructType = StructType(fields)
    return spark.createDataFrame(_DATA, schema=schema).persist()


@pytest.mark.parametrize("columns, glob", [("col_array", None), (None, "*")])
def test_empty_array_to_null(df_input, columns, glob):
    """Test EmptyArrayToNull."""
    t = EmptyArrayToNull(columns=columns, glob=glob)
    df_chk = t.transform(df_input)

    assert df_input.columns == df_chk.columns

    cond_null = F.col("col_array").isNull()
    cond_empty = F.size("col_array") == 0

    # Initial rows with null values
    n_start_null_values: int = df_input.filter(cond_null).count()
    n_start_empty_array: int = df_input.filter(cond_empty).count()

    n_end_null_values: int = df_chk.filter(cond_null).count()
    n_end_empty_array: int = df_chk.filter(cond_empty).count()
    assert n_end_empty_array == 0

    assert n_end_null_values == (n_start_null_values + n_start_empty_array)
