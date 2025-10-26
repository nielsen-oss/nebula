"""Unit-test for TakeMinMaxOverWindow.

For other tests, refer to the 'take_min_max_over_window' ones.
"""

import pyspark.sql.functions as F
import pytest
from chispa import assert_df_equality
from pyspark.sql import Window
from pyspark.sql.types import BooleanType, StringType, StructField, StructType

from nlsn.nebula.spark_transformers.aggregations import TakeMinMaxOverWindow

# The last two boolean values indicate whether the row must be kept
# after the transformation with operator set to "min" and "max" respectively.
_DATA = [
    ["a1", "b1", "2020-09-01 00:00:01", True, False],
    ["a1", "b1", "2020-09-01 00:00:02", False, True],

    ["a2", "b2", "2020-09-01 00:00:02", True, True],

    ["a3", "b3", None, False, False],

    ["", "", "2020-09-01 00:00:02", True, True],

    ["a4", "b4", "2020-09-01 00:00:01", True, False],
    ["a4", "b4", "2020-09-01 00:00:02", False, True],
    ["a4", "b4", None, False, False],

    [None, None, None, False, False],
]


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    fields = [
        StructField("c1", StringType(), True),
        StructField("c2", StringType(), True),
        StructField("dt", StringType(), True),
        StructField("keep_min", BooleanType(), True),
        StructField("keep_max", BooleanType(), True),
    ]
    fmt = "yyyy-MM-dd HH:mm:ss"
    df = spark.createDataFrame(_DATA, schema=StructType(fields))
    return df.withColumn("dt", F.to_timestamp("dt", fmt)).persist()


@pytest.mark.parametrize("op", ["min", "max"])
def test_take_min_max_over_window_filter(df_input, op: str):
    """Test TakeMinMaxOverWindow transformer with perform='filter'."""
    windowing_cols = ["c1", "c2"]
    t = TakeMinMaxOverWindow(
        partition_cols=windowing_cols, column="dt", operator=op, perform="filter"
    )
    df_chk = t.transform(df_input).persist()

    if op == "min":
        col_idx = -2
    elif op == "max":
        col_idx = -1
    else:
        raise RuntimeError

    # Assert no duplicates in the subset of the windowing columns
    n_unique_chk = df_chk.count()
    n_unique_exp = df_chk.drop_duplicates(subset=windowing_cols).count()
    assert n_unique_chk == n_unique_exp

    # Assert no null values in the specified input column
    n_nulls = df_chk.filter(F.col("dt").isNull()).count()
    assert n_nulls == 0

    n_exp = len([i for i in _DATA if i[col_idx]])
    n_chk = df_chk.filter(F.col(f"keep_{op}")).count()

    assert n_exp == n_chk


@pytest.mark.parametrize("op", ["min", "max"])
def test_take_min_max_over_window_replace(df_input, op: str):
    """Test TakeMinMaxOverWindow transformer with perform='replace'."""
    windowing_cols = ["c1", "c2"]
    t = TakeMinMaxOverWindow(
        partition_cols=windowing_cols, column="dt", operator=op, perform="replace"
    )
    df_chk = t.transform(df_input)

    w = Window.partitionBy(*windowing_cols)
    df_exp = df_input.withColumn("dt", getattr(F, op)("dt").over(w))
    assert_df_equality(df_chk, df_exp, ignore_row_order=True)
