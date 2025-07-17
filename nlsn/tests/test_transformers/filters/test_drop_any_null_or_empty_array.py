"""Unit-test for DropAnyNullOrEmptyArray."""

from functools import reduce
from operator import or_

import pytest
from chispa import assert_df_equality
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from nlsn.nebula.spark_transformers.filters import DropAnyNullOrEmptyArray
from nlsn.nebula.spark_util import get_schema_as_str


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    """Get input dataframe."""
    fields = [
        StructField("idx", IntegerType(), True),
        StructField("c1", ArrayType(StringType()), True),
        StructField("c2", ArrayType(StringType()), True),
        StructField("c3", StringType(), True),
    ]

    data = [
        [0, ["a"], ["b"], "c"],
        [1, ["a"], [], "c"],
        [2, ["a"], None, "c"],
        [3, [], None, "c"],
        [4, [], [], "c"],
        [5, None, None, "c"],
        [6, None, ["b"], "c"],
        [7, [None], ["b"], "c"],
        [8, ["a"], ["b"], "c"],
    ]

    return spark.createDataFrame(data, schema=StructType(fields))


def test_drop_any_null_or_empty_array(df_input):
    """Test DropAnyNullOrEmptyArray transformer."""
    t = DropAnyNullOrEmptyArray(columns=["c1", "c2"])
    df_out = t.transform(df_input)

    select_cols = df_out.select(["c1", "c2"]).columns

    # Get a dict(name -> type) like 'col_1' -> 'int'.
    map_field_types = dict(get_schema_as_str(df_input, False))

    # Keep only the arrayType columns for determining the condition.
    valid_cols = [i for i in select_cols if map_field_types[i] == "array"]

    # Create the expected condition.
    exp_cond = [F.col(i).isNull() for i in valid_cols]
    exp_cond += [F.size(i) == 0 for i in valid_cols]
    cond = reduce(or_, exp_cond)

    # pylint: disable=invalid-unary-operand-type
    df_exp = df_input.filter(~cond)
    assert_df_equality(df_out, df_exp, ignore_row_order=True)
