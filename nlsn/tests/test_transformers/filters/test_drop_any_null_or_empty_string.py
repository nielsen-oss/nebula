"""Unit-test for DropAnyNullOrEmptyString."""

from functools import reduce
from operator import or_

import pytest
from chispa import assert_df_equality
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from nlsn.nebula.spark_transformers.filters import DropAnyNullOrEmptyString
from nlsn.nebula.spark_util import get_schema_as_str


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    """Get input dataframe."""
    fields = [
        StructField("idx", IntegerType(), True),
        StructField("c1", StringType(), True),
        StructField("c2", StringType(), True),
    ]

    data = [
        [0, "a", "b"],
        [1, "a", "  b"],
        [2, "  a  ", "  b  "],
        [3, "", ""],
        [4, "   ", "   "],
        [5, None, None],
        [6, " ", None],
        [7, "", None],
        [8, "a", None],
        [9, "a", ""],
        [10, "   ", "b"],
        [11, "a", None],
        [12, None, "b"],
    ]

    return spark.createDataFrame(data, schema=StructType(fields))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"columns": ["c1", "c2"], "trim": False},
        {"glob": "*", "trim": True},
    ],
)
def test_drop_any_null_or_empty_array(df_input, kwargs):
    """Test DropAnyNullOrEmptyString transformer."""
    t = DropAnyNullOrEmptyString(**kwargs)
    df_out = t.transform(df_input)

    trim = kwargs.pop("trim", False)  # default trim=False in the transformer.
    # Extract the selected columns.
    kw_values = list(kwargs.values())  # can be list or list(list))
    select_cols = df_out.select(*kw_values).columns

    # Get a dict(name -> type) like 'col_1' -> 'int'.
    map_field_types = dict(get_schema_as_str(df_input, False))

    # Keep only the StringType columns for determining the condition.
    valid_cols = [i for i in select_cols if map_field_types[i] == "string"]

    # Create the spark condition w/ or w/o trim.
    spark_cond = [F.trim(c) if trim else F.col(c) for c in valid_cols]

    # Create the expected condition
    exp_cond = [i.isNull() | (i == "") for i in spark_cond]
    cond = reduce(or_, exp_cond)

    df_exp = df_input.filter(~cond)
    assert_df_equality(df_out, df_exp, ignore_row_order=True)
