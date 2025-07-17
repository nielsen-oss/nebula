"""Unit-test for BooleanMarker."""

import pyspark.sql.functions as F
import pytest
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql.types import FloatType, StringType, StructField, StructType

from nlsn.nebula.spark_transformers import BooleanMarker


@pytest.fixture(scope="module", name="df_input")
def _get_input_data(spark):
    fields = [
        StructField("input_string", StringType(), True),
        StructField("input_float", FloatType(), True),
    ]

    schema = StructType(fields)

    data = [
        ["a", 0.0],
        ["a1", 1.0],
        ["a", 2.0],
        ["a1", 3.0],
        ["b", 4.0],
        ["b", 5.0],
        ["b1", 6.0],
        ["c1", 7.0],
        ["c1", 8.0],
        ["c1", 9.0],
        ["", None],
        ["", 10.0],
        ["  ", None],
        ["  ", 11.0],
        [None, None],
    ]
    return spark.createDataFrame(data, schema=schema).persist()


# fmt: off
_params = [
    {"operator": "isin", "value": {"a", "b"}},
    {"operator": "startswith", "value": "a"},
]


# fmt: on


@pytest.mark.parametrize("kwargs", _params)
def test_boolean_marker(df_input, kwargs):
    """Test BooleanMarker."""
    t = BooleanMarker(input_col="input_string", output_col="output", **kwargs)

    df_chk = t.transform(df_input)

    # Check if the output values are True / False. None must not be present.
    results = df_chk.select("output").rdd.flatMap(lambda x: x).collect()
    assert set(results) == {True, False}

    # Create the expected dataframe.
    op = kwargs["operator"]
    value = kwargs["value"]
    cond = getattr(F.col("input_string"), op)(value)
    exp_clause = F.when(cond, F.lit(True)).otherwise(F.lit(False))
    df_exp = df_input.withColumn("output", exp_clause)

    assert_df_equality(df_chk, df_exp, ignore_row_order=True)
