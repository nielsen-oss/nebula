"""Unit-test for AggregateAndAddAsLiteral."""

# pylint: disable=ungrouped-imports

import random

import numpy as np
import pyspark.sql.functions as F
import pytest
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql.types import FloatType, StructField, StructType

from nlsn.nebula.spark_transformers._constants import ALLOWED_GROUPBY_AGG
from nlsn.nebula.spark_transformers.aggregations import AggregateAndAddAsLiteral

_not_to_test = {"collect_list", "collect_set", "grouping"}  # like in the __init__
_groupby_agg_for_test = [i for i in ALLOWED_GROUPBY_AGG if i not in _not_to_test]


@pytest.fixture(scope="module", name="df_input")
def _get_input_data(spark):
    fields = [StructField("input_value", FloatType(), True)]

    schema = StructType(fields)

    data = np.random.randint(0, 10, 20).astype(np.float32).reshape(-1, 1).tolist()
    return spark.createDataFrame(data, schema=schema).persist()


@pytest.mark.parametrize("aggregation", random.choices(_groupby_agg_for_test, k=3))
def test_aggregate_and_add_as_literal(df_input, aggregation):
    """Test AggregateAndAddAsLiteral."""
    cast_long = None if np.random.random() < 0.5 else "long"
    output_col = None if np.random.random() < 0.5 else "output_value"

    t = AggregateAndAddAsLiteral(
        input_col="input_value",
        aggregation=aggregation,
        output_col=output_col,
        cast=cast_long,
    )

    df_chk = t.transform(df_input)

    if aggregation in {"first", "last", "approx_count_distinct"}:
        # Result depends on row order
        df_chk.collect()
        return

    exp_col = "input_value" if output_col is None else output_col
    func = getattr(F, aggregation)("input_value")
    value = F.lit(df_input.agg(func).collect()[0][0])
    if cast_long:
        value = value.cast(cast_long)

    df_exp = df_input.withColumn(exp_col, value)

    assert_df_equality(df_chk, df_exp, ignore_row_order=True)
