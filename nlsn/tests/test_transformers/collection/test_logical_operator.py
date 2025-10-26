"""Unit-test for LogicalOperator."""

import operator as py_operator
from functools import reduce

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import BooleanType, StructField, StructType
from pytest import fixture

from nlsn.nebula.spark_transformers import LogicalOperator


@fixture(scope="module", name="df_input")
def _get_input_data(spark: SparkSession):
    fields = [
        StructField("c1", BooleanType(), True),
        StructField("c2", BooleanType(), True),
        StructField("c3", BooleanType(), True),
    ]

    data = [
        (True, True, True),
        (False, False, False),
        (True, False, False),
        (True, False, None),
        (True, True, None),
        (False, False, None),
        (True, False, None),
        (None, None, None),
    ]

    return spark.createDataFrame(data, StructType(fields)).cache()


@pytest.mark.parametrize("op", ["AND", "or"])
def test_logical_operator(df_input, op):
    """Test LogicalOperator transformer."""
    input_cols = df_input.columns
    t = LogicalOperator(operator=op, glob="*", output_col="out")
    df_chk = t.transform(df_input)

    collected = [i.asDict() for i in df_chk.collect()]

    cond = py_operator.and_ if op.lower() == "and" else py_operator.or_

    for row in collected:
        chk = row["out"]
        input_data = [row[i] for i in input_cols]

        li_bool = [False if i is None else i for i in input_data]

        exp = reduce(cond, li_bool)
        assert chk == exp
