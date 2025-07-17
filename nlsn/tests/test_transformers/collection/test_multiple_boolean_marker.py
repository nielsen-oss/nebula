"""Unit-test for MultipleBooleanMarker."""

import operator
from functools import reduce
from random import choice, randint, sample
from typing import Any, Generator

import pytest
from chispa import assert_df_equality
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StructField, StructType

from nlsn.nebula.spark_transformers import MultipleBooleanMarker
from nlsn.nebula.spark_util import get_spark_condition, null_cond_to_false

DATA = [{f"c{i}": randint(0, 1 << 4) for i in range(16)} for _ in range(20)]
CONDITIONS = [
    [
        {
            "column": column,
            "operator": choice(["eq", "ne", "le", "lt", "ge", "gt"]),
            "value": randint(0, 1 << 4),
        }
        for column in sample(DATA[0].keys(), randint(2, 1 << 3))
    ]
    for _ in range(3)
]
_ASSEMBLY_OPERATORS = {
    "&": operator.and_,
    "|": operator.or_,
    "and": operator.and_,
    "or": operator.or_,
}
_ASSEMBLIES = [
    [choice(list(_ASSEMBLY_OPERATORS.keys())) for _ in range(len(cond) - 1)]
    for cond in CONDITIONS
]


@pytest.fixture(scope="module", name="df_input_dynamic")
def _get_df_input(spark) -> DataFrame:
    return spark.createDataFrame(DATA).persist()


@pytest.mark.parametrize("out_col", ["out", "out2"])
@pytest.mark.parametrize("cond_number", list(range(len(CONDITIONS))))
def test_multiple_boolean_marker_dynamic(
    df_input_dynamic,
    out_col: str,
    cond_number: int,
):
    """Test MultipleBooleanMarker using dynamic parameters."""
    cond, assembly = CONDITIONS[cond_number], _ASSEMBLIES[cond_number]
    t = MultipleBooleanMarker(
        output_col=out_col,
        conditions=cond,
        logical_operators=assembly,
    )

    df_chk = t.transform(df_input_dynamic)
    assembly_iter = iter(assembly)
    cond_iter: Generator[Any, None, None] = (
        (x["column"], x["operator"], x["value"]) for x in cond
    )
    # pylint: disable=unnecessary-lambda
    full_cond = reduce(
        lambda a, b: _ASSEMBLY_OPERATORS[next(assembly_iter)](a, b),
        [
            get_spark_condition(df_input_dynamic, col, op, value=val)
            for col, op, val in cond_iter
        ],
    )

    clause_exp = null_cond_to_false(full_cond)
    df_exp = df_input_dynamic.withColumn(out_col, clause_exp)
    assert_df_equality(
        df_chk,
        df_exp,
        ignore_column_order=True,
        ignore_row_order=True,
    )


def test_multiple_boolean_marker_hardcoded(spark):
    """Test MultipleBooleanMarker using hardcoded parameters."""
    fields = [
        StructField("c1", IntegerType(), True),
        StructField("c2", IntegerType(), True),
        StructField("c3", IntegerType(), True),
        StructField("c4", IntegerType(), True),
    ]
    data = [
        [1, 3, None, 10],
        [2, 5, 10, 11],
        [2, 5, 10, 0],
        [2, 2, 10, 11],
        [2, 2, 10, 0],
        [4, 0, None, 12],
        [0, 4, None, 13],
        [3, 4, None, 13],
        [0, 0, 15, 14],
        [None, None, 16, 18],
        [None, 0, 17, 17],
        [0, None, None, None],
        [None, 3, None, 20],
        [4, None, None, 20],
    ]
    schema = StructType(fields)
    df_input = spark.createDataFrame(data, schema=schema).persist()

    conditions = [
        {
            "column": "c1",
            "operator": "le",  # Less or equal 3
            "value": 3,
        },
        {
            "column": "c2",
            "operator": "isNotNull",
        },
        {
            "column": "c3",
            "operator": "isNull",
        },
        {
            "column": "c4",
            "operator": "gt",  # Greater than
            "comparison_column": "c1",
        },
    ]
    logical_operators = ["and", "|", "&"]

    t = MultipleBooleanMarker(
        output_col="chk",
        conditions=conditions,
        logical_operators=logical_operators,
    )

    df_chk = t.transform(df_input).persist()

    # Assert that at least one True and one False exist
    n_rows_true: int = df_chk.filter(F.col("chk")).count()
    n_rows_false: int = df_chk.filter(~F.col("chk")).count()
    assert n_rows_true > 0
    assert n_rows_false > 0

    # Assert no null in the output column
    n_rows_null = df_chk.filter(F.col("chk").isNull()).count()
    assert n_rows_null == 0

    cond = F.col("c1") <= F.lit(3)
    cond &= F.col("c2").isNotNull()
    cond |= F.col("c3").isNull()
    cond &= F.col("c4") > F.col("c1")
    cond = null_cond_to_false(cond)

    df_exp = df_input.withColumn("chk", cond)
    assert_df_equality(df_chk, df_exp, ignore_row_order=True)


def test_multiple_boolean_error():
    """Test MultipleBooleanMarker with wrong input parameters."""
    with pytest.raises(ValueError):
        MultipleBooleanMarker(
            output_col="x",
            conditions=CONDITIONS[0],
            logical_operators=_ASSEMBLIES[0][1:],
        )
