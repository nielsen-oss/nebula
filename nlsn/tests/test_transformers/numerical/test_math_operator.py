"""Unit-test for MathOperator."""

import operator as py_operator

import pytest
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StructField, StructType

from nlsn.nebula.spark_transformers import MathOperator

_CONST: float = 11.0


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    fields = [
        StructField("col1", IntegerType(), True),
        StructField("col2", IntegerType(), True),
    ]
    data = [
        [1, 3],
        [2, 5],
        [4, 0],
        [0, 4],
        [0, 0],
        [None, None],
        [None, 0],
        [0, None],
        [None, 3],
        [4, None],
    ]
    schema = StructType(fields)
    df = spark.createDataFrame(data, schema=schema)

    # Create the expected columns for the test 'test_math_operator_single_operation'
    df = (
        df.withColumn("add_column", F.col("col1") + F.col("col2"))
        .withColumn("sub_column", F.col("col1") - F.col("col2"))
        .withColumn("mul_column", F.col("col1") * F.col("col2"))
        .withColumn("div_column", F.col("col1") / F.col("col2"))
        .withColumn("pow_column", F.col("col1") ** F.col("col2"))
        .withColumn("add_constant", F.col("col1") + F.lit(_CONST))
        .withColumn("sub_constant", F.col("col1") - F.lit(_CONST))
        .withColumn("mul_constant", F.col("col1") * F.lit(_CONST))
        .withColumn("div_constant", F.col("col1") / F.lit(_CONST))
        .withColumn("pow_constant", F.col("col1") ** F.lit(_CONST))
    )
    return df.cache()


@pytest.mark.parametrize(
    "strategy",
    [
        {
            "new_column_name": "col3",
            # both constant and column are not allowed at the same time
            "strategy": [{"constant": 30, "column": "col1"}, {"column": "col2"}],
            "operations": ["sub"],
        },
        [
            {
                "new_column_name": "col3",
                "strategy": [
                    {"constant": 30},
                    {"column": "col1", "cast": "float", "constant": "22"},
                ],
                "operations": ["sub"],
            }
        ],
        {
            "new_column_name": "col3",
            "strategy": [{"constant": 30}, {"column": "col1"}],
            "operations": ["sub", "sub"],
        },
        [
            {
                "new_column_name": "col3",
                "strategy": [{"constant": 30, "cast": "integer"}, {"column": "col1"}],
                "operations": ["not_found"],
            }
        ],
    ],
)
def test_math_operator_wrong_strategy(df_input, strategy):
    """Test MathOperator with wrong strategy."""
    t = MathOperator(strategy=strategy)
    with pytest.raises(ValueError):
        t.transform(df_input)


def test_math_operator_wrong_strategy_type():
    """Test MathOperator with wrong strategy type."""
    with pytest.raises(TypeError):
        MathOperator(strategy=10)


@pytest.mark.parametrize("col_or_const", ["column", "constant"])
@pytest.mark.parametrize("cast", [None, "long"])
@pytest.mark.parametrize("operation", ["add", "sub", "mul", "div", "pow"])
def test_math_operator_single_operation(
    df_input, operation: str, col_or_const: str, cast
):
    """Test MathOperator using two columns."""
    second = {"column": "col2"} if col_or_const == "column" else {"constant": _CONST}
    strategy = {
        "new_column_name": "result",
        "strategy": [{"column": "col1"}, second],
        "operations": [operation],
    }
    if cast:
        strategy["cast"] = cast
    t = MathOperator(strategy=strategy)

    df_chk = t.transform(df_input).select("col1", "col2", "result")
    col_exp = F.col(f"{operation}_{col_or_const}").alias("result")
    df_exp = df_input.select("col1", "col2", col_exp)
    if cast:
        df_exp = df_exp.withColumn("result", F.col("result").cast(cast))
    assert_df_equality(df_chk, df_exp, ignore_row_order=True)


@pytest.mark.parametrize("operations", [["add", "div"], ["mul", "div"], ["pow", "sub"]])
def test_math_operator_double_operation(df_input, operations: str):
    """Test MathOperator using three columns."""
    strategy = {
        "new_column_name": "chk",
        "strategy": [
            {"column": "col1", "cast": "float"},
            {"constant": _CONST},
            {"column": "col2"},
        ],
        "operations": operations,
    }
    t = MathOperator(strategy=strategy)
    df_chk = t.transform(df_input).select("col1", "col2", "chk")

    operators_map: dict = {
        "add": py_operator.add,
        "sub": py_operator.sub,
        "mul": py_operator.mul,
        "div": py_operator.truediv,
        "pow": py_operator.pow,
    }

    op_0 = operators_map[operations[0]]
    op_1 = operators_map[operations[1]]
    exp_op_0 = op_0(F.col("col1").cast("float"), F.lit(_CONST))
    exp_op_1 = op_1(exp_op_0, F.col("col2"))

    df_exp = df_input.select("col1", "col2").withColumn("chk", exp_op_1)
    assert_df_equality(df_chk, df_exp, ignore_row_order=True)
