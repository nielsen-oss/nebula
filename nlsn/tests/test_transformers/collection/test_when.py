"""Unit-test for When."""

import pytest
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, StringType, StructField, StructType

from nlsn.nebula.spark_transformers import When


def _assert_results(spark, df_input, df_chk, exp_data):
    fields = df_input.schema.fields[:]
    fields += [StructField("result", StringType(), nullable=True)]
    df_exp = spark.createDataFrame(exp_data, StructType(fields))
    assert_df_equality(df_chk, df_exp)


@pytest.fixture(scope="module", name="df_input_1")
def _get_df_input_1(spark):
    data = [(1, 3), (2, 5), (3, 7), (4, 9), (5, 11)]
    columns = ["c1", "c2"]
    return spark.createDataFrame(data, columns).persist()


@pytest.mark.parametrize(
    "conditions, otherwise_const, otherwise_col, exp_data",
    [
        (  # Test 1
            [
                {
                    "input_col": "c1",
                    "operator": "eq",
                    "value": 2,
                    "output_constant": "two",
                },
                {
                    "input_col": "c1",
                    "operator": "gt",
                    "value": 3,
                    "output_constant": "greater",
                },
            ],
            None,
            None,
            [
                (1, 3, None),
                (2, 5, "two"),
                (3, 7, None),
                (4, 9, "greater"),
                (5, 11, "greater"),
            ],
        ),
        (  # Test 2
            [
                {
                    "input_col": "c1",
                    "operator": "eq",
                    "value": 2,
                    "output_constant": "two",
                },
                {
                    "input_col": "c1",
                    "operator": "gt",
                    "value": 3,
                    "output_constant": "greater",
                    "output_column": "c2",
                },
                {
                    "input_col": "c2",
                    "operator": "between",
                    "value": (5, 9),
                    "output_constant": "between_5_and_9",
                },
            ],
            "superseded",  # will be superseded by "c1" column
            "c1",
            [
                (1, 3, "1"),
                (2, 5, "two"),
                (3, 7, "between_5_and_9"),
                (4, 9, "9"),
                (5, 11, "11"),
            ],
        ),
    ],
)
def test_when_1(
    spark, df_input_1, conditions, otherwise_const, otherwise_col, exp_data
):
    """Test When transformer using df_input_1."""
    transformer = When(
        output_column="result",
        conditions=conditions,
        otherwise_constant=otherwise_const,
        otherwise_column=otherwise_col,
        cast_output="string",
    )
    df_chk = transformer.transform(df_input_1)
    _assert_results(spark, df_input_1, df_chk, exp_data)


@pytest.fixture(scope="module", name="df_input_2")
def _get_df_input_2(spark):
    data = [
        (1, 3, None, "abc", "def"),
        (2, None, 5, "def", "ghi"),
        (3, 7, 9, "ghi", "jkl"),
        (4, 9, 11, "jkl", "mno"),
        (None, 11, 13, None, None),
        (6, 13, None, "mno", "pqr"),
        (7, None, 15, "pqr", "stu"),
        (8, 17, 17, "stu", "vwx"),
        (9, 19, 19, "vwx", "yz"),
        (10, 21, 21, "yz", None),
    ]
    schema = StructType(
        [
            StructField("col1", LongType(), nullable=True),
            StructField("col2", LongType(), nullable=True),
            StructField("col3", LongType(), nullable=True),
            StructField("col4", StringType(), nullable=True),
            StructField("col5", StringType(), nullable=True),
        ]
    )
    return spark.createDataFrame(data, schema).persist()


@pytest.mark.parametrize(
    "conditions, otherwise_const, otherwise_col, exp_data",
    [
        (  # Test 1
            [
                {
                    "input_col": "col1",
                    "operator": "eq",
                    "value": 2,
                    "output_constant": "two",
                },
                {
                    "input_col": "col2",
                    "operator": "lt",
                    "value": 9,
                    "output_constant": None,
                },
                {
                    "input_col": "col3",
                    "operator": "ge",
                    "value": 15,
                    "output_column": "col1",
                },
                {
                    "input_col": "col4",
                    "operator": "contains",
                    "value": "jkl",
                    "output_constant": "contains_jkl",
                },
                {
                    "input_col": "col5",
                    "operator": "isNull",
                    "value": None,
                    "output_constant": "is_null",
                },
            ],
            "default",
            None,
            [
                (1, 3, None, "abc", "def", None),
                (2, None, 5, "def", "ghi", "two"),
                (3, 7, 9, "ghi", "jkl", None),
                (4, 9, 11, "jkl", "mno", "contains_jkl"),
                (None, 11, 13, None, None, "is_null"),
                (6, 13, None, "mno", "pqr", "default"),
                (7, None, 15, "pqr", "stu", "7"),
                (8, 17, 17, "stu", "vwx", "8"),
                (9, 19, 19, "vwx", "yz", "9"),
                (10, 21, 21, "yz", None, "10"),
            ],
        ),
        (  # Test 2
            [
                {
                    "input_col": "col1",
                    "operator": "gt",
                    "value": 9,
                    "output_constant": "greater_than_9",
                },
                {
                    "input_col": "col2",
                    "operator": "le",
                    "value": 13,
                    "output_constant": "less_than_or_equal_13",
                },
                {
                    "input_col": "col3",
                    "operator": "isNull",
                    "value": None,
                    "output_constant": "is_null",
                },
                {
                    "input_col": "col4",
                    "operator": "contains",
                    "value": "mno",
                    "output_constant": "contains_mno",
                },
                {
                    "input_col": "col5",
                    "operator": "ne",
                    "value": "pqr",
                    "output_column": "col5",
                },
            ],
            "REALLY?",
            None,
            [
                (1, 3, None, "abc", "def", "less_than_or_equal_13"),
                (2, None, 5, "def", "ghi", "ghi"),
                (3, 7, 9, "ghi", "jkl", "less_than_or_equal_13"),
                (4, 9, 11, "jkl", "mno", "less_than_or_equal_13"),
                (None, 11, 13, None, None, "less_than_or_equal_13"),
                (6, 13, None, "mno", "pqr", "less_than_or_equal_13"),
                (7, None, 15, "pqr", "stu", "stu"),
                (8, 17, 17, "stu", "vwx", "vwx"),
                (9, 19, 19, "vwx", "yz", "yz"),
                (10, 21, 21, "yz", None, "greater_than_9"),
            ],
        ),
        (  # Test 3
            [
                {
                    "input_col": "col1",
                    "operator": "eq",
                    "value": 2,
                    "output_column": "col1",
                },
                {
                    "input_col": "col2",
                    "operator": "lt",
                    "value": 9,
                    "output_constant": None,
                },
                {
                    "input_col": "col3",
                    "operator": "ge",
                    "value": 15,
                    "output_constant": "greater_or_equal",
                },
                {
                    "input_col": "col4",
                    "operator": "contains",
                    "value": "jkl",
                    "output_constant": "contains_jkl",
                },
                {
                    "input_col": "col5",
                    "operator": "isNull",
                    "value": None,
                    "output_constant": "is_null",
                },
                {
                    "input_col": "col1",
                    "operator": "lt",
                    "value": 5,
                    "output_constant": "less_than_5",
                },
            ],
            None,
            "col2",
            [
                (1, 3, None, "abc", "def", None),
                (2, None, 5, "def", "ghi", "2"),
                (3, 7, 9, "ghi", "jkl", None),
                (4, 9, 11, "jkl", "mno", "contains_jkl"),
                (None, 11, 13, None, None, "is_null"),
                (6, 13, None, "mno", "pqr", "13"),
                (7, None, 15, "pqr", "stu", "greater_or_equal"),
                (8, 17, 17, "stu", "vwx", "greater_or_equal"),
                (9, 19, 19, "vwx", "yz", "greater_or_equal"),
                (10, 21, 21, "yz", None, "greater_or_equal"),
            ],
        ),
        (  # Test 4
            [
                {
                    "input_col": "col1",
                    "operator": "gt",
                    "value": 9,
                    "output_constant": "greater_than_9",
                },
                {
                    "input_col": "col2",
                    "operator": "le",
                    "value": 12,
                    "output_constant": "less_than_or_equal_12",
                },
                {
                    "input_col": "col3",
                    "operator": "isNull",
                    "value": None,
                    "output_constant": "is_null",
                },
                {
                    "input_col": "col4",
                    "operator": "contains",
                    "value": "mno",
                    "output_constant": "contains_mno",
                },
                {
                    "input_col": "col5",
                    "operator": "ne",
                    "value": "pqr",
                    "output_constant": None,
                },
                {
                    "input_col": "col1",
                    "operator": "ge",
                    "value": 5,
                    "output_constant": "greater_than_or_equal_5",
                },
            ],
            None,
            "col5",
            [
                (1, 3, None, "abc", "def", "less_than_or_equal_12"),
                (2, None, 5, "def", "ghi", None),
                (3, 7, 9, "ghi", "jkl", "less_than_or_equal_12"),
                (4, 9, 11, "jkl", "mno", "less_than_or_equal_12"),
                (None, 11, 13, None, None, "less_than_or_equal_12"),
                (6, 13, None, "mno", "pqr", "is_null"),
                (7, None, 15, "pqr", "stu", None),
                (8, 17, 17, "stu", "vwx", None),
                (9, 19, 19, "vwx", "yz", None),
                (10, 21, 21, "yz", None, "greater_than_9"),
            ],
        ),
    ],
)
def test_when_2(
    spark, df_input_2, conditions, otherwise_const, otherwise_col, exp_data
):
    """Test When transformer using df_input_2."""
    transformer = When(
        output_column="result",
        conditions=conditions,
        otherwise_constant=otherwise_const,
        otherwise_column=otherwise_col,
        cast_output="string",
    )
    df_chk = transformer.transform(df_input_2)
    _assert_results(spark, df_input_2, df_chk, exp_data)


@pytest.mark.parametrize("output_column", ["c1", "result"])
def test_when_mixed_types_1(df_input_1, output_column):
    """Test When transformer with mixed data types using a literal as otherwise."""
    transformer = When(
        output_column=output_column,
        conditions=[
            {
                "input_col": "c1",
                "operator": "lt",
                "value": 3,
                "output_column": "c1",
            }
        ],
        otherwise_constant="3+",
        cast_output="string",
    )
    df_chk = transformer.transform(df_input_1)
    df_exp = df_input_1.withColumn(
        output_column,
        F.when(F.col("c1") < 3, F.col("c1").cast("string")).otherwise(F.lit("3+")),
    )

    assert_df_equality(df_chk, df_exp, ignore_row_order=True, ignore_nullable=True)


@pytest.mark.parametrize("output_column", ["c1", "result"])
def test_when_mixed_types_2(df_input_1, output_column):
    """Test When transformer with mixed data types using a column as otherwise."""
    transformer = When(
        output_column=output_column,
        conditions=[
            {
                "input_col": "c1",
                "operator": "ge",
                "value": 3,
                "output_constant": "3+",
            }
        ],
        otherwise_column="c1",
        cast_output="string",
    )
    df_chk = transformer.transform(df_input_1)
    df_exp = df_input_1.withColumn(
        output_column,
        F.when(F.col("c1") >= 3, F.lit("3+")).otherwise(F.col("c1").cast("string")),
    )

    assert_df_equality(df_chk, df_exp, ignore_row_order=True, ignore_nullable=True)
