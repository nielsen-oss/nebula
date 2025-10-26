"""Unit-test for Melt."""

import pytest
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql.types import StructType, StructField, StringType, LongType

from nlsn.nebula.spark_transformers import Melt


@pytest.fixture(scope="module")
def sample_df(spark):
    data = [
        ("A", 1, 2, 3, 4, 5),
        ("B", 6, 7, 8, 9, 10),
        ("C", 11, 12, 13, 14, 15),
        ("D", 16, 17, 18, 19, 20),
    ]
    schema = StructType(
        [
            StructField("id", StringType(), True),
            StructField("col1", LongType(), True),
            StructField("col2", LongType(), True),
            StructField("col3", LongType(), True),
            StructField("col4", LongType(), True),
            StructField("col5", LongType(), True),
        ]
    )
    return spark.createDataFrame(data, schema)


@pytest.mark.parametrize(
    "id_cols,melt_cols,variable_col,value_col,expected_data",
    [
        (
                ["id"],
                ["col1", "col2", "col3", "col4", "col5"],
                "variable",
                "value",
                [
                    ("A", "col1", 1),
                    ("A", "col2", 2),
                    ("A", "col3", 3),
                    ("A", "col4", 4),
                    ("A", "col5", 5),
                    ("B", "col1", 6),
                    ("B", "col2", 7),
                    ("B", "col3", 8),
                    ("B", "col4", 9),
                    ("B", "col5", 10),
                    ("C", "col1", 11),
                    ("C", "col2", 12),
                    ("C", "col3", 13),
                    ("C", "col4", 14),
                    ("C", "col5", 15),
                    ("D", "col1", 16),
                    ("D", "col2", 17),
                    ("D", "col3", 18),
                    ("D", "col4", 19),
                    ("D", "col5", 20),
                ],
        ),
        (
                ["id"],
                ["col1", "col2", "col3"],
                "variable",
                "value",
                [
                    ("A", "col1", 1),
                    ("A", "col2", 2),
                    ("A", "col3", 3),
                    ("B", "col1", 6),
                    ("B", "col2", 7),
                    ("B", "col3", 8),
                    ("C", "col1", 11),
                    ("C", "col2", 12),
                    ("C", "col3", 13),
                    ("D", "col1", 16),
                    ("D", "col2", 17),
                    ("D", "col3", 18),
                ],
        ),
    ],
)
def test_melt_with_id_cols_and_melt_cols(
        spark, sample_df, id_cols, melt_cols, variable_col, value_col, expected_data
):
    melt = Melt(
        id_cols=id_cols,
        melt_cols=melt_cols,
        variable_col=variable_col,
        value_col=value_col,
    )
    result_df = melt.transform(sample_df)

    expected_df = spark.createDataFrame(
        expected_data, [id_cols[0], variable_col, value_col]
    )
    assert_df_equality(result_df, expected_df, ignore_nullable=True)


def test_melt_with_no_id_cols(spark, sample_df):
    melt = Melt(
        melt_cols=["col1", "col2", "col3"], variable_col="variable", value_col="value"
    )
    result_df = melt.transform(sample_df)

    expected_data = [
        ("col1", 1),
        ("col2", 2),
        ("col3", 3),
        ("col1", 6),
        ("col2", 7),
        ("col3", 8),
        ("col1", 11),
        ("col2", 12),
        ("col3", 13),
        ("col1", 16),
        ("col2", 17),
        ("col3", 18),
    ]
    expected_df = spark.createDataFrame(expected_data, ["variable", "value"])
    assert_df_equality(result_df, expected_df, ignore_nullable=True)


def test_melt_with_single_melt_col(spark, sample_df):
    melt = Melt(
        id_cols=["id"], melt_cols=["col1"], variable_col="variable", value_col="value"
    )
    result_df = melt.transform(sample_df)

    expected_data = [
        ("A", "col1", 1),
        ("B", "col1", 6),
        ("C", "col1", 11),
        ("D", "col1", 16),
    ]
    expected_df = spark.createDataFrame(expected_data, ["id", "variable", "value"])
    assert_df_equality(result_df, expected_df, ignore_nullable=True)
