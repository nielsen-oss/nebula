"""Unit-test for Pivot."""

import pytest
from chispa import assert_df_equality
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from nlsn.nebula.spark_transformers import Pivot


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark: SparkSession) -> DataFrame:
    data = [
        ["dotNET", 2012, 10000, 3],
        ["Java", 2012, 20000, 5],
        ["dotNET", 2012, 500, 6],
        ["dotNET", 2013, 16000, 8],
        ["Java", 2013, 11000, 10],
    ]
    schema_str = "course: string, year: int, earnings: int, students: int"
    return spark.createDataFrame(data, schema=schema_str).cache()


def _get_dfs(df_input, aggregations, distinct_values):
    t = Pivot(
        pivot_col="course",
        distinct_values=distinct_values,
        aggregations=aggregations,
        groupby_columns="year",
    )
    df_chk = t.transform(df_input)

    df_grouped = df_input.groupBy("year")
    args = ["course"]
    if distinct_values is not None:
        args.append(distinct_values)
    df_pivoted = df_grouped.pivot(*args)
    return df_chk, df_pivoted


@pytest.mark.parametrize("distinct_values", (None, ["dotNET", "Java"]))
def test_pivot_multiple_aggregations(df_input, distinct_values):
    """Test Pivot transformer with multiple aggregations."""
    aggregations = [
        {"agg": "sum", "col": "earnings", "alias": "earnings"},
        {"agg": "count", "col": "students", "alias": "students"},
    ]

    df_chk, df_pivoted = _get_dfs(df_input, aggregations, distinct_values)
    df_exp = df_pivoted.agg(
        F.sum("earnings").alias("earnings"), F.count("students").alias("students")
    )

    assert_df_equality(df_chk, df_exp, ignore_row_order=True, ignore_column_order=True)


@pytest.mark.parametrize("distinct_values", (None, ["dotNET", "Java"]))
def test_pivot_single_aggregation(df_input, distinct_values):
    """Test Pivot transformer with a single aggregation as <dict>."""
    aggregations = {"agg": "sum", "col": "earnings"}
    df_chk, df_pivoted = _get_dfs(df_input, aggregations, distinct_values)
    df_exp = df_pivoted.agg(F.sum("earnings"))
    assert_df_equality(df_chk, df_exp, ignore_row_order=True, ignore_column_order=True)
