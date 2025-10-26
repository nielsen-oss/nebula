"""Unit-test for Coalesce."""

import pytest
from chispa import assert_df_equality
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, StringType, StructField, StructType

from nlsn.nebula.spark_transformers import Coalesce


@pytest.fixture(scope="module", name="df_input_numeric")
def _get_df_input(spark):
    fields = [
        StructField("a", FloatType(), True),
        StructField("b", FloatType(), True),
        StructField("c", FloatType(), True),
    ]

    data = [
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [None, None, None],
        [None, float("nan"), 6.0],
        [float("nan"), float("nan"), float("nan")],
        [None, 0.0, 7.0],
        [8.0, None, 9.0],
    ]

    return spark.createDataFrame(data, schema=StructType(fields)).persist()


def test_coalesce_numeric_nan_as_null(df_input_numeric):
    """Test Coalesce transformer for numerical values with 'treat_nan_as_null=True'."""
    t = Coalesce(columns=["a", "b", "c"], output_col="d", treat_nan_as_null=True)
    df_transf = t.transform(df_input_numeric)

    df_pd = df_transf.toPandas()
    s_coalesce = (
        df_pd[["a", "b", "c"]].bfill(axis=1).iloc[:, 0]
    )  # the "coalesce" column is the first one
    assert s_coalesce.equals(df_pd["d"])


@pytest.mark.parametrize("drop", [True, False])
def test_coalesce_numeric(df_input_numeric, drop: bool):
    """Test Coalesce transformer for numerical values with 'treat_nan_as_null=True'."""
    t = Coalesce(columns=["a", "b", "c"], output_col="c", drop_input_cols=drop)
    df_chk = t.transform(df_input_numeric)
    input_cols = df_input_numeric.columns
    df_exp = df_input_numeric.withColumn("c", F.coalesce(*input_cols))
    if drop:
        df_exp = df_exp.drop(*[i for i in input_cols if i != "c"])
    assert_df_equality(df_chk, df_exp, ignore_row_order=True, allow_nan_equality=True)


def test_coalesce_str(spark):
    """Test Coalesce transformer for string values with 'treat_blank_string_as_null=True'."""
    schema = StructType(
        [
            StructField("a", StringType(), True),
            StructField("b", StringType(), True),
            StructField("c", StringType(), True),
        ]
    )

    data = [
        ["a1", "b1", ""],
        ["  ", None, "c"],
        ["", "", ""],
        ["  ", "  ", "   "],
        [None, None, None],
        ["a   ", None, "b"],
    ]

    df_input_test = spark.createDataFrame(data, schema=schema)

    t = Coalesce(
        columns=["a", "b", "c"], output_col="d", treat_blank_string_as_null=True
    )
    df_transf = t.transform(df_input_test)

    df_pd = df_transf.toPandas()
    for c in df_pd.columns:
        df_pd[c] = df_pd[c].str.strip().replace({"": None})

    s_coalesce = (
        df_pd[["a", "b", "c"]].bfill(axis=1).iloc[:, 0]
    )  # the "coalesce" column is the first one
    assert s_coalesce.equals(df_pd["d"])
