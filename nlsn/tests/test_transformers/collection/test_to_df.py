"""Unit-test for ToDF."""

import pytest
from chispa import assert_df_equality
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, StructField, StructType

from nlsn.nebula.spark_transformers import ToDF


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    schema = StructType(
        [
            StructField("a", FloatType(), True),
            StructField("b", FloatType(), True),
            StructField("c", FloatType(), True),
        ]
    )

    nested_data = [
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [None, None, None],
        [None, 7.0, 6.0],
        [None, None, None],
        [None, 0.0, 7.0],
        [8.0, None, 9.0],
    ]

    return spark.createDataFrame(nested_data, schema=schema).persist()


@pytest.mark.parametrize("columns", [None, ["c1", "c2", "c3"]])
def test_to_df(df_input, columns):
    """Test ToDF transformer."""
    t = ToDF(columns=columns)
    df_chk = t.transform(df_input)

    if columns is None:
        df_exp = df_input
    else:
        exp_cols = [F.col(i).alias(j) for i, j in zip(df_input.columns, columns)]
        df_exp = df_input.select(exp_cols)

    assert_df_equality(df_chk, df_exp, ignore_row_order=True, ignore_nullable=True)
