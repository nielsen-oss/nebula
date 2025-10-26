"""Unit-test for LogicalOperatorOverWindow."""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import BooleanType, StringType, StructField, StructType
from pytest import fixture

from nlsn.nebula.spark_transformers.aggregations import LogicalOperatorOverWindow


@fixture(scope="module", name="df_input")
def _get_df_input(spark: SparkSession):
    fields = [
        StructField("c1", StringType(), True),
        StructField("c2", BooleanType(), True),
    ]

    data = [
        ("a", True),
        ("a", False),
        ("a", None),
        ("b", True),
        ("b", True),
        ("c", False),
        ("c", False),
        ("c", False),
        ("d", True),
        ("d", True),
        ("e", True),
        ("e", False),
        ("e", None),
        (None, True),
        (None, False),
    ]
    return spark.createDataFrame(data, StructType(fields)).cache()


@pytest.mark.parametrize("op", ["and", "OR"])
def test_logical_operator_over_window(df_input, op):
    """Test LogicalOperatorOverWindow transformer."""
    t = LogicalOperatorOverWindow(
        partition_cols="c1", input_col="c2", operator=op, output_col="out"
    )
    df_chk = t.transform(df_input)
    df_pd = df_chk.toPandas()

    for _, df_grouped in df_pd.groupby("c1", dropna=False):
        ar = df_grouped["c2"].tolist()
        n = len(ar)
        input_values = [False if i is None else i for i in ar]
        li_chk = df_grouped["out"].tolist()

        if op.strip().lower() == "and":
            base_value = False not in input_values
        else:
            base_value = True in input_values

        li_exp = [base_value] * n
        assert li_exp == li_chk
