"""Unit-test for MultipleLiterals."""

import pytest
from chispa import assert_df_equality
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType

from nlsn.nebula.spark_transformers import MultipleLiterals


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    data = [["a1", "a2"], ["b1", "b2"]]
    return spark.createDataFrame(data, schema="c1: string, c2: string").persist()


@pytest.mark.parametrize(
    "values, err",
    [
        ({1: {"value": "finance"}}, TypeError),
        ({"dep": ["wrong type"]}, AssertionError),
        ({"dep": {"value": "finance", "wrong": "x"}}, AssertionError),
        ({"dep": {"value": "finance", "cast": 1}}, AssertionError),
    ],
)
def test_multiple_literals_error(values: dict, err):
    """Test MultipleLiterals transformer with wrong inputs."""
    with pytest.raises(err):
        MultipleLiterals(values=values)


def test_with_column(df_input):
    """Test MultipleLiterals transformer."""
    values = {
        "department": {"value": "finance"},
        "employees": {"value": 10, "cast": "bigint"},
        "c2": {"value": 10, "cast": "float"},
        "active": {"value": True, "cast": BooleanType()},
        "date": {"value": "2024-01-01", "cast": "string"},
    }
    t = MultipleLiterals(values=values)
    df_chk = t.transform(df_input)

    df_exp = df_input
    for name, nd in values.items():
        col = F.lit(nd["value"])
        if nd.get("cast"):
            col = col.cast(nd["cast"])
        df_exp = df_exp.withColumn(name, col)

    assert_df_equality(
        df_chk,
        df_exp,
        ignore_column_order=True,
        ignore_row_order=True,
        ignore_nullable=True,
    )
