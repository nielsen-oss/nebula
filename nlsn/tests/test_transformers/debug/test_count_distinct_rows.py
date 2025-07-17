"""Unit-test for CountDistinctRows."""

import pytest
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from nlsn.nebula.spark_transformers.logs import CountDistinctRows
from nlsn.nebula.storage import nebula_storage as ns


@pytest.fixture(scope="module", name="df_input")
def _get_input_df(spark):
    fields = [
        StructField("col1", IntegerType(), True),
        StructField("col2", StringType(), True),
    ]

    data = [
        (1, "a"),
        (1, "a"),
        (1, "b"),
        (2, "c"),
        (2, "c"),
        (2, "d"),
    ]

    return spark.createDataFrame(data, schema=StructType(fields)).persist()


def _get_expected(df, columns) -> int:
    select = columns if columns else df.columns
    return df.select(select).distinct().count()


@pytest.mark.parametrize("columns", [None, ["col1", "col2"]])
def test_count_distinct_rows(df_input, columns):
    """Test CountDistinctRows transformer."""
    store_key = "count_distinct_rows"
    ns.clear()
    n_exp: int = _get_expected(df_input, columns)

    t = CountDistinctRows(columns=columns, store_key=store_key)
    df_chk = t.transform(df_input)

    # Assert the input dataframe is not modified
    assert df_chk is df_input

    n_chk = ns.get(store_key)
    assert n_chk == n_exp
