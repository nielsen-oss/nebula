"""Unit-test for GroupByCountRows."""

import pytest
from chispa import assert_df_equality
from pyspark.sql.types import StringType, StructField, StructType

from nlsn.nebula.spark_transformers.logs import GroupByCountRows
from nlsn.nebula.storage import nebula_storage as ns


@pytest.fixture(scope="module", name="df_input")
def _get_input_df(spark):
    fields = [
        StructField("c1", StringType(), True),
        StructField("c2", StringType(), True),
        StructField("c3", StringType(), True),
    ]
    data = [
        ("a", "a", "x"),
        ("b", "b", "x"),
        ("b", "b", "x"),
    ]
    return spark.createDataFrame(data, schema=StructType(fields)).persist()


def test_group_by_count_rows(df_input):
    """Test GroupByCountRows transformer."""
    columns = ["c1", "c2"]
    store_key = "group_by_count_rows"
    ns.clear()

    t = GroupByCountRows(columns=columns, store_key=store_key)
    df_chk = t.transform(df_input)

    # Assert the input dataframe is not modified
    assert_df_equality(df_chk, df_input, ignore_row_order=True)

    chk = ns.get(store_key)

    # Create the expected output with pandas
    df_group = df_input.groupBy(columns).count().toPandas()
    exp = df_group.set_index(columns)["count"].to_dict()
    exp = {k if isinstance(k, tuple) else tuple(k): v for k, v in exp.items()}

    assert chk == exp
