"""Unit-test for ShowDistinct."""

import pandas as pd
import pytest
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from nlsn.nebula.spark_transformers import ShowDistinct


@pytest.fixture(scope="module", name="df_input")
def _get_input_df(spark):
    fields = [
        StructField("col1", IntegerType(), True),
        StructField("col2", StringType(), True),
        StructField("col3", StringType(), True),
    ]

    data = [
        (5, "e", "ee"),
        (1, "a", "aa"),
        (2, "b", "bb"),
        (4, "d", "dd"),
        (1, "a", "aa"),
        (2, "b", "bb"),
        (4, "d", "dd"),
        (3, "c", "cc"),
        (4, "d", "dd"),
        (4, "d", "dd"),
        (2, "b", "bb"),
        (1, "a", "aa"),
    ]

    return spark.createDataFrame(data, schema=StructType(fields)).persist()


@pytest.mark.parametrize(
    "columns, ascending",
    [
        (["a", "b"], [True]),
        (["a", "b"], [True, True, True]),
        (None, [True]),
        (None, [True, False]),
    ],
)
def test_show_distinct_error(columns, ascending):
    """Test ShowDistinct transformer with input errors."""
    with pytest.raises(AssertionError):
        ShowDistinct(columns=columns, ascending=ascending)


def _get_expected(df, columns, ascending):
    select = columns if columns else df.columns
    df_exp = df.select(select).distinct().sort(select, ascending=ascending)
    return df_exp.toPandas()


@pytest.mark.parametrize(
    "columns, ascending",
    [
        ("col1", True),
        (["col1"], False),
        (["col2", "col3"], True),
        (["col2", "col3"], [False, True]),
        (None, True),
    ],
)
def test_show_distinct(df_input, columns, ascending):
    """Test ShowDistinct transformer."""
    df_exp_pd: pd.DataFrame = _get_expected(df_input, columns, ascending)

    t = ShowDistinct(columns=columns, ascending=ascending)
    df_chk_pd = t._sort_distinct(df_input).toPandas()

    pd.testing.assert_frame_equal(df_exp_pd, df_chk_pd)

    # Call just to test if it raises any error
    t.transform(df_input)
