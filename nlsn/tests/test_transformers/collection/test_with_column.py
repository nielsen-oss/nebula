"""Unit-test for WithColumn."""

import pytest
from pyspark.sql.types import StringType, StructField, StructType

from nlsn.nebula.spark_transformers import WithColumn


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    """Define test data."""
    fields = [StructField("col_1", StringType())]
    return spark.createDataFrame([["a"], ["b"]], schema=StructType(fields)).cache()


@pytest.mark.parametrize(
    "value, cast",
    [
        ([[1, 2]], None),  # nested lists are not allowed
        ({1, 2}, None),  # sets are not allowed
        ({"a": [1, 2], "b": [3, 4]}, None),  # nested complex dictionary
        (None, "set<str>"),  # <set> cast is not allowed
        (None, "ARRAY<MAP<STR,INT>>"),  # nested complex cast type
    ],
)
def test_with_column_error(df_input, value, cast):
    """Invalid 'value' and 'cast'."""
    t = WithColumn(column_name="col_2", value=value, cast=cast)
    with pytest.raises(ValueError):
        t.transform(df_input)


@pytest.mark.parametrize(
    "value, cast",
    [
        (1, None),
        (None, None),
        ("1", "string"),
        ([1, 2], "array<int>"),
        ([], "array<int>"),  # empty
        ({"a": 1, "b": 2}, "map<string,int>"),
    ],
)
def test_with_column(df_input, value, cast):
    """Test WithColumn transformer."""
    column_name = "col_2"
    t = WithColumn(column_name=column_name, value=value, cast=cast)
    df_out = t.transform(df_input).select(column_name)
    if cast is not None:
        type_name = df_out.schema.fields[0].dataType.simpleString()
        type_name = type_name.lower().replace(" ", "")
        assert cast.lower().replace(" ", "") == type_name

    res = df_out.rdd.flatMap(lambda x: x).collect()
    assert [i == value for i in res]
