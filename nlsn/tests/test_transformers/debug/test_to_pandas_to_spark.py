"""Unit-Test for 'ToPandasToSpark' function."""

import pytest
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    MapType,
    StringType,
    StructField,
    StructType,
)

from nlsn.nebula.helpers import assert_pandas_df_equal
from nlsn.nebula.spark_transformers import ToPandasToSpark


@pytest.fixture(scope="module", name="df_input")
def _get_input_df(spark):
    # value | column name | datatype
    input_data = [
        (3, "col_3", IntegerType()),
        ("string A", "col_1", StringType()),
        (1.5, "col_2", FloatType()),
        ([1, 2, 3], "col_4", ArrayType(IntegerType())),
        ({"a": 1}, "col_5", MapType(StringType(), IntegerType())),
        ([[1, 2], [2, 3]], "col_6", ArrayType(ArrayType(IntegerType()))),
        (
            {"a": {"b": 2}},
            "col_7",
            MapType(StringType(), MapType(StringType(), IntegerType())),
        ),
    ]

    data = [[i[0] for i in input_data]]
    # Add a row with null values to see if it is able to handle it.
    # The first column is an integer and cannot accept None in pandas,
    # just put a real integer
    null_row = [1]
    null_row += [None for _ in input_data][:-1]
    data += [null_row]
    fields = [StructField(*i[1:]) for i in input_data]
    schema = StructType(fields)
    return spark.createDataFrame(data, schema=schema)


def test_to_pandas_to_spark_transformer(df_input):
    """Test ToPandasToSpark transformer."""
    t = ToPandasToSpark()
    df_chk = t.transform(df_input)

    # Chispa cannot be used in this case because the DataFrames contain arrays
    # and maps, which are not sortable or hashable.
    # To address this, convert the DataFrames to Pandas and use
    # 'assert_pandas_df_equal' which is capable of handling complex data types.
    df_chk_pd = df_chk.toPandas()
    df_exp_pd = df_input.toPandas()

    assert_pandas_df_equal(df_chk_pd, df_exp_pd, assert_not_deep=False)
