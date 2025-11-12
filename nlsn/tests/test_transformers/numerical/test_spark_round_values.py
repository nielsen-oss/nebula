"""Unit-test for RoundValues."""
import random
from decimal import Decimal

import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DecimalType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from nlsn.nebula.transformers.numerical import RoundValues

# Do not use FloatType, precision is too low, and the test can fail.

_params = [
    ("c_int", 0, None),
    ("c_double", 3, None),
    ("c_double", -2, None),
    # To Allow negative rounding in DecimalType set
    # spark.sql.legacy.allowNegativeScaleOfDecimal -> true
    (["c_decimal"], 1, None),
    (["c_double"], 2, "col_output"),
    (["c_double", "c_int", "c_decimal"], 1, None),
    (["c_double", "c_int"], -1, None),
]

_data = [
    ("1", 101.342),
    ("2", 3.14159),
    ("3", 0.123456782),
    ("4", -354.123456789),
    ("5", -6.0),
    ("6", None),
]


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    fields = [
        StructField("id", StringType(), nullable=True),
        StructField("c_double", DoubleType(), nullable=True),
    ]
    ret = (
        spark.createDataFrame(data=_data, schema=StructType(fields))
        .withColumn("c_int", F.col("c_double").cast(IntegerType()))
        .withColumn("c_long", F.col("c_double").cast(LongType()))
        .withColumn(
            "c_decimal", F.col("c_double").cast(DecimalType(precision=10, scale=4))
        )
    )
    return ret.persist()


def _check(input_value, chk, precision, dtype):
    if input_value is None:
        assert chk is None
        return

    if isinstance(chk, Decimal):
        # Convert to float any Decimal type
        chk = float(chk)

    # Handle input and cast to int if necessary
    if isinstance(dtype, (IntegerType, LongType)):
        cast_value = int(input_value)
    else:
        cast_value = input_value
    exp = round(cast_value, precision)

    assert chk == exp


def _test_round_values(t, df_input, input_columns, precision, output_column):
    df_out = t.transform(df_input)

    li_chk = df_out.rdd.map(lambda x: x.asDict()).collect()

    dict_input = dict(_data)
    dict_schema = {i.name: i.dataType for i in df_out.schema}

    if output_column:
        for row in li_chk:
            id_: str = row["id"]
            input_value = dict_input[id_]
            chk = row[output_column]
            assert row[input_columns[0]] == input_value
            dtype = dict_schema[output_column]
            _check(input_value, chk, precision, dtype)
        return

    input_columns = [input_columns] if isinstance(input_columns, str) else input_columns

    for row in li_chk:
        id_: str = row["id"]
        input_value = dict_input[id_]
        for col in input_columns:
            chk = row[col]
            dtype = dict_schema[col]
            _check(input_value, chk, precision, dtype)


@pytest.mark.parametrize("input_columns, precision, output_column", _params)
def test_round_values(df_input, input_columns, precision, output_column):
    """Unit-test RoundValues."""
    t = RoundValues(
        input_columns=input_columns, decimals=precision, output_column=output_column
    )
    _test_round_values(t, df_input, input_columns, precision, output_column)

