"""Unit-test for Timedelta."""

from datetime import datetime, timedelta

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from nlsn.nebula.spark_transformers import Timedelta

_ARG_NAMES: str = "days, hours, minutes, seconds, output_fmt"
_ARG_VALUES: list = [
    (10, 0, 0, 0, "%Y-%m-%d"),
    (0, 0, 0, 0, "%Y-%m-%d"),
    (-180, 0, 0, 0, None),
    (-1, 1, -2, 3, "%Y-%m-%d %H:%M:%S"),
    (0, -3, -21, 33, None),
]


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark: SparkSession):
    fields = [StructField("col_str", StringType(), nullable=True)]
    data = [["a"], ["b"]]
    return spark.createDataFrame(data=data, schema=StructType(fields)).cache()


def test_timedelta_wrong_input_type(df_input):
    """Unit-test for Timedelta transformer with wrong input type."""
    t = Timedelta(output_col="x", input_col="c_int")
    df = df_input.withColumn("c_int", F.lit(10))
    with pytest.raises(TypeError):
        t.transform(df)


def test_timedelta_without_format(df_input):
    """Unit-test for Timedelta transformer with string columns and without format."""
    t = Timedelta(output_col="x", input_col="col_str")
    with pytest.raises(AssertionError):
        t.transform(df_input)


@pytest.mark.parametrize("as_literal", [True, False])
@pytest.mark.parametrize("cast_timestamp", [True, False])
@pytest.mark.parametrize(_ARG_NAMES, _ARG_VALUES)
def test_timedelta(
    df_input,
    days: int,
    hours: int,
    minutes: int,
    seconds: int,
    output_fmt: str,
    as_literal: bool,
    cast_timestamp: bool,
):
    """Unit-test for Timedelta transformer."""
    date_str: str = "2024-01-01"
    input_fmt = "%Y-%m-%d"
    date_dt: datetime = datetime.strptime(date_str, input_fmt)

    if as_literal:
        input_col_name = None
        if cast_timestamp:
            input_date = date_dt
        else:
            input_date = date_str
    else:
        input_date = None
        input_col_name = "input_col"
        if cast_timestamp:
            lit_value = F.lit(date_dt)
        else:
            lit_value = F.lit(date_str)
        df_input = df_input.withColumn(input_col_name, lit_value)

    if cast_timestamp:
        input_fmt = None

    t = Timedelta(
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds,
        date=input_date,
        input_col=input_col_name,
        input_dt_format=input_fmt,
        output_col="new_col",
        output_dt_format=output_fmt,
    )

    df_chk = t.transform(df_input)
    chk = df_chk.select("new_col").distinct().collect()[0][0]

    td: timedelta = timedelta(
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds,
    )

    exp = date_dt + td

    if output_fmt:
        exp = exp.strftime(output_fmt)

    assert chk == exp
