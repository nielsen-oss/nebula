"""Unit-test for FromUtcTimestamp and ToUtcTimestamp."""

import pandas as pd
import pyspark.sql.functions as F
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType

from nlsn.nebula.spark_transformers import FromUtcTimestamp, ToUtcTimestamp


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark: SparkSession):
    fields = [
        StructField("time_str", StringType(), True),
        StructField("tz", StringType(), True),
    ]

    fmt = "yyyy-MM-dd HH:mm:ss"
    data = [
        ["2021-01-15 17:23:11", "America/New_York"],
        ["2021-01-15 17:23:11", "Europe/Rome"],
    ]

    df = spark.createDataFrame(data, schema=StructType(fields))
    return df.withColumn("time_dt", F.to_timestamp("time_str", format=fmt)).cache()


def _exp_from_utc(dt_in: pd.Timestamp, tz: str):
    return dt_in.tz_localize("UTC").tz_convert(tz).tz_localize(None)


def _exp_to_utc(dt_in: pd.Timestamp, tz: str):
    return dt_in.tz_localize(tz).tz_convert("UTC").tz_localize(None)


_PARAMS = [(FromUtcTimestamp, _exp_from_utc), (ToUtcTimestamp, _exp_to_utc)]


@pytest.mark.parametrize("cls, func", _PARAMS)
def test_from_utc_timestamp_literal(df_input, cls, func):
    """Test FromUtcTimestamp / ToUtcTimestamp using a literal timezone."""
    tz = "America/New_York"

    t = cls(input_col="time_dt", output_col="result", timezone=tz)
    df_out = t.transform(df_input)

    t_in, t_chk = df_out.select("time_dt", "result").limit(1).collect()[0][:]

    dt_in = pd.Timestamp(t_in)
    dt_chk = pd.Timestamp(t_chk)
    dt_exp = func(dt_in, tz)
    assert dt_chk == dt_exp


@pytest.mark.parametrize("cls, func", _PARAMS)
def test_from_utc_timestamp_column(df_input, cls, func):
    """Test FromUtcTimestamp / ToUtcTimestamp using a timezone column."""
    t = cls(input_col="time_dt", output_col="result", timezone_col="tz")
    df_out = t.transform(df_input).select("time_dt", "tz", "result").toPandas()

    for _, row in df_out.iterrows():
        dt_in = pd.Timestamp(row.time_dt)
        tz = row.tz
        dt_chk = pd.Timestamp(row.result)
        dt_exp = func(dt_in, tz)
        assert dt_chk == dt_exp
