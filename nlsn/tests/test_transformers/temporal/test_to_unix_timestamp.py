"""Unit-test for ToUnixTimestamp."""

from typing import List

import pandas as pd
import pyspark.sql.functions as F
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType

from nlsn.nebula.spark_transformers import ToUnixTimestamp

_fmt: str = "yyyy-MM-dd HH:mm:ss"


def _get_timezone(spark) -> str:
    # IANA format, 'Europe/Rome'
    return spark.conf.get("spark.sql.session.timeZone")


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark: SparkSession):
    fields = [StructField("col_str", StringType(), nullable=True)]

    data = [
        ["2020-09-01 17:27:03"],
        ["2020-09-01 23:30:38"],
        [None],
        [""],
        ["2020-09-01 16:45:02"],
        ["2020-09-01 11:13:15"],
    ]

    ret = spark.createDataFrame(data=data, schema=StructType(fields))
    return ret.withColumn("col_dt", F.to_timestamp("col_str", _fmt)).cache()


def test_to_unix_timestamp_error(df_input):
    """Unit-test with a missing parameter."""
    t = ToUnixTimestamp(
        input_col="col_str",
        output_col="col_str_to_unix",
    )
    with pytest.raises(AssertionError):
        t.transform(df_input)


def _to_unix_ts(s: pd.Series, tz: str) -> pd.Series:
    """Convert a pd.Series <Timestamp> to a list of unix timestamp float.

    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#from-timestamps-to-epoch

    Apply a timezone conversion to mirror the spark behavior
    """
    s = s.dt.tz_localize(tz)
    s = s.dt.tz_convert(None)
    delta = s - pd.Timestamp("1970-01-01")
    return delta // pd.Timedelta("1s")


def _iterable_from_float_to_int(o) -> List[int]:
    """Convert a list of <float> and NaN to a list of <int> and None."""
    return [None if pd.isna(i) else int(i) for i in o]


@pytest.mark.parametrize("input_col, dt_format", [("col_str", _fmt), ("col_dt", None)])
def test_to_unix_timestamp(spark, df_input, input_col: str, dt_format):
    """Test ToUnixTimestamp transformer."""
    tz: str = _get_timezone(spark)  # Get the TZ seen by spark
    t = ToUnixTimestamp(
        input_col=input_col,
        dt_format=dt_format,
        output_col="unix_ts",
    )
    df_chk = t.transform(df_input)

    # After the Pandas conversion, the "unix_ts" columns is float64 array
    # containing NaT. Later must be converted to a list of integers and None.
    df_pd = df_chk.select(input_col, "unix_ts").toPandas()

    # Ensure the input columns is a timestamp series.
    s_dt: pd.Series = pd.to_datetime(df_pd[input_col])

    # Convert it to a list of unix timestamp float.
    s_ts_float = _to_unix_ts(s_dt, tz)

    # Convert it to a list of unix timestamp integers.
    li_exp = _iterable_from_float_to_int(s_ts_float)

    # Convert the transformer output in a list of integers/None
    li_chk = _iterable_from_float_to_int(df_pd["unix_ts"].tolist())

    assert li_chk == li_exp
