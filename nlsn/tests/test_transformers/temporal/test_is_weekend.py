"""Unit-test for IsWeekend."""

import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from nlsn.nebula.spark_transformers import IsWeekend

_working_days = {"tuesday", "monday", "wednesday", "thursday", "friday"}


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark: SparkSession):
    fields = [StructField("ts", StringType(), True)]

    data = [
        [None],
        [None],
        ["2021-05-05"],
        ["2021-05-06"],
        ["2021-05-07"],
        ["2021-05-08"],  # Saturday
        ["2021-05-09"],  # Sunday
        ["2021-05-10"],
        ["2021-05-11"],
        ["2021-05-12"],
        ["2021-05-13"],
        ["2021-05-14"],
    ]

    ret = (
        spark.createDataFrame(data, schema=StructType(fields))
        .withColumn("ts", F.to_timestamp("ts", format="yyyy-MM-dd"))
        # define 3 "day of week" ("dow") columns with of different data types.
        .withColumn("dow_int", F.dayofweek("ts").cast("int"))
        .withColumn("dow_long", F.dayofweek("ts").cast("long"))
        .withColumn("dow_float", F.dayofweek("ts").cast("float"))
        .cache()
    )
    return ret


def _test_weekend(df_spark):
    df = df_spark.toPandas()
    for _, row in df.iterrows():
        ts = row["ts"]
        if pd.isnull(ts):
            assert pd.isnull(row["is_working_day"])
            assert pd.isnull(row["is_weekend"])
            continue

        day_name = ts.day_name().lower()

        if day_name in _working_days:
            assert row["is_working_day"]
            assert not row["is_weekend"]

        elif day_name in {"saturday", "sunday"}:
            assert not row["is_working_day"]
            assert row["is_weekend"]

        else:
            raise ValueError(f"Unknown weekday: {day_name}")


def _test_split_saturday_sunday(df_spark):
    df = df_spark.toPandas()
    for _, row in df.iterrows():
        ts = row["ts"]
        if pd.isnull(ts):
            assert pd.isnull(row["is_working_day"])
            assert pd.isnull(row["is_saturday"])
            assert pd.isnull(row["is_sunday"])
            continue

        day_name = ts.day_name().lower()

        if day_name == "sunday":
            assert not row["is_working_day"]
            assert not row["is_saturday"]
            assert row["is_sunday"]

        elif day_name in _working_days:
            assert row["is_working_day"]
            assert not row["is_saturday"]
            assert not row["is_sunday"]

        elif day_name == "saturday":
            assert not row["is_working_day"]
            assert row["is_saturday"]
            assert not row["is_sunday"]

        else:
            raise ValueError(f"Unknown weekday: {day_name}")


@pytest.mark.parametrize(
    "input_col, dow, split",
    [
        ("ts", None, True),
        ("ts", None, False),
        (None, "dow_int", True),
        (None, "dow_long", False),
    ],
)
def test_is_weekend(df_input, input_col, dow, split: bool):
    """Unit-test for IsWeekend transformer."""
    t = IsWeekend(input_col=input_col, dayofweek=dow, split_saturday_sunday=split)
    df_chk = t.transform(df_input)

    if split:
        _test_split_saturday_sunday(df_chk)
    else:
        _test_weekend(df_chk)
