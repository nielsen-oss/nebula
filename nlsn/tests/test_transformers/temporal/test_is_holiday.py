"""Unit-test for IsHoliday."""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, StringType, StructField, StructType

from nlsn.nebula.spark_transformers import IsHoliday

_fields = [
    StructField("ts", StringType(), True),
    StructField("expected", BooleanType(), True),
]

_schema = StructType(_fields)


def _check_output(df):
    # df a pandas dataframe with the columns "ts", expected" and "is_holiday"
    for ts, chk, exp in df[["ts", "expected", "is_holiday"]].values:
        # safe way to handle None comparison None == None
        assert chk == exp, f"{ts} exp={exp}, check={chk}"


def _create_df(spark, data):
    ret = spark.createDataFrame(data, schema=_schema).withColumn(
        "ts", F.to_timestamp("ts", format="yyyy-MM-dd")
    )
    return ret


def _transform(spark, data, country, persist):
    df_input = _create_df(spark, data)

    transf = IsHoliday(
        input_col="ts", country=country, output_col="is_holiday", persist=persist
    )
    # return a pandas dataframe
    return transf.transform(df_input).toPandas()


def test_is_holiday_wrong_country():
    """Test IsHoliday with a wrong country."""
    with pytest.raises(KeyError):
        IsHoliday(input_col="ts", country="wrong")


def test_is_holiday_italy(spark):
    """Test IsHoliday when country = Italy."""
    data = [
        [None, None],
        [None, None],
        ["2023-12-25", True],  # Christmas
        ["2023-01-01", True],  # new year
        ["2023-05-10", False],
        ["2023-05-11", False],
        ["2023-01-01", True],  # new year
    ]

    df_pd = _transform(spark, data, "IT", True)
    _check_output(df_pd)


def test_is_holiday_thailand(spark):
    """Test IsHoliday when country = Thailand."""
    data = [
        [None, None],
        ["2023-01-01", True],
        ["2023-01-02", True],
        ["2023-01-05", False],
        ["2023-01-25", False],
        ["2023-03-06", True],
        ["2023-03-12", False],
        ["2023-04-06", True],
        ["2023-04-13", True],
        ["2023-04-25", False],
        ["2023-05-01", True],
        ["2023-05-04", True],
        ["2023-05-05", True],
        ["2023-05-11", True],
        ["2023-06-03", True],
        ["2023-06-05", True],
        ["2023-06-15", False],
        ["2023-06-20", False],
        ["2023-08-01", True],
        ["2023-08-02", True],
        ["2023-08-12", True],
        ["2023-08-14", True],
        ["2023-08-24", False],
        ["2023-10-13", True],
        ["2023-10-23", True],
        ["2023-12-05", True],
        ["2023-12-10", True],
        ["2023-12-11", True],
        ["2023-12-19", False],
    ]

    df_pd = _transform(spark, data, "TH", False)
    _check_output(df_pd)
