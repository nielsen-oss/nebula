"""Unit-test for FillColumnWithMode."""

import pyspark.sql.functions as F
import pytest
from pyspark.sql.types import StringType, StructField, StructType

from nlsn.nebula.spark_transformers import FillColumnWithMode


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    fields = [
        StructField("platform", StringType(), True),
        StructField("device_type", StringType(), True),
        StructField("os_group", StringType(), True),
    ]

    data = [
        ["OTT", "STV", "Android"],
        ["OTT", "STV", None],
        ["OTT", "STV", "Android"],
        ["OTT", "STV", "Linux"],
        ["OTT", "STV", "Android"],
        ["OTT", "STV", None],
        ["DSK", "DSK", "Windows"],
        ["DSK", "DSK", None],
        ["DSK", "DSK", "MacOS"],
        ["DSK", "DSK", "MacOS"],
        ["DSK", "DSK", "MacOS"],
        ["MBL", "TAB", None],
        ["MBL", "TAB", None],
        ["MBL", "TAB", None],
        ["MBL", "TAB", None],
        ["MBL", "PHN", None],
        ["MBL", "PHN", "iOS"],
    ]

    return spark.createDataFrame(data, schema=StructType(fields)).persist()


def _assert_results(df_chk, df_exp, n_exp: int):
    n_chk = df_chk.filter(F.col("os_group").isNull()).count()

    m1 = "Unexpected null values in the output dataframe"
    assert n_chk == n_exp, m1

    sort_columns = ["platform", "device_type", "os_group"]
    m2 = f"Different columns: {df_chk.columns}"
    assert sort_columns == df_chk.columns, m2

    df_chk_pd = df_chk.select(sort_columns).sort(sort_columns).toPandas()
    df_exp_pd = df_exp.select(sort_columns).sort(sort_columns).toPandas()

    m3 = "Transformer did not fill the values correctly!"
    assert df_chk_pd.equals(df_exp_pd), m3


@pytest.mark.parametrize("df_mode_to_pandas", [True, False])
def test_fill_column_with_mode_fill_only(spark, df_input, df_mode_to_pandas: bool):
    """Test FillColumnWithMode transformer w/ fill option and w/o replace."""
    t = FillColumnWithMode(
        to_fill_cols=["os_group"],
        groupby_columns=["platform", "device_type"],
        replace=False,
        df_mode_to_pandas=df_mode_to_pandas,
    )

    df_chk = t.transform(df_input)

    output_data_exp = [
        ["OTT", "STV", "Android"],
        ["OTT", "STV", "Android"],
        ["OTT", "STV", "Android"],
        ["OTT", "STV", "Linux"],
        ["OTT", "STV", "Android"],
        ["OTT", "STV", "Android"],
        ["DSK", "DSK", "Windows"],
        ["DSK", "DSK", "MacOS"],
        ["DSK", "DSK", "MacOS"],
        ["DSK", "DSK", "MacOS"],
        ["DSK", "DSK", "MacOS"],
        ["MBL", "TAB", None],
        ["MBL", "TAB", None],
        ["MBL", "TAB", None],
        ["MBL", "TAB", None],
        ["MBL", "PHN", "iOS"],
        ["MBL", "PHN", "iOS"],
    ]

    # OUTPUT of FILL transformer
    # +--------+-----------+--------+
    # |platform|device_type|os_group|
    # +--------+-----------+--------+
    # |     OTT|        STV| Android|
    # |     OTT|        STV| Android|
    # |     OTT|        STV| Android|
    # |     OTT|        STV|   Linux|
    # |     OTT|        STV| Android|
    # |     OTT|        STV| Android|
    # |     DSK|        DSK| Windows|
    # |     DSK|        DSK|   MacOS|
    # |     DSK|        DSK|   MacOS|
    # |     DSK|        DSK|   MacOS|
    # |     DSK|        DSK|   MacOS|
    # |     MBL|        TAB|    null| # cannot be filled since no values are available
    # |     MBL|        TAB|    null| # cannot be filled since no values are available
    # |     MBL|        TAB|    null| # cannot be filled since no values are available
    # |     MBL|        TAB|    null| # cannot be filled since no values are available
    # |     MBL|        PHN|     iOS|
    # |     MBL|        PHN|     iOS|
    # +--------+-----------+--------+

    df_exp = spark.createDataFrame(output_data_exp, schema=df_input.schema)
    _assert_results(df_chk, df_exp, 4)


@pytest.mark.parametrize("df_mode_to_pandas", [True, False])
def test_fill_column_with_mode_replace(spark, df_input, df_mode_to_pandas: bool):
    """Test FillColumnWithMode transformer w/ replace option and w/o fill."""
    t = FillColumnWithMode(
        to_fill_cols=["os_group"],
        groupby_columns=["platform", "device_type"],
        replace=True,
        df_mode_to_pandas=df_mode_to_pandas,
    )

    df_chk = t.transform(df_input)

    output_data_exp = [
        ["OTT", "STV", "Android"],
        ["OTT", "STV", "Android"],
        ["OTT", "STV", "Android"],
        ["OTT", "STV", "Android"],
        ["OTT", "STV", "Android"],
        ["OTT", "STV", "Android"],
        ["DSK", "DSK", "MacOS"],
        ["DSK", "DSK", "MacOS"],
        ["DSK", "DSK", "MacOS"],
        ["DSK", "DSK", "MacOS"],
        ["DSK", "DSK", "MacOS"],
        ["MBL", "TAB", None],
        ["MBL", "TAB", None],
        ["MBL", "TAB", None],
        ["MBL", "TAB", None],
        ["MBL", "PHN", "iOS"],
        ["MBL", "PHN", "iOS"],
    ]

    # OUTPUT with replace=True
    # +--------+-----------+--------+
    # |platform|device_type|os_group|
    # +--------+-----------+--------+
    # |OTT     |STV        |Android |
    # |OTT     |STV        |Android |
    # |OTT     |STV        |Android |
    # |OTT     |STV        |Android |
    # |OTT     |STV        |Android |
    # |OTT     |STV        |Android |
    # |DSK     |DSK        |MacOS   |
    # |DSK     |DSK        |MacOS   |
    # |DSK     |DSK        |MacOS   |
    # |DSK     |DSK        |MacOS   |
    # |DSK     |DSK        |MacOS   |
    # |MBL     |TAB        |null    | # cannot be filled since no values are available
    # |MBL     |TAB        |null    | # cannot be filled since no values are available
    # |MBL     |TAB        |null    | # cannot be filled since no values are available
    # |MBL     |TAB        |null    | # cannot be filled since no values are available
    # |MBL     |PHN        |iOS     |
    # |MBL     |PHN        |iOS     |
    # +--------+-----------+--------+

    df_exp = spark.createDataFrame(output_data_exp, schema=df_input.schema)
    _assert_results(df_chk, df_exp, 4)
