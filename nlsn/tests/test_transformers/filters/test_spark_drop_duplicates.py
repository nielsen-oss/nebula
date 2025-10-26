"""Unit-test for Distinct and DropDuplicates."""

import pytest
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from nlsn.nebula.spark_transformers import Distinct, DropDuplicates
from nlsn.nebula.spark_util import drop_duplicates_no_randomness


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


def test_distinct(df_input):
    """Test Distinct transformer."""
    t = Distinct()
    df_chk = t.transform(df_input)
    df_exp = df_input.distinct()
    assert_df_equality(df_chk, df_exp, ignore_row_order=True)


def test_drop_duplicates_no_subset(df_input):
    """Test DropDuplicates transformer w/o subset columns."""
    t = DropDuplicates()
    df_chk = t.transform(df_input)
    df_exp = df_input.drop_duplicates()
    assert_df_equality(df_chk, df_exp, ignore_row_order=True)


def _test_complex_types(df_input):
    mapping = F.create_map(F.lit("a"), F.lit("1"), F.lit("b"), F.lit("2"))
    array = F.array(F.lit("x"), F.lit("y"))

    df_complex = df_input.withColumn("mapping", mapping).withColumn("array", array)

    subset = ["platform", "device_type"]
    chk = drop_duplicates_no_randomness(df_complex, subset)

    # just to check if MapType and ArrayType raise any error
    chk.count()


def test_drop_duplicates_subset(df_input):
    """Test DropDuplicates transformer w/ subset columns."""
    # just to check if MapType and ArrayType raise any error
    _test_complex_types(df_input)

    # check if the same df order in the opposite manner gives the same result
    df_desc = df_input.sort([F.col(i).desc() for i in df_input.columns])
    df_asc = df_input.sort([F.col(i).asc() for i in df_input.columns])

    subset = ["platform", "device_type"]
    t = DropDuplicates(columns=subset)
    df_chk_desc = t.transform(df_desc)
    df_chk_asc = t.transform(df_asc)

    assert_df_equality(df_chk_desc, df_chk_asc, ignore_row_order=True)
