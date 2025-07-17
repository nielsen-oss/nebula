"""Unit-test for DiscardNulls."""

import pytest
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql.types import StringType, StructField, StructType

from nlsn.nebula.spark_transformers import DiscardNulls


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    fields = [
        StructField("a_1", StringType(), True),
        StructField("a_2", StringType(), True),
        StructField("a_3", StringType(), True),
        StructField("b_1", StringType(), True),
        StructField("b_2", StringType(), True),
        StructField("b_3", StringType(), True),
    ]

    # fmt: off
    data = [
        ("1", "11", None, "4", "41", "411"),
        ("1", "12", "120", "4", None, "412"),
        ("1", "12", "120", "4", "41", "412"),
        (None, None, None, None, None, None),
        ("1", "12", "120", "4", "41", None),
        (None, None, None, "4", "41", "412"),
    ]
    # fmt: on
    return spark.createDataFrame(data, schema=StructType(fields)).persist()


def test_discard_nulls_no_subset(df_input):
    """Test DiscardNulls transformer w/o any subsets."""
    how = "any"
    t = DiscardNulls(how=how)
    df_chk = t.transform(df_input)
    df_exp = df_input.dropna(how=how)
    assert_df_equality(df_chk, df_exp, ignore_row_order=True)


@pytest.mark.parametrize("how", ["any", "all"])
def test_discard_nulls_columns_subset(df_input, how):
    """Test DiscardNulls transformer selecting specific columns."""
    subset = ["a_1", "b_1"]
    t = DiscardNulls(columns=subset, how=how)
    df_chk = t.transform(df_input)

    df_exp = df_input.dropna(subset=subset, how=how)

    assert_df_equality(df_chk, df_exp, ignore_row_order=True)
