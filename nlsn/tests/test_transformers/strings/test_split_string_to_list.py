"""Unit-test for SplitStringToList."""

import pytest
from chispa import assert_df_equality
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType

from nlsn.nebula.spark_transformers import SplitStringToList


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark: SparkSession):
    data = [
        [1, "apple,orange,banana", "1"],
        [2, "grape,kiwi", "0,1"],
        [3, "melon", "0,1,2"],
        [3, "oneAtwoBthreeC", "0,1,2"],
    ]
    schema = "id: int, c_str: string, c_int: string"
    return spark.createDataFrame(data, schema=schema).cache()


@pytest.mark.parametrize("input_col", ["c_str", "c_int"])
@pytest.mark.parametrize("regex", [None, r"[ABC]"])
def test_string_to_list_no_cast(df_input, input_col: str, regex: str):
    """Test SplitStringToList with default cast."""
    kws = {"input_col": input_col, "output_col": "result"}
    if regex is not None:
        kws["regex"] = regex

    t = SplitStringToList(**kws)
    df_chk = t.transform(df_input)
    regex = regex or r"[^a-zA-Z0-9\s]"
    df_exp = df_input.withColumn("result", F.split(input_col, regex))
    assert_df_equality(
        df_chk,
        df_exp,
        ignore_row_order=True,
        ignore_column_order=True,
        ignore_nullable=True,
    )


@pytest.mark.parametrize("limit", [-1, 1])
def test_string_to_list_cast_int(df_input, limit: int):
    """Test SplitStringToList with default integer cast."""
    t = SplitStringToList(
        input_col="c_int", output_col="result", limit=limit, cast="int"
    )
    df_chk = t.transform(df_input)

    exp_type = ArrayType(IntegerType())
    df_exp = df_input.withColumn(
        "result", F.split("c_int", r"[^a-zA-Z0-9\s]", limit).cast(exp_type)
    )

    assert_df_equality(
        df_chk,
        df_exp,
        ignore_row_order=True,
        ignore_column_order=True,
        ignore_nullable=True,
    )
