"""Unit-test for ClipOrNull."""

import operator

import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, StructField, StructType

from nlsn.nebula.spark_transformers import ClipOrNull


def test_clip_or_null_both_ref_value_and_comparison_col():
    """Test with both 'ref_value' and 'comparison_col'."""
    with pytest.raises(AssertionError):
        ClipOrNull(
            input_col="a",
            operator="lt",
            ref_value=5,
            perform="clip",
            comparison_col="x",
        )


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark: SparkSession):
    fields = [
        StructField("value", FloatType(), True),
        StructField("c1", FloatType(), True),
    ]

    _nan = float("nan")

    data = [
        [0.0, None],
        [0.0, 5.0],
        [1.0, 5.0],
        [1.0, _nan],
        [9.0, 9.0],
        [9.0, _nan],
        [None, 5.0],
        [None, None],
        [None, _nan],
        [_nan, 1.0],
        [_nan, None],
        [_nan, _nan],
    ]
    df = spark.createDataFrame(data, schema=StructType(fields))
    df = df.withColumn("orig_input", F.col("value"))
    # Add "c2" column, like "c1" but cast to integer
    return df.withColumn("c2", F.col("c1").cast("int")).cache()


def _assert(results: list, op: str, perform: str):
    for input_value, cmp_value, chk in results:
        exp_null = input_value is None
        exp_null |= pd.isna(input_value)
        exp_null |= cmp_value is None
        exp_null |= pd.isna(cmp_value)

        msg = f"input_value={input_value}, cmp_value={cmp_value}, chk={chk}"

        if exp_null:
            assert chk is None, msg
            continue

        cond = getattr(operator, op)(input_value, cmp_value)
        if cond:
            # print("COND", input_value, cmp_value, chk)
            if perform == "clip":
                assert chk == cmp_value, msg
            elif perform == "null":
                assert chk is None, msg
            else:
                raise RuntimeError

        else:
            # print("ELSE", input_value, cmp_value, chk)
            assert input_value == chk, msg


@pytest.mark.parametrize("comparison_col", [None, "c1", "c2"])
@pytest.mark.parametrize("op", ["lt", "gt"])
@pytest.mark.parametrize("perform", ["clip", "null"])
@pytest.mark.parametrize("output_col", ["result", None])
def test_clip_or_null(df_input, comparison_col, op: str, perform: str, output_col: str):
    """Test ClipOrNull transformer."""
    if comparison_col:
        ref = comparison_col
        ref_value = None
    else:
        ref = "ref_value"
        ref_value = 5.0
        df_input = df_input.withColumn(ref, F.lit(ref_value))

    t = ClipOrNull(
        input_col="value",
        operator=op,
        ref_value=ref_value,
        comparison_col=comparison_col,
        perform=perform,
        output_col=output_col,
    )

    if output_col:
        cols = ["value", ref, "result"]
    else:
        cols = ["orig_input", ref, "value"]

    df_chk = t.transform(df_input)
    df_chk_selection = df_chk.select(cols)
    results = df_chk_selection.rdd.map(lambda x: x[:]).collect()
    _assert(results, op, perform)
