"""Unit-test for Sequence."""

import pytest
from chispa import assert_df_equality
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StructField, StructType

from nlsn.nebula.spark_transformers.arrays import Sequence


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    fields = [
        StructField("a", IntegerType(), True),
        StructField("b", IntegerType(), True),
        StructField("c", IntegerType(), True),
    ]

    data = [
        [0, 10, 2],
        [0, 1, 2],
        [3, 10, 2],
        [None, None, 1],
        [None, 5, 3],
        [None, 0, 1],
        [8, None, 9],
    ]

    return spark.createDataFrame(data, schema=StructType(fields)).persist()


@pytest.mark.parametrize(
    "start, stop, step", [(1, 10.1, 2), ("a", None, "b"), (None, 100, 1), (1, 10, 1.5)]
)
def test_assert_args(start, stop, step):
    """Test Sequence with invalid arguments."""
    with pytest.raises(TypeError):
        Sequence(column="x", start=start, stop=stop, step=step)


@pytest.mark.parametrize(
    "start, stop, step", [(1, 10, None), (1, 10, 2), ("a", "b", "c")]
)
def test_sequence(df_input, start, stop, step):
    """Test Sequence."""
    t = Sequence(column="out", start=start, stop=stop, step=step)
    df_chk = t.transform(df_input)

    seq = F.sequence(
        start if isinstance(start, str) else F.lit(start),
        stop if isinstance(stop, str) else F.lit(stop),
        step if isinstance(step, str) else F.lit(1 if step is None else step),
    )
    df_exp = df_input.withColumn("out", seq)

    assert_df_equality(df_chk, df_exp, ignore_row_order=True, ignore_nullable=True)
