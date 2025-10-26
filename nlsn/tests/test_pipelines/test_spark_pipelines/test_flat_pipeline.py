"""Test a simple flat pipeline."""

import pytest
from chispa import assert_df_equality
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from nlsn.nebula.pipelines.pipelines import TransformerPipeline
from nlsn.nebula.spark_transformers import Cache, Distinct, LogDataSkew, NanToNull

_TRANSFORMERS = [
    NanToNull(glob="*"),
    Distinct(),
]

_INTERLEAVED = [
    Cache(),
    LogDataSkew(),
]

_nan = float("nan")


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    """Get input dataframe."""
    fields = [
        StructField("idx", IntegerType(), True),
        StructField("c1", StringType(), True),
        StructField("c2", StringType(), True),
    ]

    data = [
        [0, "a", "b"],
        [0, "a", "b"],  # dropped with Distinct()
        [0, "a", "b"],  # dropped with Distinct()
        [1, "a", "  b"],
        [2, "  a  ", "  b  "],
        [3, "", ""],
        [4, "   ", "   "],
        [5, None, None],
        [6, " ", None],
        [7, "", None],
        [8, "a", None],
        [9, "a", ""],
        [10, "   ", "b"],
        [11, "a", None],
        [12, None, "b"],
        [13, _nan, "b"],
        [14, _nan, None],
        [15, _nan, _nan],
    ]

    return spark.createDataFrame(data, schema=StructType(fields)).persist()


@pytest.fixture(scope="module", name="df_exp")
def _get_df_exp(df_input):
    """Get the expected dataframe."""
    df_ret = df_input
    for t in _TRANSFORMERS:
        df_ret = t.transform(df_ret)
    # "interleaved" transformers are just for log in this unit-test.
    return df_ret.persist()


@pytest.mark.parametrize(
    "interleaved, prepend_interleaved, append_interleaved, name",
    [
        (None, True, False, None),
        ([], True, False, None),
        (_INTERLEAVED, True, False, "name_01"),
        (_INTERLEAVED, True, True, "name_02"),
        (_INTERLEAVED[0], False, False, "name_03"),
    ],
)
def test_pipeline_flat_list_transformers(
    df_input,
    df_exp,
    interleaved: list,
    prepend_interleaved: bool,
    append_interleaved: bool,
    name: str,
):
    """Test TransformerPipeline pipeline w/ list of transformers."""
    pipe = TransformerPipeline(
        _TRANSFORMERS,
        interleaved=interleaved,
        prepend_interleaved=prepend_interleaved,
        append_interleaved=append_interleaved,
        name=name,
        backend="spark",
    )
    pipe.show_pipeline()
    pipe._print_dag()

    df_chk = pipe.run(df_input)
    assert_df_equality(df_chk, df_exp, ignore_row_order=True)


@pytest.mark.parametrize("split_func", [None, lambda x: x])
def test_pipeline_single_split(df_input, df_exp, split_func):
    """Test TransformerPipeline pipeline w/ list of transformers."""
    pipe = TransformerPipeline(
        {"no split": _TRANSFORMERS},
        split_function=split_func,
        name="single split",
    )

    pipe.show_pipeline()
    pipe._print_dag()

    df_chk = pipe.run(df_input)
    assert_df_equality(df_chk, df_exp, ignore_row_order=True)
