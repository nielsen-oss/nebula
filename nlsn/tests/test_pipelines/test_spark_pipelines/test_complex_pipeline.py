"""Test a complex pipeline."""

import pytest
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from nlsn.nebula.base import Transformer
from nlsn.nebula.pipelines.pipeline_loader import load_pipeline
from nlsn.nebula.pipelines.pipelines import TransformerPipeline
from nlsn.nebula.spark_transformers import Cache, Cast, Distinct, NanToNull, RoundValues
from nlsn.nebula.spark_util import null_cond_to_false
from nlsn.nebula.storage import nebula_storage as ns
from nlsn.tests.test_pipelines.pipeline_yaml.auxiliaries import load_yaml

_TRANSFORMERS = [
    NanToNull(glob="*"),
    Distinct(),
]

_INTERLEAVED = [
    Cache(),
]

_APPLY_BEFORE_APPENDING = Distinct()

_nan = float("nan")
_TRF_LOW: list = [RoundValues(input_columns="c1", precision=3)]


def _get_cond():
    return F.col("c1") < 10


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    """Get input dataframe."""
    fields = [
        StructField("c1", DoubleType(), True),
        StructField("c2", StringType(), True),
        StructField("c3", StringType(), True),
    ]

    data = [
        [0.1234, "a", "b"],
        [0.1234, "a", "b"],
        [0.1234, "a", "b"],
        [1.1234, "a", "  b"],
        [2.1234, "  a  ", "  b  "],
        [3.1234, "", ""],
        [4.1234, "   ", "   "],
        [5.1234, None, None],
        [6.1234, " ", None],
        [7.1234, "", None],
        [8.1234, "a", None],
        [9.1234, "a", ""],
        [9.1234, "a", ""],
        [10.1234, "   ", "b"],
        [10.1234, "   ", "b"],
        [11.1234, "a", None],
        [11.1234, "a", None],
        [11.1234, "a", None],
        [11.1234, "a", None],
        [11.1234, "a", None],
        [12.1234, None, "b"],
        [12.1234, None, "b"],
        [13.1234, _nan, "b"],
        [13.1234, _nan, "b"],
        [13.1234, _nan, "b"],
        [14.1234, _nan, None],
        [14.1234, _nan, None],
        [15.1234, _nan, _nan],
        [15.1234, _nan, _nan],
    ]

    return spark.createDataFrame(data, schema=StructType(fields)).persist()


def split_function(df):
    """Split function."""
    _cond = _get_cond()
    return {
        "low": df.filter(_cond),
        "hi": df.filter(~null_cond_to_false(_cond)),
    }


def _get_df_exp(df_input, trf_hi):
    """Get expected dataframes."""
    # flat-pipeline
    df_prep = df_input
    for t in _TRANSFORMERS:
        df_prep = t.transform(df_prep)

    # split-pipeline
    _cond = _get_cond()
    df_low = df_prep.filter(_cond)
    df_hi = df_prep.filter(~null_cond_to_false(_cond))

    for t in _TRF_LOW + [_APPLY_BEFORE_APPENDING]:
        df_low = t.transform(df_low)

    for t in trf_hi:
        if isinstance(t, Transformer):  # skip the storage request
            df_hi = t.transform(df_hi)

    # Store the partial dataframe as requested before applying _APPLY_BEFORE_APPENDING
    df_hi = df_hi.cache()
    df_hi_stored = df_hi

    for t in [_APPLY_BEFORE_APPENDING]:
        df_hi = t.transform(df_hi)

    # Emulate 'cast_subset_to_input_schema' = True
    df_hi = df_hi.withColumn("c1", F.col("c1").cast(DoubleType()))

    df_ret = df_low.unionByName(df_hi).cache()
    # "interleaved" transformers are just for log in this unit-test.
    return df_ret, df_hi_stored


@pytest.mark.parametrize(
    "interleaved, prepend_interleaved, append_interleaved",
    [
        (_INTERLEAVED, True, False),
        (_INTERLEAVED, True, True),
    ],
)
def test_complex_pipeline(
    df_input,
    interleaved: list,
    prepend_interleaved: bool,
    append_interleaved: bool,
):
    """Test TransformerPipeline composition."""
    ns.allow_debug(False)  # disallow debug

    trf_hi: list = [
        RoundValues(input_columns="c1", precision=1),
        Cast(cast={"c1": "float"}),
        {"store": "df_high"},
        {"storage_debug_mode": False},
        # This should not be stored as the debug is not active
        {"store_debug": "df_high_debug_false"},
        {"storage_debug_mode": True},
        # This should be stored as the debug is now active
        {"store_debug": "df_high_debug_true"},
        # Revert to the default state to avoid affecting other tests
        {"storage_debug_mode": False},
    ]

    dict_transformers = {
        "low": _TRF_LOW,
        "hi": trf_hi,
    }

    flat_pipeline = TransformerPipeline(
        _TRANSFORMERS,
        interleaved=interleaved,
        prepend_interleaved=prepend_interleaved,
        append_interleaved=append_interleaved,
        name="flat-pipeline",
    )

    split_pipeline = TransformerPipeline(
        dict_transformers,
        split_function=split_function,
        interleaved=interleaved,
        prepend_interleaved=prepend_interleaved,
        append_interleaved=append_interleaved,
        split_apply_before_appending=_APPLY_BEFORE_APPENDING,
        name="split-pipeline",
        cast_subset_to_input_schema=True,
    )

    pipe = TransformerPipeline([flat_pipeline, split_pipeline])
    df_exp, df_high_exp = _get_df_exp(df_input, trf_hi)

    pipe.show_pipeline()
    pipe._print_dag()

    df_chk = pipe.run(df_input)
    assert_df_equality(df_chk, df_exp, ignore_row_order=True)

    df_high_chk = ns.get("df_high")
    assert_df_equality(df_high_chk, df_high_exp, ignore_row_order=True)

    assert not ns.isin("df_high_debug_false")
    df_high_chk_debug = ns.get("df_high_debug_true")
    assert_df_equality(df_high_chk_debug, df_high_exp, ignore_row_order=True)


def test_pipeline_loader_with_storage(df_input):
    """Test split-pipelines with an empty split.

    The transformers in this pipeline do nothing, just count.

    Check if the pipeline returns the same dataframe.
    """
    fname = "storage.yml"
    data = load_yaml(fname)

    trf_hi: list = [
        RoundValues(input_columns="c1", precision=1),
        Cast(cast={"c1": "float"}),
        {"store": "df_high"},
    ]

    pipe = load_pipeline(data, extra_functions=split_function)
    pipe.show_pipeline(add_transformer_params=True)
    pipe._print_dag()

    df_exp, df_high_exp = _get_df_exp(df_input, trf_hi)

    pipe.show_pipeline()
    pipe._print_dag()

    df_chk = pipe.run(df_input)
    assert_df_equality(df_chk, df_exp, ignore_row_order=True)

    assert not ns.isin("df_high_debug_false")
    df_high_chk_debug = ns.get("df_high_debug_true")
    assert_df_equality(df_high_chk_debug, df_high_exp, ignore_row_order=True)
