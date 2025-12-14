"""Test the laziness functionality."""

import polars as pl

from nlsn.nebula import nebula_storage as ns
from nlsn.nebula.base import LazyWrapper, nlazy
from nlsn.nebula.pipelines.pipeline_loader import load_pipeline
from nlsn.nebula.pipelines.pipelines import TransformerPipeline
from nlsn.nebula.transformers import AddLiterals
from .auxiliaries import *


def _get_df_input():
    data = [
        [0, "a"],
        [1, "b"],  # duplicate
        [1, "b"],
    ]
    return pl.DataFrame(data, schema=["c1", "c2"])


def _get_expected_output() -> pl.DataFrame:
    df = (
        _get_df_input().unique()
        .with_columns(
            pl.lit("lazy").alias("c3"),
            pl.lit("lazy-func").alias("c4"),
            pl.lit("lazy-ns").alias("c5"),
        )
    )
    return df


@nlazy
def my_func() -> str:  # mock function
    return "lazy-func"


def test_laziness_py():
    """Test laziness capabilities using a python pipeline."""
    ns.clear()
    df_input: pl.DataFrame = _get_df_input()
    list_trf = [
        Distinct(),
        LazyWrapper(AddLiterals, data=[{"alias": "c3", "value": "lazy"}]),
        LazyWrapper(AddLiterals, data=[{"alias": "c4", "value": my_func}]),
        LazyWrapper(AddLiterals, data=[{"alias": "c5", "value": (ns, "my_key")}]),
    ]
    pipe = TransformerPipeline(list_trf)
    pipe.show_pipeline(add_transformer_params=True)

    try:
        ns.set("my_key", "lazy-ns")
        df_chk = pipe.run(df_input)
        df_exp = _get_expected_output()
        pl_assert_equal(df_chk, df_exp, sort=["c1"])
    finally:
        ns.clear()


def test_laziness_yaml():
    """Test laziness capabilities using the YAML file."""
    ns.clear()
    df_input: pl.DataFrame = _get_df_input()
    data = load_yaml("laziness.yml")

    pipe_kws = dict(extra_functions={"my_func": my_func},
                    extra_transformers=[ExtraTransformers])
    pipe = load_pipeline(data, **pipe_kws)
    pipe.show_pipeline(add_transformer_params=True)

    try:
        ns.set("my_key", "lazy-ns")
        df_chk = pipe.run(df_input)
        df_expected = _get_expected_output()
        pl_assert_equal(df_chk, df_expected, sort=df_chk.columns)
    finally:
        ns.clear()
