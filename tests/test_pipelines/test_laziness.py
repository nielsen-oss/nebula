"""Test the laziness functionality."""

import polars as pl

from nebula import TransformerPipeline
from nebula import nebula_storage as ns
from nebula.base import LazyWrapper
from nebula.pipelines.pipeline_loader import load_pipeline
from nebula.transformers import AddLiterals

from ..auxiliaries import pl_assert_equal
from .auxiliaries import *


def _get_df_input():
    data = [
        [0, "a"],
        [1, "b"],  # duplicate
        [1, "b"],
    ]
    return pl.DataFrame(data, schema=["c1", "c2"], orient="row")


def _get_expected_output() -> pl.DataFrame:
    df = (
        _get_df_input()
        .unique()
        .with_columns(
            pl.lit("lazy").alias("c3"),
            pl.lit("lazy-ns").alias("c4"),
            pl.lit("lazy-ns").alias("c5"),
        )
    )
    return df


def test_laziness_py():
    """Test laziness capabilities using a python pipeline."""
    ns.clear()
    df_input: pl.DataFrame = _get_df_input()
    list_trf = [
        Distinct(),
        LazyWrapper(AddLiterals, data=[{"alias": "c3", "value": "lazy"}]),
        LazyWrapper(AddLiterals, data=[{"alias": "c4", "value": (ns, "my_key")}]),  # as list
        LazyWrapper(AddLiterals, data=({"alias": "c5", "value": (ns, "my_key2")},)),  # as tuple
    ]
    pipe = TransformerPipeline(list_trf)
    pipe.show(add_params=True)

    try:
        ns.set("my_key", "lazy-ns")
        ns.set("my_key2", "lazy-ns")
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

    pipe_kws = dict(extra_transformers=[ExtraTransformers])
    pipe = load_pipeline(data, **pipe_kws)
    pipe.show(add_params=True)

    try:
        ns.set("my_key", "lazy-ns")
        ns.set("my_key2", "lazy-ns")
        df_chk = pipe.run(df_input, show_params=True)
        df_expected = _get_expected_output()
        pl_assert_equal(df_chk, df_expected, sort=df_chk.columns)
    finally:
        ns.clear()
