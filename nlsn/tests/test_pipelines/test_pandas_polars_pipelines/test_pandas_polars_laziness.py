"""Test the laziness functionality using Pandas."""

import pandas as pd

from nlsn.nebula import nebula_storage as ns
from nlsn.nebula.base import LazyWrapper, nlazy
from nlsn.nebula.pipelines.pipeline_loader import load_pipeline
from nlsn.nebula.pipelines.pipelines import TransformerPipeline
from nlsn.nebula.shared_transformers import Distinct, WithColumn
from nlsn.tests.test_pipelines.pipeline_yaml.auxiliaries import load_yaml


def _get_df_input():
    """Get input dataframe."""
    data = [
        [0, "a"],
        [1, "b"],  # duplicate
        [1, "b"],
    ]
    return pd.DataFrame(data, columns=["c1", "c2"])


def _get_expected_output() -> pd.DataFrame:
    """Get the expected output dataframe."""
    df = _get_df_input().drop_duplicates()
    df["c3"] = "lazy"
    df["c4"] = "lazy-func"
    df["c5"] = "lazy-ns"
    return df


@nlazy
def my_func() -> str:
    """Mock function."""
    return "lazy-func"


def test_laziness_py():
    """Test laziness capabilities using a python pipeline."""
    ns.clear()
    df_input: pd.DataFrame = _get_df_input()
    list_trf = [
        Distinct(),
        LazyWrapper(WithColumn, column_name="c3", value="lazy"),
        LazyWrapper(WithColumn, column_name="c4", value=my_func),
        LazyWrapper(WithColumn, column_name="c5", value=(ns, "my_key")),
    ]
    pipe = TransformerPipeline(list_trf)
    pipe.show_pipeline(add_transformer_params=True)

    try:
        ns.set("my_key", "lazy-ns")
        df_chk = pipe.run(df_input)
        df_expected = _get_expected_output()
        pd.testing.assert_frame_equal(df_chk, df_expected)
    finally:
        ns.clear()


def test_laziness_yaml():
    """Test laziness capabilities using the YAML file."""
    ns.clear()
    df_input: pd.DataFrame = _get_df_input()
    data = load_yaml("laziness.yml")

    pipe = load_pipeline(data, extra_functions={"my_func": my_func})
    pipe.show_pipeline(add_transformer_params=True)

    try:
        ns.set("my_key", "lazy-ns")
        df_chk = pipe.run(df_input)
        df_expected = _get_expected_output()
        pd.testing.assert_frame_equal(df_chk, df_expected)
    finally:
        ns.clear()
