"""Test the pipeline loader starting from YAML data."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
import pytest

from nlsn.nebula.pipelines.pipeline_loader import load_pipeline
from nlsn.tests.test_pipelines.pipeline_yaml.auxiliaries import load_yaml
from .auxiliaries import *


@pytest.mark.parametrize("pipeline_key", ["split-is-none", "split-is-empty-list"])
def test_mock_pipelines_empty_splits(pipeline_key: str):
    """Test split-pipelines with an empty split.

    The transformers in this pipeline do nothing, just count.

    Ensure the pipeline returns the same dataframe.
    """
    df_input = pd.DataFrame({"idx": np.arange(0, 20), "c1": np.arange(10, 30)})
    file_name = "empty_split.yml"
    data = load_yaml(file_name)[pipeline_key]

    def _split_func_outer(df):
        mask = df["idx"] < 10
        return {"outer_low": df[mask].copy(), "outer_hi": df[~mask].copy()}

    def _split_func_inner(df):
        mask = df["idx"] < 5
        return {"inner_low": df[mask].copy(), "inner_hi": df[~mask].copy()}

    dict_split_functions = {
        "split_func_outer": _split_func_outer,
        "split_func_inner": _split_func_inner,
    }

    pipe = load_pipeline(data, extra_functions=dict_split_functions)
    pipe.show_pipeline(add_transformer_params=True)
    df_chk = pipe.run(df_input).sort_index()
    pd.testing.assert_frame_equal(df_input, df_chk)


def test_pipeline_loader_list_tuple():
    """Test the automatic pipeline conversion from list / tuple to dict."""
    df_input = pd.DataFrame({"idx": np.arange(0, 20), "c1": np.arange(10, 30)})
    core_pipe = [{"transformer": "AssertNotEmpty"}]

    pipe_dict = load_pipeline({"pipeline": core_pipe})
    pipe_list = load_pipeline(core_pipe, evaluate_loops=False)
    pipe_tuple = load_pipeline(tuple(core_pipe))

    pd.testing.assert_frame_equal(df_input, pipe_dict.run(df_input))
    pd.testing.assert_frame_equal(df_input, pipe_list.run(df_input))
    pd.testing.assert_frame_equal(df_input, pipe_tuple.run(df_input))


def test_extra_transformers():
    df = pl.DataFrame({"a": [1, 1, 2]})

    @dataclass
    class ExtraTransformers:
        Distinct = Distinct

    pipe = load_pipeline(
        [{"transformer": "Distinct"}],
        extra_transformers=[ExtraTransformers]
    )
    df_chk = pipe.run(df)
    df_exp = df.unique()
    pl_assert_equal(df_chk.sort("a"), df_exp.sort("a"))
