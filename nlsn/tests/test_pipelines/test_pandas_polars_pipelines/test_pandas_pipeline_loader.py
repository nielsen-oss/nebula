"""Test the pipeline loader starting from YAML data."""

import pandas as pd
import pytest

from nlsn.nebula.pipelines.pipeline_loader import load_pipeline
from nlsn.tests.test_pipelines.pipeline_yaml.auxiliaries import load_yaml


def _get_df_input():
    """Get input dataframe."""
    data = [
        [0, "a", "b"],
        [0, "a", "b"],
        [0, "a", "b"],
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
        [13, None, "b"],
        [14, "a", None],
        [15, "a", None],
    ]
    return pd.DataFrame(data, columns=["idx", "c1", "c2"])


def _split_func_outer(df):
    mask = df["idx"] < 10
    return {"outer_low": df[mask].copy(), "outer_hi": df[~mask].copy()}


def _split_func_inner(df):
    mask = df["idx"] < 5
    return {"inner_low": df[mask].copy(), "inner_hi": df[~mask].copy()}


@pytest.mark.parametrize("pipeline_key", ["split-is-none", "split-is-empty-list"])
def test_mock_pipelines_empty_splits(pipeline_key: str):
    """Test split-pipelines with an empty split.

    The transformers in this pipeline do nothing, just count.

    Check whether the pipeline returns the same dataframe.
    """
    df_input = _get_df_input()
    fname = "empty_split.yml"
    data = load_yaml(fname)[pipeline_key]

    dict_split_functions = {
        "split_func_outer": _split_func_outer,
        "split_func_inner": _split_func_inner,
    }

    pipe = load_pipeline(data, extra_functions=dict_split_functions)
    pipe.show_pipeline(add_transformer_params=True)

    n_exp = df_input.shape[0]

    df_chk = pipe.run(df_input)
    n_chk = df_chk.shape[0]

    assert n_chk == n_exp

    df_chk = df_chk.sort_index()

    pd.testing.assert_frame_equal(df_input, df_chk)


def test_pipeline_loader_list_tuple():
    """Test the automatic pipeline conversion from list / tuple to dict."""
    df_input = _get_df_input()
    n_rows = 10
    core_pipe = [{"transformer": "Head", "params": {"n": n_rows}}]

    pipe_dict = load_pipeline({"pipeline": core_pipe}, backend="pandas")
    pipe_list = load_pipeline(core_pipe, backend="pandas", evaluate_loops=False)
    pipe_tuple = load_pipeline(tuple(core_pipe), backend="pandas")

    assert n_rows == pipe_dict.run(df_input).shape[0]
    assert n_rows == pipe_list.run(df_input).shape[0]
    assert n_rows == pipe_tuple.run(df_input).shape[0]


def test_pipeline_loader_splits_allow_missing_columns():
    """Test split-pipelines with an allow_missing_columns=True.

    Check whether the pipeline returns the same dataframe.
    """
    df_input = _get_df_input()
    fname = "split_allow_missing_columns.yml"
    data = load_yaml(fname)["split-allow-missing-columns"]

    dict_split_functions = {
        "split_func_outer": _split_func_outer,
    }

    pipe = load_pipeline(data, extra_functions=dict_split_functions)
    df_chk = pipe.run(df_input)
    assert df_chk.shape[0] == df_input.shape[0]

    dict_df = _split_func_outer(df_input)
    df_low = dict_df["outer_low"]
    df_low["new_column"] = "new_value"
    df_exp = pd.concat([dict_df["outer_hi"], df_low], axis=0)
    exp_columns = df_exp.columns.tolist()
    assert set(exp_columns) == set(df_chk.columns.tolist())
    pd.testing.assert_frame_equal(df_chk[exp_columns], df_exp)


class TestPipelineLoaderEdgeCase:
    """Test a split-pipeline with for loops and a split named 'loop'."""

    pipe_cfg = {
        "backend": "pandas",
        "split_function": "split_func",
        "pipeline": {
            "loop": {"transformer": "PrintSchema"},
            "hi_values": [{"transformer": "PrintSchema"}],
        },
    }

    @staticmethod
    def _split_func(df):
        mask = df["idx"] < 10
        return {"loop": df[mask].copy(), "hi_values": df[~mask].copy()}

    @pytest.mark.parametrize("evaluate_loops", [True, False])
    def test_split_named_loop(self, evaluate_loops):
        """A split cannot be named 'loop' if the pipe is loaded from a text source."""
        dict_split_functions = {"split_func": self._split_func}

        with pytest.raises(AssertionError):
            load_pipeline(
                self.pipe_cfg,
                extra_functions=dict_split_functions,
                evaluate_loops=evaluate_loops,
            )
