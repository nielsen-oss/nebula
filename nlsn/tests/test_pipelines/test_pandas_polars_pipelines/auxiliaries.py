"""Auxiliaries for Pandas / Polars pipelines."""

import pandas as pd
import polars as pl

from nlsn.nebula.pipelines.pipeline_loader import load_pipeline
from nlsn.nebula.shared_transformers import Count
from nlsn.tests.auxiliaries import pandas_to_polars
from nlsn.tests.test_pipelines.pipeline_yaml.auxiliaries import load_yaml

from . import _custom_extra_transformers

__all__ = [
    "align_and_assert",
    "get_input_pandas_df",
    "get_pandas_polars_pipe",
    "run_pandas_polars_pipeline",
]


def mock_split_function():
    """Mock split function."""
    return


_PIPE_KWARGS = {
    "extra_transformers": [_custom_extra_transformers],
    "extra_functions": [mock_split_function],
}


def align_and_assert(
    backend, df1, df2, check_dtype: bool = True, force_sort: bool = False
):
    """Align and assert two dataframes.

    Args:
        backend (str):
            Backend to use.
        df1 (pd.DataFrame | pl.DataFrame):
            First dataframe.
        df2 (pd.DataFrame | pl.DataFrame):
            Second dataframe.
        check_dtype (bool):
            If False don't check the dtypes.
            If backend == "pandas" this parameter is ignored.
        force_sort (bool):
            If True sort the dataframe by all columns.
    """
    if isinstance(df1, pl.DataFrame):
        df1 = df1.to_pandas()

    if isinstance(df2, pl.DataFrame):
        df2 = df2.to_pandas()

    if (backend == "polars") or force_sort:
        # In Polars pipeline the order is not ensured.
        sort = list(df1.columns)
        df2 = df2.sort_values(sort).reset_index(drop=True)
        df1 = df1.sort_values(sort).reset_index(drop=True)
    else:
        # Sometimes when converting from pandas to polars
        # and vice versa, there is an automatic upcasting
        # from int32 to int64
        check_dtype = True

    pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)


def get_input_pandas_df():
    """Input pandas dataframe."""
    _nan = float("nan")

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
        [10.1234, "   ", "b"],
        [11.1234, "a", None],
        [12.1234, None, "b"],
        [13.1234, _nan, "b"],
        [14.1234, _nan, None],
        [15.1234, _nan, _nan],
    ]

    return pd.DataFrame(data, columns=["idx", "c1", "c2"])


def get_pandas_polars_pipe(yml_file: str, name: str, source: str, dict_pipelines: dict):
    """Load a Pandas or Polars pipeline."""
    if source == "yaml":
        yaml_data = load_yaml(yml_file)
        return load_pipeline(yaml_data[name], **_PIPE_KWARGS)
    elif source == "py":
        return dict_pipelines[name]()
    else:
        raise RuntimeError


def run_pandas_polars_pipeline(
    df_input,
    yml_file: str,
    pipe_name: str,
    source: str,
    backend: str,
    dict_pipelines: dict,
):
    """Run a pipeline using a Pandas or Polars DF and return a Pandas DF."""
    df = df_input.copy()
    df = pandas_to_polars(backend, df)
    pipeline = get_pandas_polars_pipe(yml_file, pipe_name, source, dict_pipelines)
    pipeline.show_pipeline(add_transformer_params=True)
    df_out = pipeline.run(df, Count())  # Add a forced transformer
    if backend == "polars":
        return df_out.to_pandas()
    return df_out
