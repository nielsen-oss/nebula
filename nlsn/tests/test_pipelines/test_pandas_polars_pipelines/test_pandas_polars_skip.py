"""Test 'skip' pipeline functionalities on pandas and polars."""

import pandas as pd
import pytest

from nlsn.nebula.pipelines.pipeline_loader import load_pipeline

from .._shared import DICT_SKIP_PIPELINE, DICT_SKIP_TRANSFORMER
from .auxiliaries import get_input_pandas_df, run_pandas_polars_pipeline

_YML_FILE: str = "skip.yml"
_SOURCES = ["py", "yaml"]
_BACKENDS = ["pandas", "polars"]


@pytest.fixture(scope="module", name="df_input")
def _get_df_input():
    return get_input_pandas_df()


@pytest.mark.parametrize("pipe_name", sorted(DICT_SKIP_PIPELINE))
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("source", _SOURCES)
def test_skip_pipeline(df_input, pipe_name: str, source: str, backend: str):
    """Test 'skip' pipeline functionality."""
    df_exp = run_pandas_polars_pipeline(
        df_input, _YML_FILE, pipe_name, source, backend, DICT_SKIP_PIPELINE
    )
    pd.testing.assert_frame_equal(df_input, df_exp)


def test_skip_pipeline_invalid_arguments():
    """Ensures that a skipped pipeline does not attempt to parse its arguments.

    This is useful when the pipeline is skipped and its arguments might be
    incomplete or invalid if the pipeline is not executed. By skipping the
    argument-parsing step, we avoid potential / unnecessary errors.
    """
    data = {
        "skip": True,
        "pipeline": [{"transformer": "NotExists", "wrong_key": "wrong_value"}],
    }
    load_pipeline(data)


@pytest.mark.parametrize("pipe_name", sorted(DICT_SKIP_TRANSFORMER))
@pytest.mark.parametrize("backend", _BACKENDS)
def test_skip_transformer(df_input, pipe_name: str, backend: str):
    """Test 'skip' / 'perform' transformer functionality."""
    df_exp = run_pandas_polars_pipeline(
        df_input, _YML_FILE, pipe_name, "yaml", backend, DICT_SKIP_TRANSFORMER
    )
    pd.testing.assert_frame_equal(df_input, df_exp)
