"""Test 'apply_to_rows' pipeline functionalities."""

# pylint: disable=unused-wildcard-import

import pandas as pd
import pytest

from nlsn.nebula.pipelines._pandas_split_functions import pandas_split_function
from nlsn.nebula.storage import nebula_storage as ns
from nlsn.tests.auxiliaries import pandas_to_polars

from .._shared import DICT_APPLY_TO_ROWS_PIPELINES
from .auxiliaries import *

_YML_FILE: str = "apply_to_rows.yml"


@pytest.fixture(scope="module", name="df_input")
def _get_df_input():
    return get_input_pandas_df()


_SOURCES = ["py", "yaml"]
_BACKENDS = ["pandas", "polars"]


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("source", _SOURCES)
def test_apply_to_rows_is_null_dead_end(df_input, source, backend: str):
    """Test 'apply_to_rows' & 'dead-end' with 'storage'."""
    ns.clear()

    pipe_name = "apply_to_rows_is_null_dead_end"
    df_out = run_pandas_polars_pipeline(
        df_input, _YML_FILE, pipe_name, source, backend, DICT_APPLY_TO_ROWS_PIPELINES
    )

    df_chk_fork = ns.get("df_fork")
    if backend == "polars":
        df_chk_fork = df_chk_fork.to_pandas()

    df_exp = df_input[~df_input["c1"].isnull()]

    align_and_assert(backend, df_out, df_exp)

    df_exp_fork = df_input[df_input["c1"].isnull()].copy()
    df_exp_fork["new"] = -1

    align_and_assert(backend, df_chk_fork, df_exp_fork, check_dtype=False)
    ns.clear()


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("source", _SOURCES)
def test_apply_to_rows_gt(df_input, source, backend: str):
    """Test 'apply_to_rows'."""
    ns.clear()

    pipe_name = "apply_to_rows_gt"
    df_out = run_pandas_polars_pipeline(
        df_input, _YML_FILE, pipe_name, source, backend, DICT_APPLY_TO_ROWS_PIPELINES
    )

    df_apply, df_untouched = pandas_split_function(
        df_input,
        input_col="idx",
        operator="gt",
        value=5,
        compare_col=None,
    )
    df_fork = df_apply.copy()
    df_fork["c1"] = "x"
    df_exp = pd.concat([df_untouched, df_fork], axis=0)

    pd.testing.assert_frame_equal(df_out, df_exp)
    ns.clear()


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("source", _SOURCES)
def test_apply_to_rows_comparison_col(df_input, source, backend: str):
    """Test 'apply_to_rows' with 'allow_missing_columns'."""
    ns.clear()

    pipe_name = "apply_to_rows_comparison_col"
    df_out = run_pandas_polars_pipeline(
        df_input, _YML_FILE, pipe_name, source, backend, DICT_APPLY_TO_ROWS_PIPELINES
    )

    df_apply, df_untouched = pandas_split_function(
        df_input,
        input_col="c1",
        operator="gt",
        value=None,
        compare_col="c2",
    )
    df_fork = df_apply.copy()
    df_fork["new_column"] = "new_value"
    df_exp = pd.concat([df_untouched, df_fork], axis=0)

    align_and_assert(backend, df_out, df_exp)

    ns.clear()


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("source", _SOURCES)
def test_apply_to_rows_error(df_input, source, backend: str):
    """Test wrong 'apply_to_rows' without 'allow_missing_columns'."""
    ns.clear()

    df_input = pandas_to_polars(backend, df_input)

    pipe_name = "apply_to_rows_error"
    pipe = get_pandas_polars_pipe(
        _YML_FILE, pipe_name, source, DICT_APPLY_TO_ROWS_PIPELINES
    )
    pipe.show_pipeline()

    with pytest.raises(ValueError):
        pipe.run(df_input)

    ns.clear()


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("source", _SOURCES)
def test_apply_to_rows_otherwise(df_input, source, backend: str):
    """Test 'apply_to_rows' with 'otherwise' pipeline."""
    ns.clear()

    pipe_name = "apply_to_rows_otherwise"
    df_out = run_pandas_polars_pipeline(
        df_input, _YML_FILE, pipe_name, source, backend, DICT_APPLY_TO_ROWS_PIPELINES
    )

    df_apply, df_untouched = pandas_split_function(
        df_input,
        input_col="idx",
        operator="gt",
        value=5,
        compare_col=None,
    )
    df_fork = df_apply.copy()
    df_fork["c1"] = "x"

    df_otherwise = df_untouched.copy()
    df_otherwise["c1"] = "other"

    df_exp = pd.concat([df_otherwise, df_fork], axis=0)

    pd.testing.assert_frame_equal(df_out, df_exp)
    ns.clear()


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("source", _SOURCES)
def test_apply_to_rows_skip_if_empty(df_input, source, backend: str):
    """Test 'apply_to_rows' with 'skip_if_empty' True."""
    ns.clear()

    pipe_name = "apply_to_rows_skip_if_empty"
    df_out = run_pandas_polars_pipeline(
        df_input, _YML_FILE, pipe_name, source, backend, DICT_APPLY_TO_ROWS_PIPELINES
    )

    pd.testing.assert_frame_equal(df_out, df_input.copy())
    ns.clear()
