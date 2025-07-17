"""Test 'branch' pipeline functionalities."""

# pylint: disable=unused-wildcard-import

import pandas as pd
import pytest

from nlsn.nebula.storage import nebula_storage as ns
from nlsn.tests.auxiliaries import pandas_to_polars

from .._shared import DICT_BRANCH_PIPELINE
from .auxiliaries import *

_YML_FILE: str = "branch.yml"
_SOURCES = ["py", "yaml"]
_BACKENDS = ["pandas", "polars"]


@pytest.fixture(scope="module", name="df_input")
def _get_df_input():
    return get_input_pandas_df()


def _concat_assert_clear(backend: str, df1, df2, df_out, new_col, new_value):
    df2[new_col] = new_value
    df_exp = pd.concat([df1, df2], axis=0)
    align_and_assert(backend, df_out, df_exp)
    ns.clear()


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_dead_end_without_storage(df_input, source, backend: str):
    """Test 'branch' & 'dead-end' without 'storage'."""
    ns.clear()

    pipe_name = "branch_dead_end_without_storage"
    df_out = run_pandas_polars_pipeline(
        df_input, _YML_FILE, pipe_name, source, backend, DICT_BRANCH_PIPELINE
    )
    df_exp = df_input.copy().drop_duplicates()
    align_and_assert(backend, df_out, df_exp)

    df_exp["new"] = -1
    df_chk_fork = ns.get("df_fork")
    align_and_assert(backend, df_chk_fork, df_exp, check_dtype=False)
    ns.clear()


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_append_without_storage(df_input, source, backend: str):
    """Test 'branch' & 'append' without 'storage'."""
    ns.clear()

    pipe_name = "branch_append_without_storage"
    df_out = run_pandas_polars_pipeline(
        df_input, _YML_FILE, pipe_name, source, backend, DICT_BRANCH_PIPELINE
    )

    df_distinct = df_input.drop_duplicates()
    df_fork = df_distinct.copy()
    _concat_assert_clear(backend, df_distinct, df_fork, df_out, "c1", "c")


@pytest.mark.parametrize("source", _SOURCES)
def test_branch_append_without_storage_error(df_input, source):
    """Test 'branch' & 'append' without 'storage'."""
    ns.clear()

    pipe_name = "branch_append_without_storage_error"
    pipeline = get_pandas_polars_pipe(
        _YML_FILE, pipe_name, source, DICT_BRANCH_PIPELINE
    )
    pipeline.show_pipeline()

    with pytest.raises(ValueError):
        pipeline.run(df_input)

    ns.clear()


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_append_without_storage_new_col(df_input, source, backend: str):
    """Test 'branch' & 'append' without 'storage'."""
    ns.clear()

    pipe_name = "branch_append_without_storage_new_col"
    df_out = run_pandas_polars_pipeline(
        df_input, _YML_FILE, pipe_name, source, backend, DICT_BRANCH_PIPELINE
    )

    df_distinct = df_input.drop_duplicates()
    df_fork = df_distinct.copy()
    _concat_assert_clear(
        backend, df_distinct, df_fork, df_out, "new_column", "new_value"
    )


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_join_without_storage(df_input, source, backend: str):
    """Test 'branch' & 'join' without 'storage'."""
    ns.clear()

    pipe_name = "branch_join_without_storage"
    df_out = run_pandas_polars_pipeline(
        df_input, _YML_FILE, pipe_name, source, backend, DICT_BRANCH_PIPELINE
    )

    df_distinct = df_input.drop_duplicates()
    df_fork = df_distinct[["idx"]].copy()
    df_fork["new"] = -1

    df_exp = df_distinct.merge(df_fork, on="idx", how="inner")
    align_and_assert(backend, df_out, df_exp, check_dtype=False)
    ns.clear()


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_dead_end_with_storage(df_input, source, backend: str):
    """Test 'branch' & 'dead-end' with 'storage'."""
    ns.clear()
    ns.set("df_x", df_input)

    pipe_name = "branch_dead_end_with_storage"
    df_out = run_pandas_polars_pipeline(
        df_input, _YML_FILE, pipe_name, source, backend, DICT_BRANCH_PIPELINE
    )

    df_exp = df_input.drop_duplicates()
    align_and_assert(backend, df_out, df_exp)

    df_chk_fork = ns.get("df_fork")
    df_exp_fork = df_input.copy()
    df_exp_fork["new"] = -1
    align_and_assert(backend, df_chk_fork, df_exp_fork)
    ns.clear()


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_append_with_storage(df_input, source, backend: str):
    """Test 'branch' & 'append' with 'storage'."""
    ns.clear()
    ns.set("df_x", pandas_to_polars(backend, df_input))

    pipe_name = "branch_append_with_storage"
    df_out = run_pandas_polars_pipeline(
        df_input, _YML_FILE, pipe_name, source, backend, DICT_BRANCH_PIPELINE
    )

    df_distinct = df_input.drop_duplicates()
    df_fork = df_input.copy()
    _concat_assert_clear(backend, df_distinct, df_fork, df_out, "c1", "c")


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_append_with_storage_error(df_input, source, backend: str):
    """Test 'branch' & 'append' with 'storage' but not allowed new column."""
    ns.clear()
    ns.set("df_x", pandas_to_polars(backend, df_input))

    pipe_name = "branch_append_with_storage_error"
    pipeline = get_pandas_polars_pipe(
        _YML_FILE, pipe_name, source, DICT_BRANCH_PIPELINE
    )
    pipeline.show_pipeline()

    with pytest.raises(ValueError):
        pipeline.run(df_input)

    ns.clear()


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_append_with_storage_new_col(df_input, source, backend: str):
    """Test 'branch' & 'append' with 'storage' and a new column."""
    ns.clear()
    ns.set("df_x", pandas_to_polars(backend, df_input))

    pipe_name = "branch_append_with_storage_new_col"
    df_out = run_pandas_polars_pipeline(
        df_input, _YML_FILE, pipe_name, source, backend, DICT_BRANCH_PIPELINE
    )
    df_distinct = df_input.drop_duplicates().copy()
    df_fork = df_input.copy()
    _concat_assert_clear(
        backend, df_distinct, df_fork, df_out, "new_column", "new_value"
    )


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_join_with_storage(df_input, source, backend: str):
    """Test 'branch' & 'join' with 'storage'."""
    ns.clear()

    # Recreate the dataframe, do not pass 'df_input' as it
    # will be joined with itself.
    df_secondary = df_input.copy()
    ns.set("df_x", pandas_to_polars(backend, df_secondary))

    pipe_name = "branch_join_with_storage"
    df_out = run_pandas_polars_pipeline(
        df_input, _YML_FILE, pipe_name, source, backend, DICT_BRANCH_PIPELINE
    )

    df_distinct = df_input.drop_duplicates()
    df_fork = df_secondary[["idx"]].copy()
    df_fork["new"] = -1

    df_exp = df_distinct.merge(df_fork, on="idx", how="inner")
    align_and_assert(backend, df_out, df_exp, check_dtype=False)
    ns.clear()


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_append_otherwise(df_input, source, backend: str):
    """Test 'branch' & 'append' with 'otherwise' pipeline."""
    ns.clear()

    pipe_name = "branch_append_otherwise"
    df_out = run_pandas_polars_pipeline(
        df_input, _YML_FILE, pipe_name, source, backend, DICT_BRANCH_PIPELINE
    )

    df_branch = df_input.drop_duplicates().copy()
    df_otherwise = df_branch.copy()
    df_otherwise["c1"] = "other"
    _concat_assert_clear(backend, df_otherwise, df_branch, df_out, "c1", "c")


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_join_otherwise(df_input, source, backend: str):
    """Test 'branch' & 'join' with 'otherwise' pipeline."""
    ns.clear()

    pipe_name = "branch_join_otherwise"
    df_out = run_pandas_polars_pipeline(
        df_input, _YML_FILE, pipe_name, source, backend, DICT_BRANCH_PIPELINE
    )

    df_distinct = df_input.drop_duplicates()
    df_fork = df_distinct[["idx"]].copy()
    df_fork["new"] = -1
    df_otherwise = df_distinct.copy()
    df_otherwise["other_col"] = "other"

    df_exp = df_otherwise.merge(df_fork, on="idx", how="inner")
    align_and_assert(backend, df_out, df_exp, check_dtype=False)
    ns.clear()


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_skip(df_input, source, backend: str):
    """Test 'branch' & 'skip' pipeline."""
    ns.clear()

    pipe_name = "branch_skip"
    df_out = run_pandas_polars_pipeline(
        df_input, _YML_FILE, pipe_name, source, backend, DICT_BRANCH_PIPELINE
    )

    df_exp = df_input.drop_duplicates().copy()
    align_and_assert(backend, df_out, df_exp)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_not_perform(df_input, source, backend: str):
    """Test 'branch' & 'perform' pipeline."""
    ns.clear()

    pipe_name = "branch_not_perform"
    df_out = run_pandas_polars_pipeline(
        df_input, _YML_FILE, pipe_name, source, backend, DICT_BRANCH_PIPELINE
    )

    df_exp = df_input.drop_duplicates().copy()
    align_and_assert(backend, df_out, df_exp)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_skip_otherwise(df_input, source, backend: str):
    """Test 'branch' & 'skip' with 'otherwise' pipeline."""
    ns.clear()

    pipe_name = "branch_skip_otherwise"
    df_out = run_pandas_polars_pipeline(
        df_input, _YML_FILE, pipe_name, source, backend, DICT_BRANCH_PIPELINE
    )

    df_exp = df_input.drop_duplicates().copy()
    df_exp["c1"] = "other"
    align_and_assert(backend, df_out, df_exp)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("source", _SOURCES)
def test_branch_not_perform_otherwise(df_input, source, backend: str):
    """Test 'branch' & 'perform' with 'otherwise' pipeline."""
    ns.clear()

    pipe_name = "branch_not_perform_otherwise"
    df_out = run_pandas_polars_pipeline(
        df_input, _YML_FILE, pipe_name, source, backend, DICT_BRANCH_PIPELINE
    )

    df_exp = df_input.drop_duplicates().copy()
    df_exp["c1"] = "other"
    align_and_assert(backend, df_out, df_exp)
