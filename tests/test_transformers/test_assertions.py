"""Unit-tests for 'assertions' transformers."""

import pandas as pd
import polars as pl
import pytest

from nebula.transformers import *

from ..auxiliaries import from_pandas
from ..constants import TEST_BACKENDS


class TestAssertContainsColumns:
    @pytest.mark.parametrize(
        "cols, error",
        [
            ([], False),
            ("a", False),
            (["a"], False),
            (["a", "b"], False),
            (["a", "b", "c"], True),
            ("c", True),
            (["c"], True),
        ],
    )
    def test_dataframe_contains_columns(self, cols, error: bool):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        t = AssertContainsColumns(columns=cols)
        if error:
            with pytest.raises(AssertionError):
                t.transform(df)
        else:
            t.transform(df)


class TestAssertCount:
    @pytest.mark.parametrize("min_count, max_count", [(10, None), (None, 10), (10, 10)])
    def test_invalid_init_with_expected(self, min_count, max_count):
        with pytest.raises(AssertionError):
            AssertCount(expected=100, min_count=min_count, max_count=max_count)

    def test_invalid_init(self):
        with pytest.raises(AssertionError):
            AssertCount(min_count=10, max_count=5)

    @pytest.mark.parametrize("backend", [*TEST_BACKENDS, "polars_lazy"])
    @pytest.mark.parametrize(
        "expected, min_count, max_count",
        [(5, None, None), (None, 2, None), (None, None, 10), (None, 2, 10)],
    )
    def test_valid(self, spark, duckdb_con, backend, expected, min_count, max_count):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "w", "v"]})
        if backend == "polars_lazy":
            df = pl.from_pandas(df).lazy()
        else:
            df = from_pandas(df, backend, to_nw=False, spark=spark, duckdb_con=duckdb_con)
        t = AssertCount(expected=expected, min_count=min_count, max_count=max_count)
        result = t.transform(df)
        assert result is df

    @pytest.mark.parametrize("to_nw", [True, False])
    @pytest.mark.parametrize("backend", [*TEST_BACKENDS, "polars_lazy"])
    @pytest.mark.parametrize(
        "expected, min_count, max_count",
        [(0, None, None), (None, 0, None), (None, None, 5), (None, 0, 5)],
    )
    def test_valid_empty_df(self, backend, to_nw, expected, min_count, max_count):
        if backend == "polars_lazy":
            df = pl.LazyFrame({"a": [], "b": []})
        else:
            df = pd.DataFrame({"a": [], "b": []})
            df = from_pandas(df, backend, to_nw=to_nw)
        t = AssertCount(expected=expected, min_count=min_count, max_count=max_count)
        result = t.transform(df)
        if backend != "polars_lazy" and not to_nw:
            assert result is df

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    @pytest.mark.parametrize(
        "expected, min_count, max_count",
        [(5, None, None), (None, 3, None), (None, None, 1)],
    )
    def test_invalid(self, spark, duckdb_con, backend, expected, min_count, max_count):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        df = from_pandas(df, backend, to_nw=False, spark=spark, duckdb_con=duckdb_con)
        t = AssertCount(expected=expected, min_count=min_count, max_count=max_count)
        with pytest.raises(AssertionError):
            t.transform(df)


@pytest.mark.parametrize("to_nw", [True, False])
class TestAssertNotEmpty:
    """Test 'AssertNotEmpty' transformer, Complete tests in 'TestDfIsEmpty'."""

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    def test_not_empty(self, duckdb_con, backend: str, to_nw: bool):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df = from_pandas(df, backend, to_nw=to_nw, duckdb_con=duckdb_con)
        t = AssertNotEmpty()
        t.transform(df)

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    def test_empty(self, duckdb_con, backend: str, to_nw: bool):
        df = pd.DataFrame({"a": [], "b": []})
        df = from_pandas(df, backend, to_nw=to_nw, duckdb_con=duckdb_con)
        t = AssertNotEmpty()
        with pytest.raises(AssertionError):
            t.transform(df)
