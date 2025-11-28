"""Unit-test for AssertNotEmpty."""

import pandas as pd
import polars as pl
import pytest

from nlsn.nebula.transformers import *
from nlsn.tests.auxiliaries import from_pandas
from nlsn.tests.constants import TEST_BACKENDS


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

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    @pytest.mark.parametrize(
        "expected, min_count, max_count",
        [(5, None, None), (None, 2, None), (None, None, 10), (None, 2, 10)]
    )
    def test_valid(self, spark, backend, expected, min_count, max_count):
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': ['x', 'y', 'z', 'w', 'v']
        })
        df = from_pandas(df, backend, to_nw=False, spark=spark)
        t = AssertCount(expected=expected, min_count=min_count, max_count=max_count)
        result = t.transform(df)
        assert result is df

    @pytest.mark.parametrize("to_nw", [True, False])
    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    @pytest.mark.parametrize(
        "expected, min_count, max_count",
        [(0, None, None), (None, 0, None), (None, None, 5), (None, 0, 5)]
    )
    def test_valid_empty_df(self, backend, to_nw, expected, min_count, max_count):
        df = pd.DataFrame({'a': [], 'b': []})
        df = from_pandas(df, backend, to_nw=to_nw)
        t = AssertCount(expected=expected, min_count=min_count, max_count=max_count)
        result = t.transform(df)
        if not to_nw:
            assert result is df

    @pytest.mark.parametrize(
        "expected, min_count, max_count",
        [(2, None, None), (None, 1, None), (None, None, 5), (None, 2, 5)]
    )
    def test_valid_polars_lazy(self, expected, min_count, max_count):
        df = pl.LazyFrame({'a': [1, 2], 'b': ['x', 'y']})
        t = AssertCount(expected=expected, min_count=min_count, max_count=max_count)
        result = t.transform(df)
        assert result is df

    @pytest.mark.parametrize(
        "expected, min_count, max_count",
        [(0, None, None), (None, 0, None), (None, None, 5), (None, 0, 5)]
    )
    def test_valid_empty_df_polars_lazy(self, expected, min_count, max_count):
        df = pl.LazyFrame({'a': [], 'b': []})
        t = AssertCount(expected=expected, min_count=min_count, max_count=max_count)
        result = t.transform(df)
        assert result is df

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    @pytest.mark.parametrize(
        "expected, min_count, max_count",
        [(5, None, None), (None, 3, None), (None, None, 1)]
    )
    def test_invalid(self, spark, backend, expected, min_count, max_count):
        df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
        df = from_pandas(df, backend, to_nw=False, spark=spark)
        t = AssertCount(expected=expected, min_count=min_count, max_count=max_count)
        with pytest.raises(AssertionError):
            t.transform(df)


class TestAssertNotEmpty:
    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    def test_not_empty(self, spark, backend: str):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df = from_pandas(df, backend, to_nw=False, spark=spark)
        t = AssertNotEmpty()
        t.transform(df)

    @pytest.mark.parametrize("lazy", [True, False])
    def test_not_empty_polars_lazy(self, lazy: bool):
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        if lazy:
            df = df.lazy()
        t = AssertNotEmpty()
        t.transform(df)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_empty(self, backend: str):
        df = pd.DataFrame({"a": [], "b": []}).astype("int32")
        df = from_pandas(df, backend, to_nw=False)
        t = AssertNotEmpty()
        with pytest.raises(AssertionError):
            t.transform(df)

    def test_spark_empty(self, spark):
        df = spark.createDataFrame([], schema="a: int, b: int")
        t = AssertNotEmpty()
        with pytest.raises(AssertionError):
            t.transform(df)

    @pytest.mark.parametrize("lazy", [True, False])
    def test_empty_polars_lazy(self, lazy: bool):
        df = pl.DataFrame({})
        if lazy:
            df = df.lazy()
        t = AssertNotEmpty()
        with pytest.raises(AssertionError):
            t.transform(df)
