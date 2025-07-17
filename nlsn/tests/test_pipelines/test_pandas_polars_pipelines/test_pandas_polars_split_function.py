"""Test Pandas and Polars split functions."""

import pandas as pd
import polars as pl
import pytest

from nlsn.nebula.pipelines._pandas_split_functions import pandas_split_function
from nlsn.nebula.pipelines._polars_split_functions import polars_split_function
from nlsn.tests.auxiliaries import pandas_to_polars

_BACKENDS = ["pandas", "polars"]

_SPLIT_FUNCS = {
    "pandas": pandas_split_function,
    "polars": polars_split_function,
}


class TestPandasPolarsSplitFunctions:
    @staticmethod
    def _test(data, backend, func_args, n_true, n_false):
        df = pd.DataFrame({"c1": data})
        df = pandas_to_polars(backend, df)

        func = _SPLIT_FUNCS[backend]

        df_true, df_false = func(df, "c1", *func_args)
        assert df_true.shape[0], n_true
        assert df_false.shape[0], n_false

    @pytest.mark.parametrize("backend", _BACKENDS)
    @pytest.mark.parametrize(
        "operator, n_true, n_false", [("isNull", 1, 4), ("isNotNull", 4, 1)]
    )
    def test_null_values(self, backend: str, operator, n_true, n_false):
        """Null values function."""
        data = [1.1, 2.2, None, 3.3, 4.4]
        args = [operator, None, None]
        self._test(data, backend, args, n_true, n_false)

    @pytest.mark.parametrize("backend", _BACKENDS)
    @pytest.mark.parametrize(
        "operator, value, n_true, n_false", [("gt", 30, 2, 3), ("isin", [10, 30], 2, 3)]
    )
    def test_comparison_operator(self, backend: str, operator, value, n_true, n_false):
        """Comparison operator."""
        data = [10, 20, 30, 40, 50]
        args = [operator, value, None]
        self._test(data, backend, args, n_true, n_false)

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_comparison_operator_against_column(self, backend):
        """Comparison operator used again another column."""
        data = {
            "c1": [10, 20, 30, 40, 50],
            "c2": [25, 25, 25, 25, 25],
        }
        df = pd.DataFrame(data)
        df = pandas_to_polars(backend, df)

        func = _SPLIT_FUNCS[backend]

        df_true, df_false = func(df, "c1", "gt", None, "c2")
        assert df_true.shape[0], 3
        assert df_false.shape[0], 2

    @pytest.mark.parametrize(
        "operator, n_true", [("isNull", 1), ("isNaN", 1), ("isNullOrNaN", 2)]
    )
    def test_polars_null_nan_operator(self, operator: str, n_true: int):
        """Unit-test specific for Polars Dataframes, where nan != null."""
        data = {"c1": [10, None, float("nan"), 40]}
        df = pl.DataFrame(data)
        n_tot = df.shape[0]
        n_valid = n_tot - n_true

        df_true, df_false = polars_split_function(df, "c1", operator, None, None)
        assert df_true.shape[0], n_true
        assert df_false.shape[0], n_valid

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_contains_operator(self, backend: str):
        """Contains operator."""
        data = ["hello", "world", "lovely"]
        args = ["contains", "lo", None]
        self._test(data, backend, args, 2, 1)

    @pytest.mark.parametrize("backend", _BACKENDS)
    @pytest.mark.parametrize("value, cmp_col", [(None, "a"), (1, None)])
    def test_invalid_null_operator(self, backend: str, value, cmp_col):
        """Value with null operator."""
        func = _SPLIT_FUNCS[backend]
        df = pd.DataFrame({"c1": list(range(5))})
        df = pandas_to_polars(backend, df)
        with pytest.raises(AssertionError):
            func(df, "c1", "isNull", value, cmp_col)

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_missing_value(self, backend: str):
        """Missing value when needed."""
        func = _SPLIT_FUNCS[backend]
        df = pd.DataFrame({"c1": list(range(5))})
        df = pandas_to_polars(backend, df)
        with pytest.raises(AssertionError):
            func(df, "c1", "startswith", None, None)
