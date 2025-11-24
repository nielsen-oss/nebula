"""Unit tests for Narwhals meta-transformers."""

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from nlsn.nebula.transformers.meta import *
from nlsn.tests.auxiliaries import from_pandas, to_pandas, sort_reset_assert
from nlsn.tests.constants import TEST_BACKENDS


class TestDataFrameMethod:
    """Test suite for DataFrameMethod transformer."""

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    def test_unique_method(self, spark, backend: str):
        """Test calling unique() method via DataFrameMethod."""
        df_pd = pd.DataFrame({
            "c1": ["a", "a", "b", "b", "c"],
            "c2": [1, 1, 2, 2, 3],
        })

        df = from_pandas(df_pd, backend, to_nw=True, spark=spark)

        t = DataFrameMethod(method="unique")
        df_out = t.transform(df)

        result_pd = to_pandas(df_out)
        expected_pd = df_pd.drop_duplicates().reset_index(drop=True)

        sort_reset_assert(result_pd, expected_pd)

    def test_head_method_with_args(self):
        """Test calling head() method with positional args."""
        df = pd.DataFrame({
            "c1": ["a", "b", "c", "d", "e"],
            "c2": [1, 2, 3, 4, 5],
        })

        # Call head(3) via DataFrameMethod
        t = DataFrameMethod(method="head", args=[3])
        df_out = t.transform(df)

        result_pd = to_pandas(df_out)
        expected_pd = df.head(3).reset_index(drop=True)

        pd.testing.assert_frame_equal(
            result_pd.reset_index(drop=True),
            expected_pd
        )

    def test_sort_method_with_kwargs(self):
        """Test calling sort() method with keyword args."""
        df = pd.DataFrame({
            "c1": ["c", "a", "b"],
            "c2": [3, 1, 2],
        })

        # Call sort(by="c1") via DataFrameMethod
        t = DataFrameMethod(method="sort", args=["c1"])
        df_out = t.transform(df)

        result_pd = to_pandas(df_out)
        expected_pd = df.sort_values("c1").reset_index(drop=True)

        pd.testing.assert_frame_equal(
            result_pd.reset_index(drop=True),
            expected_pd
        )

    def test_invalid_method_name_init(self):
        """Test that invalid method names raise ValueError at init."""
        with pytest.raises(ValueError):
            DataFrameMethod(method="nonexistent_method")

    def test_method_not_available_for_lazyframe(self):
        """Test that DataFrame-only methods raise AttributeError on LazyFrame."""
        df_lazy = pl.LazyFrame({
            "c1": ["a", "b", "c"],
            "c2": [1, 2, 3],
        })
        df_nw = nw.from_native(df_lazy)

        # Try to call a DataFrame-only method (e.g., to_dict)
        # Note: We need to find a method that exists in DataFrame but not LazyFrame
        # For this test, we'll assume 'to_dict' or similar exists
        # If not available, adjust to an actual DataFrame-only method

        # Check if there's actually a difference in methods
        df_methods = {i for i in dir(nw.DataFrame) if i.islower() and i[0] != "_"}
        lazy_methods = {i for i in dir(nw.LazyFrame) if i.islower() and i[0] != "_"}
        df_only = df_methods - lazy_methods

        if not df_only:
            pytest.skip("No DataFrame-only methods found in current Narwhals version")

        # Use the first DataFrame-only method
        df_only_method = sorted(df_only)[0]

        t = DataFrameMethod(method=df_only_method)

        with pytest.raises(AttributeError):
            t.transform(df_nw)

    def test_method_with_args_and_kwargs(self):
        """Test calling unique() method with both args and kwargs."""
        df = pd.DataFrame({
            "c1": ["a", "b", "c", "d", "e", "f"],
            "c2": [1, 2, 3, 4, 5, 6],
        })

        t = DataFrameMethod(
            method="sort",
            args=["c1"],
            kwargs={"descending": True}
        )
        df_out = t.transform(df)

        result_pd = to_pandas(df_out)
        expected_pd = df.sort_values("c1", ascending=False)

        pd.testing.assert_frame_equal(
            result_pd.reset_index(drop=True),
            expected_pd.reset_index(drop=True)
        )
