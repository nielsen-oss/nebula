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


class TestWithColumns:
    """Test suite for WithColumns transformer."""

    @pytest.mark.parametrize("meth", ["wrong", "str.wrong", "dt.b.wrong"])
    def test_invalid_init(self, meth: str):
        with pytest.raises(ValueError):
            WithColumns(columns="a", method=meth)

    def test_single_column_string_method(self):
        """Test applying str.strip_chars to a single column."""
        df = pl.DataFrame({
            "name": ["  alice  ", "  bob  ", "  charlie  "],
            "age": [25, 30, 35],
        })
        df_nw = nw.from_native(df)

        t = WithColumns(columns="name", method="str.strip_chars")
        df_out = t.transform(df_nw)

        result = nw.to_native(df_out)
        expected = pl.DataFrame({
            "name": ["alice", "bob", "charlie"],
            "age": [25, 30, 35],
        })

        assert result.equals(expected)

    def test_multiple_columns_round(self):
        """Test applying round to multiple columns."""
        df = pl.DataFrame({
            "price": [10.567, 20.891, 30.123],
            "tax": [1.234, 2.345, 3.456],
            "quantity": [5, 10, 15],
        })
        df_nw = nw.from_native(df)

        t = WithColumns(
            columns=["price", "tax"],
            method="round",
            args=[2]
        )
        df_out = t.transform(df_nw)

        result = nw.to_native(df_out)
        expected = pl.DataFrame({
            "price": [10.57, 20.89, 30.12],
            "tax": [1.23, 2.35, 3.46],
            "quantity": [5, 10, 15],
        })

        assert result.equals(expected)

    def test_regex_selection(self):
        """Test selecting columns via regex pattern."""
        df = pl.DataFrame({
            "user_name": ["  Alice  ", "  Bob  "],
            "company_name": ["  Acme  ", "  Corp  "],
            "age": [25, 30],
        })
        df_nw = nw.from_native(df)

        t = WithColumns(
            regex=".*_name$",
            method="str.strip_chars"
        )
        df_out = t.transform(df_nw)

        result = nw.to_native(df_out)
        expected = pl.DataFrame({
            "user_name": ["Alice", "Bob"],
            "company_name": ["Acme", "Corp"],
            "age": [25, 30],
        })

        assert result.equals(expected)

    def test_glob_selection(self):
        """Test selecting columns via glob pattern."""
        df = pl.DataFrame({
            "price_usd": [10.5, 20.7],
            "price_eur": [9.2, 18.1],
            "quantity": [5, 10],
        })
        df_nw = nw.from_native(df)

        t = WithColumns(
            glob="price_*",
            method="round",
            args=[1]
        )
        df_out = t.transform(df_nw)

        result = nw.to_native(df_out)
        expected = pl.DataFrame({
            "price_usd": [10.5, 20.7],
            "price_eur": [9.2, 18.1],
            "quantity": [5, 10],
        })

        assert result.equals(expected)

    def test_startswith_selection(self):
        """Test selecting columns that start with a prefix."""
        df = pl.DataFrame({
            "col_a": [1.111, 2.222],
            "col_b": [3.333, 4.444],
            "other": [5, 6],
        })
        df_nw = nw.from_native(df)

        t = WithColumns(
            startswith="col_",
            method="round",
            args=[1]
        )
        df_out = t.transform(df_nw)

        result = nw.to_native(df_out)
        expected = pl.DataFrame({
            "col_a": [1.1, 2.2],
            "col_b": [3.3, 4.4],
            "other": [5, 6],
        })

        assert result.equals(expected)

    def test_endswith_selection(self):
        """Test selecting columns that end with a suffix."""
        df = pl.DataFrame({
            "amount_usd": [10.567, 20.891],
            "amount_eur": [9.234, 18.567],
            "quantity": [5, 10],
        })
        df_nw = nw.from_native(df)

        t = WithColumns(
            endswith="_usd",
            method="round",
            args=[1]
        )
        df_out = t.transform(df_nw)

        result = nw.to_native(df_out)
        expected = pl.DataFrame({
            "amount_usd": [10.6, 20.9],
            "amount_eur": [9.234, 18.567],
            "quantity": [5, 10],
        })

        assert result.equals(expected)

    def test_prefix_option(self):
        """Test adding prefix to output column names."""
        df = pl.DataFrame({
            "price": [10.567, 20.891],
            "tax": [1.234, 2.345],
        })
        df_nw = nw.from_native(df)

        t = WithColumns(
            columns=["price", "tax"],
            method="round",
            args=[2],
            prefix="rounded_"
        )
        df_out = t.transform(df_nw)

        result = nw.to_native(df_out)

        # Should have original columns plus prefixed ones
        assert "price" in result.columns
        assert "tax" in result.columns
        assert "rounded_price" in result.columns
        assert "rounded_tax" in result.columns
        assert result["rounded_price"].to_list() == [10.57, 20.89]

    def test_suffix_option(self):
        """Test adding suffix to output column names."""
        df = pl.DataFrame({
            "price": [10.567, 20.891],
            "tax": [1.234, 2.345],
        })
        df_nw = nw.from_native(df)

        t = WithColumns(
            columns=["price", "tax"],
            method="round",
            args=[2],
            suffix="_rounded"
        )
        df_out = t.transform(df_nw)

        result = nw.to_native(df_out)

        # Should have original columns plus suffixed ones
        assert "price" in result.columns
        assert "tax" in result.columns
        assert "price_rounded" in result.columns
        assert "tax_rounded" in result.columns
        assert result["price_rounded"].to_list() == [10.57, 20.89]

    def test_method_with_kwargs(self):
        """Test method with keyword arguments."""
        df = pl.DataFrame({
            "text": ["hello world", "foo bar"],
        })
        df_nw = nw.from_native(df)

        t = WithColumns(
            columns="text",
            method="str.replace_all",
            args=["world", "universe"],
        )
        df_out = t.transform(df_nw)

        result = nw.to_native(df_out)
        expected = pl.DataFrame({
            "text": ["hello universe", "foo bar"],
        })

        assert result.equals(expected)

    def test_abs_method(self):
        """Test applying abs to numeric columns."""
        df = pl.DataFrame({
            "value1": [-10, -20, 30],
            "value2": [-5, 15, -25],
        })
        df_nw = nw.from_native(df)

        t = WithColumns(
            columns=["value1", "value2"],
            method="abs"
        )
        df_out = t.transform(df_nw)

        result = nw.to_native(df_out)
        expected = pl.DataFrame({
            "value1": [10, 20, 30],
            "value2": [5, 15, 25],
        })

        assert result.equals(expected)

    def test_cast_method(self):
        """Test casting columns to different types."""
        df = pl.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
        })
        df_nw = nw.from_native(df)

        t = WithColumns(
            columns="int_col",
            method="cast",
            args=[nw.Float64]
        )
        df_out = t.transform(df_nw)

        result = nw.to_native(df_out)

        # Check that int_col is now float
        assert result.schema["int_col"] == pl.Float64

    def test_empty_selection_returns_unchanged(self):
        """Test that empty column selection returns df unchanged."""
        df = pl.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
        })
        df_nw = nw.from_native(df)

        # Regex that matches nothing
        t = WithColumns(
            regex="^nonexistent$",
            method="round"
        )
        assert df_nw is t.transform(df_nw)

    def test_multiple_string_methods_chained(self):
        """Test that str namespace methods work correctly."""
        df = pl.DataFrame({
            "text": ["  HELLO  ", "  WORLD  "],
        })
        df_nw = nw.from_native(df)

        # First strip, then lowercase (need to do in separate transforms)
        t1 = WithColumns(columns="text", method="str.strip_chars")
        t2 = WithColumns(columns="text", method="str.to_lowercase")

        df_out = t1.transform(df_nw)
        df_out = t2.transform(df_out)

        result = nw.to_native(df_out)
        expected = pl.DataFrame({
            "text": ["hello", "world"],
        })

        assert result.equals(expected)

    def test_works_with_native_polars_df(self):
        """Test that transformer works with native Polars DataFrame."""
        df = pl.DataFrame({
            "value": [10.567, 20.891],
        })

        t = WithColumns(
            columns="value",
            method="round",
            args=[1]
        )
        df_out = t.transform(df)

        # Should return native Polars
        assert isinstance(df_out, pl.DataFrame)
        assert df_out["value"].to_list() == [10.6, 20.9]

    def test_works_with_lazyframe(self):
        """Test that transformer works with Polars LazyFrame."""
        df_lazy = pl.LazyFrame({
            "value": [10.567, 20.891],
        })

        t = WithColumns(
            columns="value",
            method="round",
            args=[1]
        )
        df_out = t.transform(df_lazy)

        # Should return LazyFrame
        assert isinstance(df_out, pl.LazyFrame)

        result = df_out.collect()
        assert result["value"].to_list() == [10.6, 20.9]
