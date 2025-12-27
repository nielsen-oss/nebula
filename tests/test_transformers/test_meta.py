"""Unit tests for Narwhals meta-transformers."""

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from nebula.transformers.meta import *
from ..auxiliaries import from_pandas, to_pandas, pl_assert_equal, pd_sort_assert
from ..constants import TEST_BACKENDS


class TestDataFrameMethod:
    """Test suite for DataFrameMethod transformer."""

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    def test_unique_method(self, spark, backend: str):
        """Test calling unique() method via DataFrameMethod."""
        df_pd = pd.DataFrame(
            {
                "c1": ["a", "a", "b", "b", "c"],
                "c2": [1, 1, 2, 2, 3],
            }
        )

        df = from_pandas(df_pd, backend, to_nw=True, spark=spark)

        t = DataFrameMethod(method="unique")
        df_out = t.transform(df)

        result_pd = to_pandas(df_out)
        expected_pd = df_pd.drop_duplicates().reset_index(drop=True)

        pd_sort_assert(result_pd, expected_pd)

    def test_head_method_with_args(self):
        """Test calling head() method with positional args."""
        df = pd.DataFrame(
            {
                "c1": ["a", "b", "c", "d", "e"],
                "c2": [1, 2, 3, 4, 5],
            }
        )

        # Call head(3) via DataFrameMethod
        t = DataFrameMethod(method="head", args=[3])
        df_out = t.transform(df)

        result_pd = to_pandas(df_out)
        expected_pd = df.head(3).reset_index(drop=True)

        pd.testing.assert_frame_equal(result_pd.reset_index(drop=True), expected_pd)

    def test_sort_method_with_kwargs(self):
        """Test calling sort() method with keyword args."""
        df = pd.DataFrame(
            {
                "c1": ["c", "a", "b"],
                "c2": [3, 1, 2],
            }
        )

        # Call sort(by="c1") via DataFrameMethod
        t = DataFrameMethod(method="sort", args=["c1"])
        df_out = t.transform(df)

        result_pd = to_pandas(df_out)
        expected_pd = df.sort_values("c1").reset_index(drop=True)

        pd.testing.assert_frame_equal(result_pd.reset_index(drop=True), expected_pd)

    def test_invalid_method_name_init(self):
        """Test that invalid method names raise ValueError at init."""
        with pytest.raises(ValueError):
            DataFrameMethod(method="nonexistent_method")

    @pytest.mark.parametrize("to_lazy", [True, False])
    def test_method_not_available_for_lazyframe(self, to_lazy: bool):
        """Test that Data/LazyFrame-only methods raise an error on Lazy/DataFrame."""
        df = pl.DataFrame({"c1": ["a", "b"], "c2": [1, 2]})

        # Note: need to find a method that exists in DataFrame/LazyFrame
        # but not LazyFrame/DataFrame
        df_methods = {i for i in dir(nw.DataFrame) if i.islower() and i[0] != "_"}
        lazy_methods = {i for i in dir(nw.LazyFrame) if i.islower() and i[0] != "_"}
        if to_lazy:
            df = df.lazy()
            only = df_methods - lazy_methods
        else:
            only = lazy_methods - df_methods

        if not only:
            pytest.skip("No DataFrame-only methods found in current Narwhals version")

        # Use the first only method
        only_method = sorted(only)[0]

        t = DataFrameMethod(method=only_method)

        with pytest.raises(AttributeError):
            t.transform(df)

    def test_method_with_args_and_kwargs(self):
        """Test calling unique() method with both args and kwargs."""
        df = pd.DataFrame(
            {
                "c1": ["a", "b", "c", "d", "e", "f"],
                "c2": [1, 2, 3, 4, 5, 6],
            }
        )

        t = DataFrameMethod(method="sort", args=["c1"], kwargs={"descending": True})
        df_out = t.transform(df)

        result_pd = to_pandas(df_out)
        expected_pd = df.sort_values("c1", ascending=False)

        pd.testing.assert_frame_equal(
            result_pd.reset_index(drop=True), expected_pd.reset_index(drop=True)
        )


class TestHorizontalFunction:
    """Test suite for HorizontalFunction transformer."""

    def test_coalesce_basic(self):
        """Test basic coalesce across multiple columns."""
        df = pl.DataFrame(
            {
                "email_primary": [None, "bob@example.com", None],
                "email_secondary": ["alice@work.com", None, None],
                "email_backup": ["alice@home.com", "bob@home.com", "charlie@home.com"],
            }
        )
        df_nw = nw.from_native(df)

        t = HorizontalFunction(
            output_col="best_email",
            function="coalesce",
            columns=["email_primary", "email_secondary", "email_backup"],
        )
        df_out = t.transform(df_nw)

        result = nw.to_native(df_out)
        expected = pl.DataFrame(
            {
                "email_primary": [None, "bob@example.com", None],
                "email_secondary": ["alice@work.com", None, None],
                "email_backup": ["alice@home.com", "bob@home.com", "charlie@home.com"],
                "best_email": ["alice@work.com", "bob@example.com", "charlie@home.com"],
            }
        )

        assert result.equals(expected)

    def test_max_horizontal(self):
        """Test max_horizontal across numeric columns."""
        df = pl.DataFrame(
            {
                "score_math": [85, 90, 78],
                "score_english": [92, 88, 95],
                "score_science": [88, 85, None],
            }
        )
        df_nw = nw.from_native(df)

        t = HorizontalFunction(
            output_col="max_score",
            function="max_horizontal",
            columns=["score_math", "score_english", "score_science"],
        )
        df_out = t.transform(df_nw)

        result = nw.to_native(df_out)

        assert "max_score" in result.columns
        assert result["max_score"].to_list() == [92, 90, 95]

    def test_concat_str_with_separator(self):
        """Test concat_str with separator and ignore_nulls."""
        df = pl.DataFrame(
            {
                "first_name": ["Alice", "Bob"],
                "middle_name": ["M", None],
                "last_name": ["Smith", "Jones"],
            }
        )
        df_nw = nw.from_native(df)

        t = HorizontalFunction(
            output_col="full_name",
            function="concat_str",
            columns=["first_name", "middle_name", "last_name"],
            kwargs={"separator": " ", "ignore_nulls": True},
        )
        df_out = t.transform(df_nw)

        result = nw.to_native(df_out)

        assert "full_name" in result.columns
        assert result["full_name"].to_list() == ["Alice M Smith", "Bob Jones"]

    def test_sum_horizontal_with_regex(self):
        """Test sum_horizontal using regex pattern selection."""
        df = pl.DataFrame(
            {
                "revenue_q1": [100, 200],
                "revenue_q2": [150, 250],
                "revenue_q3": [120, 180],
                "cost": [50, 60],
            }
        )
        df_nw = nw.from_native(df)

        t = HorizontalFunction(
            output_col="total_revenue", function="sum_horizontal", regex="^revenue_.*"
        )
        df_out = t.transform(df_nw)

        result = nw.to_native(df_out)

        assert "total_revenue" in result.columns
        assert result["total_revenue"].to_list() == [370, 630]
        # Cost column should not be included
        assert "cost" in result.columns

    def test_mean_horizontal_with_glob(self):
        """Test mean_horizontal using glob pattern selection."""
        df = pl.DataFrame(
            {
                "temp_morning": [20.0, 22.0, 19.0],
                "temp_afternoon": [28.0, 30.0, 27.0],
                "temp_evening": [24.0, 26.0, 23.0],
                "humidity": [65.0, 70.0, 60.0],
            }
        )
        df_nw = nw.from_native(df)

        t = HorizontalFunction(
            output_col="avg_temp", function="mean_horizontal", glob="temp_*"
        )
        df_out = t.transform(df_nw)

        result = nw.to_native(df_out)

        assert "avg_temp" in result.columns
        # Mean of (20, 28, 24) = 24.0, etc.
        assert result["avg_temp"].to_list() == [24.0, 26.0, 23.0]

    def test_max_horizontal_with_no_selection(self):
        """Test max_horizontal without actually selecting a column."""
        df = pl.DataFrame(
            {
                "score_math": [85, 90, 78],
                "score_english": [92, 88, 95],
                "score_max": [0, 0, 0],
            }
        )
        t = HorizontalFunction(
            output_col="score_max", function="max_horizontal", startswith="invalid_"
        )
        df_out = t.transform(df)
        pl_assert_equal(df, df_out)


class TestWithColumns:
    """Test suite for WithColumns transformer."""

    @pytest.mark.parametrize("meth", ["wrong", "str.wrong", "dt.b.wrong"])
    def test_invalid_init(self, meth: str):
        with pytest.raises(ValueError):
            WithColumns(columns="a", method=meth)

    @pytest.mark.parametrize(
        "prefix, suffix", [("xx", None), (None, "xx"), ("xx", "xx")]
    )
    def test_invalid_alias(self, prefix, suffix):
        with pytest.raises(AssertionError):
            WithColumns(
                columns="a", method="round", prefix=prefix, suffix=suffix, alias="alias"
            )

    def test_single_column_string_method(self):
        """Test applying str.strip_chars to a single column."""
        df = pl.DataFrame(
            {
                "name": ["  alice  ", "  bob  ", "  charlie  "],
                "age": [25, 30, 35],
            }
        )
        df_nw = nw.from_native(df)

        t = WithColumns(columns="name", method="str.strip_chars")
        df_out = t.transform(df_nw)

        result = nw.to_native(df_out)
        expected = pl.DataFrame(
            {
                "name": ["alice", "bob", "charlie"],
                "age": [25, 30, 35],
            }
        )

        assert result.equals(expected)

    def test_multiple_columns_round(self):
        """Test applying round to multiple columns."""
        df = pl.DataFrame(
            {
                "price": [10.567, 20.891, 30.123],
                "tax": [1.234, 2.345, 3.456],
                "quantity": [5, 10, 15],
            }
        )
        df_nw = nw.from_native(df)

        t = WithColumns(columns=["price", "tax"], method="round", args=[2])
        df_out = t.transform(df_nw)

        result = nw.to_native(df_out)
        expected = pl.DataFrame(
            {
                "price": [10.57, 20.89, 30.12],
                "tax": [1.23, 2.35, 3.46],
                "quantity": [5, 10, 15],
            }
        )

        assert result.equals(expected)

    def test_regex_selection(self):
        """Test selecting columns via regex pattern."""
        df = pl.DataFrame(
            {
                "user_name": ["  Alice  ", "  Bob  "],
                "company_name": ["  Acme  ", "  Corp  "],
                "age": [25, 30],
            }
        )
        df_nw = nw.from_native(df)

        t = WithColumns(regex=".*_name$", method="str.strip_chars")
        df_out = t.transform(df_nw)

        result = nw.to_native(df_out)
        expected = pl.DataFrame(
            {
                "user_name": ["Alice", "Bob"],
                "company_name": ["Acme", "Corp"],
                "age": [25, 30],
            }
        )

        assert result.equals(expected)

    def test_glob_selection(self):
        """Test selecting columns via glob pattern."""
        df = pl.DataFrame(
            {
                "price_usd": [10.5, 20.7],
                "price_eur": [9.2, 18.1],
                "quantity": [5, 10],
            }
        )
        df_nw = nw.from_native(df)

        t = WithColumns(glob="price_*", method="round", args=[1])
        df_out = t.transform(df_nw)

        result = nw.to_native(df_out)
        expected = pl.DataFrame(
            {
                "price_usd": [10.5, 20.7],
                "price_eur": [9.2, 18.1],
                "quantity": [5, 10],
            }
        )

        assert result.equals(expected)

    def test_startswith_selection(self):
        """Test selecting columns that start with a prefix."""
        df = pl.DataFrame(
            {
                "col_a": [1.111, 2.222],
                "col_b": [3.333, 4.444],
                "other": [5, 6],
            }
        )
        df_nw = nw.from_native(df)

        t = WithColumns(startswith="col_", method="round", args=[1])
        df_out = t.transform(df_nw)

        result = nw.to_native(df_out)
        expected = pl.DataFrame(
            {
                "col_a": [1.1, 2.2],
                "col_b": [3.3, 4.4],
                "other": [5, 6],
            }
        )

        assert result.equals(expected)

    def test_endswith_selection(self):
        """Test selecting columns that end with a suffix."""
        df = pl.DataFrame(
            {
                "amount_usd": [10.567, 20.891],
                "amount_eur": [9.234, 18.567],
                "quantity": [5, 10],
            }
        )
        df_nw = nw.from_native(df)

        t = WithColumns(endswith="_usd", method="round", args=[1])
        df_out = t.transform(df_nw)

        result = nw.to_native(df_out)
        expected = pl.DataFrame(
            {
                "amount_usd": [10.6, 20.9],
                "amount_eur": [9.234, 18.567],
                "quantity": [5, 10],
            }
        )

        assert result.equals(expected)

    def test_prefix_option(self):
        """Test adding prefix to output column names."""
        df = pl.DataFrame(
            {
                "price": [10.567, 20.891],
                "tax": [1.234, 2.345],
            }
        )
        df_nw = nw.from_native(df)

        t = WithColumns(
            columns=["price", "tax"], method="round", args=[2], prefix="rounded_"
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
        df = pl.DataFrame(
            {
                "price": [10.567, 20.891],
                "tax": [1.234, 2.345],
            }
        )
        df_nw = nw.from_native(df)

        t = WithColumns(
            columns=["price", "tax"], method="round", args=[2], suffix="_rounded"
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
        df = pl.DataFrame(
            {
                "text": ["hello world", "foo bar"],
            }
        )
        df_nw = nw.from_native(df)

        t = WithColumns(
            columns="text",
            method="str.replace_all",
            args=["world", "universe"],
        )
        df_out = t.transform(df_nw)

        result = nw.to_native(df_out)
        expected = pl.DataFrame(
            {
                "text": ["hello universe", "foo bar"],
            }
        )

        assert result.equals(expected)

    def test_abs_method(self):
        """Test applying abs to numeric columns."""
        df = pl.DataFrame(
            {
                "value1": [-10, -20, 30],
                "value2": [-5, 15, -25],
            }
        )
        df_nw = nw.from_native(df)

        t = WithColumns(columns=["value1", "value2"], method="abs")
        df_out = t.transform(df_nw)

        result = nw.to_native(df_out)
        expected = pl.DataFrame(
            {
                "value1": [10, 20, 30],
                "value2": [5, 15, 25],
            }
        )

        assert result.equals(expected)

    def test_cast_method(self):
        """Test casting columns to different types."""
        df = pl.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.5, 2.5, 3.5],
            }
        )
        df_nw = nw.from_native(df)

        t = WithColumns(
            columns="int_col", method="cast", args=[nw.Float64], alias="new_col"
        )
        df_out = t.transform(df_nw)

        result = nw.to_native(df_out)

        expected = df.with_columns(pl.col("int_col").cast(pl.Float64).alias("new_col"))
        assert result.equals(expected)

        # Check that int_col is still Int64
        assert df_out.schema["int_col"] == nw.Int64

    def test_empty_selection_returns_unchanged(self):
        """Test that empty column selection returns df unchanged."""
        df = pl.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": [4, 5, 6],
            }
        )
        df_nw = nw.from_native(df)

        # Regex that matches nothing
        t = WithColumns(regex="^nonexistent$", method="round")
        assert df_nw is t.transform(df_nw)

    def test_multiple_string_methods_chained(self):
        """Test that str namespace methods work correctly."""
        df = pl.DataFrame(
            {
                "text": ["  HELLO  ", "  WORLD  "],
            }
        )
        df_nw = nw.from_native(df)

        # First strip, then lowercase (need to do in separate transforms)
        t1 = WithColumns(columns="text", method="str.strip_chars")
        t2 = WithColumns(columns="text", method="str.to_lowercase")

        df_out = t1.transform(df_nw)
        df_out = t2.transform(df_out)

        result = nw.to_native(df_out)
        expected = pl.DataFrame(
            {
                "text": ["hello", "world"],
            }
        )

        assert result.equals(expected)

    def test_works_with_native_polars_df(self):
        """Test that transformer works with native Polars DataFrame."""
        df = pl.DataFrame(
            {
                "value": [10.567, 20.891],
            }
        )

        t = WithColumns(columns="value", method="round", args=[1])
        df_out = t.transform(df)

        # Should return native Polars
        assert isinstance(df_out, pl.DataFrame)
        assert df_out["value"].to_list() == [10.6, 20.9]

    def test_works_with_lazyframe(self):
        """Test that transformer works with Polars LazyFrame."""
        df_lazy = pl.LazyFrame(
            {
                "value": [10.567, 20.891],
            }
        )

        t = WithColumns(columns="value", method="round", args=[1])
        df_out = t.transform(df_lazy)

        # Should return LazyFrame
        assert isinstance(df_out, pl.LazyFrame)

        result = df_out.collect()
        assert result["value"].to_list() == [10.6, 20.9]
