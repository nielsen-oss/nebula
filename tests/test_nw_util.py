"""Unit-tests for Narwhals utils."""

import os

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from nebula.nw_util import *
from nebula.nw_util import COMPARISON_OPERATORS, NULL_OPERATORS, assert_join_params

from .auxiliaries import from_pandas, to_pandas
from .constants import TEST_BACKENDS


class TestAssertJoinParams:
    def test_cross_join_with_no_keys_allowed(self):
        # should not raise
        assert_join_params(
            how="cross",
            on=None,
            left_on=None,
            right_on=None,
        )

    def test_cross_join_with_on_raises(self):
        with pytest.raises(ValueError, match="cross join"):
            assert_join_params(
                how="cross",
                on="id",
                left_on=None,
                right_on=None,
            )

    def test_cross_join_with_left_on_raises(self):
        with pytest.raises(ValueError, match="cross join"):
            assert_join_params(
                how="cross",
                on=None,
                left_on="id",
                right_on=None,
            )

    def test_cross_join_with_right_on_raises(self):
        with pytest.raises(ValueError, match="cross join"):
            assert_join_params(
                how="cross",
                on=None,
                left_on=None,
                right_on="id",
            )

    def test_on_with_left_on_raises(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            assert_join_params(
                how="inner",
                on="id",
                left_on="id",
                right_on="id",
            )

    def test_on_with_right_on_raises(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            assert_join_params(
                how="inner",
                on="id",
                left_on=None,
                right_on="id",
            )

    def test_left_on_without_right_on_raises(self):
        with pytest.raises(ValueError, match="Must specify both"):
            assert_join_params(
                how="inner",
                on=None,
                left_on="id",
                right_on=None,
            )

    def test_right_on_without_left_on_raises(self):
        with pytest.raises(ValueError, match="Must specify both"):
            assert_join_params(
                how="inner",
                on=None,
                left_on=None,
                right_on="id",
            )

    def test_on_only_is_valid(self):
        # should not raise
        assert_join_params(
            how="inner",
            on="id",
            left_on=None,
            right_on=None,
        )

    def test_left_and_right_on_is_valid(self):
        # should not raise
        assert_join_params(
            how="inner",
            on=None,
            left_on="left_id",
            right_on="right_id",
        )

    def test_no_keys_is_valid_for_non_cross(self):
        # should not raise
        assert_join_params(
            how="left",
            on=None,
            left_on=None,
            right_on=None,
        )


class TestAppendDataframes:
    """Test suite for the 'append_dataframes' utility function."""

    @pytest.mark.parametrize("to_nw", [True, False])
    def test_single_dataframe(self, to_nw: bool):
        """Test that a single dataframe is returned unchanged."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = append_dataframes([df], allow_missing_cols=False)
        assert result is df

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    @pytest.mark.parametrize("to_nw", [True, False])
    def test_exact_columns(self, spark, backend: str, to_nw: bool):
        """Test concatenation when all dataframes have identical columns."""
        df1_pd = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2_pd = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
        df3_pd = pd.DataFrame({"a": [9, 10], "b": [11, 12]})

        df1 = from_pandas(df1_pd, backend, to_nw=to_nw, spark=spark)
        df2 = from_pandas(df2_pd, backend, to_nw=to_nw, spark=spark)
        df3 = from_pandas(df3_pd, backend, to_nw=to_nw, spark=spark)

        result = append_dataframes([df1, df2, df3], allow_missing_cols=False)
        result_pd = to_pandas(result).reset_index(drop=True)

        expected = pd.concat([df1_pd, df2_pd, df3_pd], axis=0).reset_index(drop=True)
        pd.testing.assert_frame_equal(result_pd, expected)

    def test_missing_columns_disallowed(self):
        """Test that column mismatch raises error when not allowed."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [5, 6], "c": [7, 8]})  # 'c' instead of 'b'
        with pytest.raises(ValueError):
            append_dataframes([df1, df2], allow_missing_cols=False)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    @pytest.mark.parametrize("to_nw", [True, False])
    def test_missing_columns_allowed(self, backend: str, to_nw: bool):
        """Test concatenation with missing columns fills with nulls."""
        df1_pd = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2_pd = pd.DataFrame({"a": [5, 6], "c": [7.0, 8.0]})  # 'c' instead of 'b'

        df1 = from_pandas(df1_pd, backend, to_nw=to_nw)
        df2 = from_pandas(df2_pd, backend, to_nw=to_nw)

        result = append_dataframes([df1, df2], allow_missing_cols=True)

        is_nw = isinstance(result, (nw.DataFrame, nw.LazyFrame))
        assert to_nw == is_nw

        result_pd = to_pandas(result).reset_index(drop=True)

        # Expected: all columns present, missing values filled with NaN/None
        expected = pd.concat([df1_pd, df2_pd], axis=0).reset_index(drop=True)
        pd.testing.assert_frame_equal(result_pd, expected)

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    def test_mix_narwhals_and_native(self, spark, backend: str):
        """Test mixing Narwhals wrappers with native dataframes (same backend)."""
        df1_pd = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2_pd = pd.DataFrame({"a": [5, 6], "b": [7, 8]})

        # First as Narwhals, second as native
        df1_nw = from_pandas(df1_pd, backend, to_nw=True, spark=spark)
        df2_native = from_pandas(df2_pd, backend, to_nw=False, spark=spark)

        result = append_dataframes([df1_nw, df2_native], allow_missing_cols=True)

        assert isinstance(result, (nw.DataFrame, nw.LazyFrame))

        result_pd = to_pandas(result).reset_index(drop=True)
        expected = pd.concat([df1_pd, df2_pd], axis=0).reset_index(drop=True)
        pd.testing.assert_frame_equal(result_pd, expected)

    @pytest.mark.parametrize("ignore_index", [True, False])
    def test_pandas_ignore_index(self, ignore_index: bool):
        """Test pandas ignore_index parameter."""
        df1_pd = pd.DataFrame({"a": [1, 2]}, index=[10, 20])
        df2_pd = pd.DataFrame({"a": [3, 4]}, index=[30, 40])

        result = append_dataframes([df1_pd, df2_pd], allow_missing_cols=False, ignore_index=ignore_index)

        if ignore_index:
            # Index should be reset to 0, 1, 2, 3
            assert list(result.index) == [0, 1, 2, 3]
        else:
            # Original indices preserved
            assert list(result.index) == [10, 20, 30, 40]

    def test_polars_relax_parameter(self):
        """Test Polars relax parameter for type coercion."""
        # One with int32, one with int64
        df1 = pl.DataFrame({"a": [1, 2]}, schema={"a": pl.Int32})
        df2 = pl.DataFrame({"a": [3, 4]}, schema={"a": pl.Int64})

        # Without relax, should error (strict mode)
        with pytest.raises(Exception):  # Polars will raise SchemaError or similar
            append_dataframes([df1, df2], allow_missing_cols=False, relax=False)

        # With relax, should succeed and cast to common type
        result = append_dataframes([df1, df2], allow_missing_cols=False, relax=True)
        assert len(result) == 4

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_different_column_order(self, backend: str):
        """Test that column order is preserved from first dataframe."""
        df1_pd = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        df2_pd = pd.DataFrame({"c": [6], "a": [4], "b": [5]})  # Different order

        df1 = from_pandas(df1_pd, backend, to_nw=False)
        df2 = from_pandas(df2_pd, backend, to_nw=False)

        result = append_dataframes([df1, df2], allow_missing_cols=False)
        result_pd = to_pandas(result).reset_index(drop=True)

        # pandas concat preserves first df's column order
        expected = pd.concat([df1_pd, df2_pd], axis=0).reset_index(drop=True)
        pd.testing.assert_frame_equal(result_pd, expected)


class TestDfIsEmpty:
    @pytest.mark.parametrize("data", ([1, 2], []))
    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    @pytest.mark.parametrize("to_nw", [True, False])
    def test_generic(self, spark, data, backend: str, to_nw):
        if (backend == "spark") and (not data):
            return  # Can not infer schema from empty dataset.
        df = pd.DataFrame({"a": data})

        df = from_pandas(df, backend, to_nw=to_nw, spark=spark)
        assert df_is_empty(df) is (False if data else True)

    @pytest.mark.parametrize("data", ([1, 2], []))
    @pytest.mark.parametrize("lazy", [True, False])
    def test_polars_lazy(self, data, lazy: bool):
        df = pl.DataFrame({"a": data})
        df = df.lazy() if lazy else df
        assert df_is_empty(df) is (False if data else True)

    @pytest.mark.skipif(os.environ.get("TESTS_NO_SPARK") == "true", reason="no spark")
    def test_spark_empty(self, spark):
        df = spark.createDataFrame([], schema="a: int, b: int")
        assert df_is_empty(df)


class TestGetCondition:
    """Test the get_condition function."""

    @pytest.fixture(scope="class")
    def sample_df(self):
        """Create a sample dataframe for testing."""
        return nw.from_native(
            pd.DataFrame(
                {
                    "age": [15, 25, 35, None],
                    "score": [50, 75, 85, 90],
                    "name": ["Alice", "Bob", "Charlie", "Dave"],
                    "status": ["active", "pending", "active", "inactive"],
                    "salary": [30000.0, 50000.0, float("nan"), 70000.0],
                }
            )
        )

    # --- Comparison Operators ---

    def test_eq_with_value(self, sample_df):
        """Test equality comparison."""
        cond = get_condition("age", "eq", value=25)
        result = sample_df.filter(cond)
        assert len(result) == 1
        assert nw.to_native(result)["name"].iloc[0] == "Bob"

    def test_ne_with_value(self, sample_df):
        """Test not-equal comparison."""
        cond = get_condition("score", "ne", value=50)
        result = sample_df.filter(cond)
        assert len(result) == 3

    def test_ne_with_value_and_null(self, sample_df):
        """Test not-equal comparison with null."""
        cond = get_condition("age", "ne", value=35)
        result = sample_df.filter(cond)
        assert len(result) == 3

    def test_ne_with_value_and_nan(self, sample_df):
        """Test not-equal comparison with NaN."""
        cond = get_condition("salary", "ne", value=50000.0)
        result = sample_df.filter(cond)
        assert len(result) == 3

    def test_lt_with_value(self, sample_df):
        """Test less-than comparison."""
        cond = get_condition("age", "lt", value=30)
        result = sample_df.filter(cond)
        assert len(result) == 2  # Alice (15) and Bob (25)

    def test_le_with_value(self, sample_df):
        """Test less-than-or-equal comparison."""
        cond = get_condition("age", "le", value=25)
        result = sample_df.filter(cond)
        assert len(result) == 2  # Alice (15) and Bob (25)

    def test_gt_with_value(self, sample_df):
        """Test greater-than comparison."""
        cond = get_condition("age", "gt", value=20)
        result = sample_df.filter(cond)
        assert len(result) == 2  # Bob (25) and Charlie (35)

    def test_ge_with_value(self, sample_df):
        """Test greater-than-or-equal comparison."""
        cond = get_condition("age", "ge", value=25)
        result = sample_df.filter(cond)
        assert len(result) == 2  # Bob (25) and Charlie (35)

    def test_eq_with_compare_col(self):
        """Test equality comparison between two columns."""
        df = nw.from_native(pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 3, 3, 5]}))

        cond = get_condition("a", "eq", compare_col="b")
        result = df.filter(cond)
        assert len(result) == 2  # Rows where a==b (1==1 and 3==3)

    def test_gt_with_compare_col(self):
        """Test greater-than comparison between two columns."""
        df = nw.from_native(pd.DataFrame({"sales": [100, 200, 300], "target": [150, 150, 150]}))

        cond = get_condition("sales", "gt", compare_col="target")
        result = df.filter(cond)
        assert len(result) == 2  # 200>150 and 300>150

    # --- Null/NaN Operators ---

    def test_is_null(self, sample_df):
        """Test null check."""
        cond = get_condition("age", "is_null")
        result = sample_df.filter(cond)
        assert len(result) == 1  # Dave has null age

    def test_is_not_null(self, sample_df):
        """Test not-null check."""
        cond = get_condition("age", "is_not_null")
        result = sample_df.filter(cond)
        assert len(result) == 3  # Alice, Bob, Charlie

    def test_is_nan(self, sample_df):
        """Test NaN check."""
        cond = get_condition("salary", "is_nan")
        result = sample_df.filter(cond)
        assert len(result) == 1  # Charlie has NaN salary

    def test_is_not_nan(self, sample_df):
        """Test not-NaN check."""
        cond = get_condition("salary", "is_not_nan")
        result = sample_df.filter(cond)
        assert len(result) == 3  # Alice, Bob, Dave

    def test_is_nan_vs_is_null_distinction(self):
        """Test that is_nan and is_null are distinct."""
        df = nw.from_native(pl.DataFrame({"value": [1.0, float("nan"), None, 4.0]}))

        # is_nan finds NaN but not None
        nan_result = df.filter(get_condition("value", "is_nan"))
        assert len(nan_result) == 1

        # is_null finds None but not NaN
        null_result = df.filter(get_condition("value", "is_null"))
        assert len(null_result) == 1

    # --- String Operators ---

    def test_contains(self, sample_df):
        """Test substring contains."""
        cond = get_condition("name", "contains", value="li")
        result = sample_df.filter(cond)
        assert len(result) == 2  # Alice and Charlie

    def test_starts_with(self, sample_df):
        """Test string starts_with."""
        cond = get_condition("name", "starts_with", value="C")
        result = sample_df.filter(cond)
        assert len(result) == 1  # Charlie

    def test_ends_with(self, sample_df):
        """Test string ends_with."""
        cond = get_condition("name", "ends_with", value="e")
        result = sample_df.filter(cond)
        assert len(result) == 3  # Alice, Charlie, Dave

    # --- Membership Operators ---

    def test_is_in(self, sample_df):
        """Test set membership."""
        cond = get_condition("status", "is_in", value=["active", "pending"])
        result = sample_df.filter(cond)
        assert len(result) == 3  # Alice, Bob, Charlie

    def test_is_not_in(self, sample_df):
        """Test set non-membership."""
        cond = get_condition("status", "is_not_in", value=["active", "pending"])
        result = sample_df.filter(cond)
        assert len(result) == 1  # Dave (inactive)

    def test_is_not_in_with_nulls(self):
        """Test that is_not_in treats nulls as True."""
        df = nw.from_native(pd.DataFrame({"status": ["active", "pending", None, "inactive"]}))

        cond = get_condition("status", "is_not_in", value=["active"])
        result = df.filter(cond)

        assert len(result) == 3
        result_native = nw.to_native(result)["status"].tolist()
        assert "pending" in result_native
        assert "inactive" in result_native
        assert None in result_native

    def test_is_between(self, sample_df):
        """Test range check (inclusive)."""
        cond = get_condition("age", "is_between", value=[20, 30])
        result = sample_df.filter(cond)
        assert len(result) == 1  # Bob (25)

    def test_is_between_inclusive(self):
        """Test that is_between is inclusive on both ends."""
        df = nw.from_native(pd.DataFrame({"value": [5, 10, 15, 20, 25]}))

        cond = get_condition("value", "is_between", value=[10, 20])
        result = df.filter(cond)
        assert len(result) == 3  # 10, 15, 20 (both boundaries included)

    # --- Edge Cases ---

    def test_comparison_with_null_propagates(self):
        """Test that comparisons with null values propagate null."""
        df = nw.from_native(pd.DataFrame({"a": [1, 2, None, 4]}))

        cond = get_condition("a", "gt", value=2)
        result = df.filter(cond)
        # Null comparison returns null, which is excluded by filter
        assert len(result) == 1  # Only 4 > 2

    def test_empty_iterable_for_is_in(self):
        """Test is_in with empty iterable."""
        df = nw.from_native(pd.DataFrame({"a": [1, 2, 3]}))

        cond = get_condition("a", "is_in", value=[])
        result = df.filter(cond)
        assert len(result) == 0  # Nothing matches empty set

    def test_case_sensitive_string_matching(self):
        """Test that string matching is case-sensitive."""
        df = nw.from_native(pd.DataFrame({"name": ["Alice", "alice", "ALICE"]}))

        cond = get_condition("name", "starts_with", value="A")
        result = df.filter(cond)
        assert len(result) == 2  # "Alice" and "ALICE", not "alice"

    def test_comparison_with_floats(self):
        """Test with float numbers."""
        df = nw.from_native(pd.DataFrame({"x": [1.5, 2.7, 3.9]}))
        cond = get_condition("x", "gt", value=2.0)
        result = df.filter(cond)
        assert len(result) == 2

    def test_comparison_with_dates(self):
        """Test with date/datetime columns."""
        df = nw.from_native(pd.DataFrame({"date": pd.to_datetime(["2023-01-01", "2023-06-01", "2023-12-01"])}))
        cond = get_condition("date", "gt", value=pd.Timestamp("2023-05-01"))
        result = df.filter(cond)
        assert len(result) == 2


class TestGetConditionIntegration:
    """Integration tests combining multiple conditions."""

    def test_combining_conditions_with_and(self):
        """Test combining multiple conditions with AND logic."""
        df = nw.from_native(
            pd.DataFrame(
                {
                    "age": [15, 25, 35, 45],
                    "status": ["active", "active", "inactive", "active"],
                }
            )
        )

        cond1 = get_condition("age", "ge", value=25)
        cond2 = get_condition("status", "eq", value="active")

        result = df.filter(cond1 & cond2)
        assert len(result) == 2  # 25/active and 45/active

    def test_combining_conditions_with_or(self):
        """Test combining multiple conditions with OR logic."""
        df = nw.from_native(
            pd.DataFrame(
                {
                    "age": [15, 25, 35, 45],
                    "status": ["pending", "active", "inactive", "active"],
                }
            )
        )

        cond1 = get_condition("age", "lt", value=20)
        cond2 = get_condition("status", "eq", value="active")

        result = df.filter(cond1 | cond2)
        # 15/pending, 25/active, 45/active
        assert len(result) == 3

    def test_negating_condition(self):
        """Test negating a condition."""
        df = nw.from_native(pd.DataFrame({"status": ["active", "inactive", "pending"]}))

        cond = get_condition("status", "eq", value="active")
        result = df.filter(~cond)
        assert len(result) == 2  # inactive and pending

    def test_complex_filtering_scenario(self):
        """Test a realistic complex filtering scenario."""
        df = nw.from_native(
            pd.DataFrame(
                {
                    "name": ["Alice", "Bob", "Charlie", "Dave", "Eve"],
                    "age": [25, 35, 45, None, 55],
                    "department": [
                        "Sales",
                        "Engineering",
                        "Sales",
                        "HR",
                        "Engineering",
                    ],
                    "salary": [50000, 80000, 60000, 55000, 90000],
                }
            )
        )

        # Find: Engineering department, age >= 30, salary > 70000
        cond_dept = get_condition("department", "eq", value="Engineering")
        cond_age = get_condition("age", "ge", value=30)
        cond_salary = get_condition("salary", "gt", value=70000)

        result = df.filter(cond_dept & cond_age & cond_salary)

        assert len(result) == 2  # Bob and Eve
        result_native = nw.to_native(result)
        assert set(result_native["name"].values) == {"Bob", "Eve"}


class TestJoinDataframes:
    """Test suite for the join_dataframes utility function."""

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    @pytest.mark.parametrize("how", ["inner", "left", "right"])
    @pytest.mark.parametrize("to_nw", [True, False])
    def test_basic(self, backend: str, to_nw: bool, how: str):
        """Test basic join on single column."""
        df_left_pd = pd.DataFrame(
            {
                "user_id": [1, 2, 3, 4],
                "name": ["Alice", "Bob", "Charlie", "David"],
                "age": [25, 30, 35, 40],
            }
        )
        df_right_pd = pd.DataFrame(
            {
                "user_id": [2, 3, 4, 5],
                "city": ["NYC", "LA", "Chicago", "Boston"],
                "country": ["USA", "USA", "USA", "USA"],
            }
        )

        df_left = from_pandas(df_left_pd, backend, to_nw=to_nw)
        df_right = from_pandas(df_right_pd, backend, to_nw=to_nw)

        result = join_dataframes(df_left, df_right, how=how, on="user_id")
        if to_nw:
            assert isinstance(result, (nw.DataFrame, nw.LazyFrame))
        elif backend == "pandas":
            assert isinstance(result, pd.DataFrame)
        else:
            assert isinstance(result, pl.DataFrame)
        df_chk = to_pandas(result).reset_index(drop=True)

        out_cols = df_chk.columns.tolist()
        df_exp = df_left_pd.merge(df_right_pd, on="user_id", how=how)
        assert set(out_cols) == set(df_exp.columns.tolist())
        df_exp = df_exp[out_cols].sort_values("user_id").reset_index(drop=True)
        # Normalize null values to avoid FutureWarning about None vs nan mismatch
        df_chk = df_chk.fillna(np.nan)
        df_exp = df_exp.fillna(np.nan)
        pd.testing.assert_frame_equal(df_chk, df_exp)

    @pytest.mark.parametrize("coalesce_keys", [True, False])
    def test_full_join_coalesce_keys_true(self, coalesce_keys: bool):
        """Test that coalesce_keys functionality."""
        df_left = pl.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        df_right = pl.DataFrame({"id": [2, 3, 4], "val2": ["x", "y", "z"]})

        result = join_dataframes(
            df_left,
            nw.from_native(df_right),
            how="full",
            on="id",
            coalesce_keys=coalesce_keys,
        )
        df_chk = nw.to_native(result)
        df_exp = df_left.join(df_right, on="id", how="full")

        if coalesce_keys:
            df_exp = df_exp.with_columns(pl.coalesce("id", "id_right").alias("id")).drop("id_right")
        pl.testing.assert_frame_equal(df_chk, df_exp)

    def test_cross_join(self):
        """Test cross join produces cartesian product."""
        df_left = pl.DataFrame({"color": ["red", "blue"]})
        df_right = pl.DataFrame({"size": ["S", "M", "L"]})

        # mix the types
        result = join_dataframes(df_left, nw.from_native(df_right), how="cross")
        df_chk = nw.to_native(result)

        df_exp = df_left.join(df_right, how="cross")
        pl.testing.assert_frame_equal(df_chk, df_exp)

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    @pytest.mark.parametrize("to_nw", [True, False])
    def test_right_join_with_different_keys(self, spark, backend: str, to_nw: bool):
        """Test right join with left_on/right_on swaps correctly."""
        df_left_pd = pd.DataFrame({"left_id": [1, 2, 3], "value": ["a", "b", "c"]})
        df_right_pd = pd.DataFrame({"right_id": [2, 3, 4], "data": ["x", "y", "z"]})

        df_left = from_pandas(df_left_pd, backend, to_nw=to_nw, spark=spark)
        df_right = from_pandas(df_right_pd, backend, to_nw=to_nw, spark=spark)

        result = join_dataframes(df_left, df_right, how="right", left_on="left_id", right_on="right_id")
        result_pd = to_pandas(result).reset_index(drop=True)

        # Should keep all right rows (right_id: 2, 3, 4)
        assert len(result_pd) == 3
        assert sorted(result_pd["right_id"].tolist()) == [2, 3, 4]

    @pytest.mark.parametrize("suffix", [None, "_b"])
    def test_suffix_default(self, suffix):
        """Test default suffix is applied to overlapping columns."""
        df_left = pd.DataFrame({"id": [1, 2], "value": [10, 20], "status": ["active", "inactive"]})
        df_right = pd.DataFrame(
            {
                "id": [1, 2],
                "value": [100, 200],  # Overlapping column
                "category": ["A", "B"],
            }
        )

        kws = {"how": "inner", "on": "id"} | ({"suffix": suffix} if suffix else {})
        result = join_dataframes(df_left, nw.from_native(df_right), **kws)
        assert isinstance(result, (nw.DataFrame, nw.LazyFrame))
        result_pd = to_pandas(result).reset_index(drop=True)

        suffix = suffix if suffix else "_right"  # the default is '_right'

        assert "value" in result_pd.columns
        assert f"value{suffix}" in result_pd.columns  # Default suffix
        assert result_pd["value"].tolist() == [10, 20]
        assert result_pd[f"value{suffix}"].tolist() == [100, 200]

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    @pytest.mark.parametrize("to_nw", [True, False])
    def test_suffix_custom(self, spark, backend: str, to_nw: bool):
        """Test custom suffix is applied correctly."""
        df_left_pd = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
        df_right_pd = pd.DataFrame({"id": [1, 2], "value": [100, 200]})

        df_left = from_pandas(df_left_pd, backend, to_nw=to_nw, spark=spark)
        df_right = from_pandas(df_right_pd, backend, to_nw=to_nw, spark=spark)

        result = join_dataframes(df_left, df_right, how="inner", on="id", suffix="_b")
        result_pd = to_pandas(result).reset_index(drop=True)

        assert "value" in result_pd.columns
        assert "value_b" in result_pd.columns
        assert result_pd["value_b"].tolist() == [100, 200]

    @pytest.mark.skipif(os.environ.get("TESTS_NO_SPARK") == "true", reason="no spark")
    def test_spark_broadcast(self, spark):
        """Test Spark broadcast hint is applied."""
        df_left_pd = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
        df_right_pd = pd.DataFrame({"id": [2, 3, 4], "val2": [40, 50, 60]})

        df_left = from_pandas(df_left_pd, "spark", to_nw=False, spark=spark)
        df_right = from_pandas(df_right_pd, "spark", to_nw=True, spark=spark)

        # Should not error with broadcast=True
        result = join_dataframes(df_left, df_right, how="inner", on="id", broadcast=True)
        df_chk = to_pandas(result).sort_values("id").reset_index(drop=True)
        df_exp = df_left_pd.merge(df_right_pd, on="id", how="inner")
        pd.testing.assert_frame_equal(df_chk, df_exp)


class TestNullCondToFalse:
    """Test the 'null_cond_to_false' helper function."""

    def test_null_becomes_false(self):
        """Test that null values in condition become False."""
        df = nw.from_native(pl.DataFrame({"a": [1, 2, None, 4], "b": [10, 20, 30, 40]}))

        # Without null_cond_to_false: nulls propagate
        cond = nw.col("a") > 2
        result_with_nulls = df.filter(cond)
        assert len(result_with_nulls) == 1  # Only row with a=4

        # With null_cond_to_false: nulls become False (same result here)
        cond_safe = null_cond_to_false(nw.col("a") > 2)
        result_without_nulls = df.filter(cond_safe)
        assert len(result_without_nulls) == 1  # Only row with a=4

    def test_true_values_unchanged(self):
        """Test that True values remain True."""
        df = nw.from_native(pd.DataFrame({"a": [True, False, None]}))

        cond = null_cond_to_false(nw.col("a"))
        result = df.with_columns(cond.alias("result"))

        expected = pd.DataFrame({"a": [True, False, None], "result": [True, False, False]})

        pd.testing.assert_frame_equal(nw.to_native(result), expected)


class TestToNativeDataframes:
    """Test suite for the to_native_dataframes utility function."""

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("to_nw", [True, False])
    def test_all_native_single_backend(self, to_nw: bool, backend: str, lazy: bool):
        """Test with all native dataframes from same backend."""
        if lazy and (backend != "polars"):
            return

        df1_pd = pd.DataFrame({"a": [1, 2]})
        df2_pd = pd.DataFrame({"b": [3, 4]})

        df1 = from_pandas(df1_pd, backend, to_nw=to_nw)
        df2 = from_pandas(df2_pd, backend, to_nw=to_nw)

        if lazy:
            df2.lazy()

        native_dfs, detected_backend, found_nw = to_native_dataframes([df1, df2])

        assert len(native_dfs) == 2
        assert detected_backend == backend
        assert found_nw is to_nw

        if to_nw:
            assert not isinstance(native_dfs[0], (nw.DataFrame, nw.LazyFrame))
            assert not isinstance(native_dfs[1], (nw.DataFrame, nw.LazyFrame))
        else:
            assert native_dfs[0] is df1
            assert native_dfs[1] is df2

    def test_mixed_narwhals_and_native_same_backend(self):
        """Test mixing Narwhals and native from same backend."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})

        native_dfs, detected_backend, found_nw = to_native_dataframes([df1, nw.from_native(df2)])

        assert len(native_dfs) == 2
        assert detected_backend == "pandas"
        assert found_nw is True  # At least one was Narwhals
        # All should be native now
        for df in native_dfs:
            assert not isinstance(df, (nw.DataFrame, nw.LazyFrame))

    def test_empty_list_raises_error(self):
        """Test that empty dataframe list raises ValueError."""
        with pytest.raises(ValueError):
            to_native_dataframes([])

    @pytest.mark.parametrize("to_nw", [True, False])
    def test_mixed_backends_pandas_polars(self, to_nw):
        """Test that mixing pandas and polars raises error."""
        df_pd = pd.DataFrame({"a": [1, 2]})
        df_pl = from_pandas(df_pd, "polars", to_nw=to_nw)

        with pytest.raises(ValueError):
            to_native_dataframes([df_pd, df_pl])


class TestValidateOperation:
    """Test Narwhals 'validate_operation' function."""

    @staticmethod
    @pytest.mark.parametrize("op", NULL_OPERATORS)
    def test_valid_null_operator(op):
        """Test with a valid null operator."""
        validate_operation(op)

    @staticmethod
    @pytest.mark.parametrize("op", COMPARISON_OPERATORS)
    def test_valid_standard_operator(op):
        """Test with a valid standard operator."""
        validate_operation(op, 1)

    @staticmethod
    def test_both_value_and_col_compare_provided():
        """Test with wrong input."""
        with pytest.raises(ValueError):
            validate_operation("eq", 10, "col2")

    @staticmethod
    @pytest.mark.parametrize("op", ["is_between"])
    def test_invalid_operator_comparison_column(op: str):
        """Test with not allowed operator for 'compare_col'."""
        with pytest.raises(ValueError):
            validate_operation(op, compare_col="compare")

    @staticmethod
    @pytest.mark.parametrize("value", ["string", 1, None, [None]])
    @pytest.mark.parametrize("op", ["is_in", "is_not_in"])
    def test_invalid_is_in_value(value, op):
        """Test with not allowed value for 'is_in' / 'is_not_in' operator."""
        with pytest.raises((TypeError, ValueError)):
            validate_operation(op, value=value)

    @pytest.mark.parametrize("value", [{1: 2, 3: 4}, [1], [1, 2, 3]])
    def test_invalid_is_between_values(self, value):
        """Test with not allowed value for ris_between operator."""
        with pytest.raises((ValueError, TypeError)):
            validate_operation("is_between", value=value)

    @pytest.mark.parametrize("op", ["contains", "starts_with", "ends_with"])
    def test_invalid_string_operator_values(self, op: str):
        """Test with not allowed value for string operators."""
        with pytest.raises(TypeError):
            validate_operation(op, value=1)
