"""Unit-tests for Narwhals utils."""

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from nlsn.nebula.nw_util import *
from nlsn.nebula.nw_util import COMPARISON_OPERATORS, NULL_OPERATORS


class TestNullCondToFalse:
    """Test the 'null_cond_to_false' helper function."""

    def test_null_becomes_false(self):
        """Test that null values in condition become False."""
        df = nw.from_native(pl.DataFrame({
            "a": [1, 2, None, 4],
            "b": [10, 20, 30, 40]
        }))

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

        expected = pd.DataFrame({
            "a": [True, False, None],
            "result": [True, False, False]
        })

        pd.testing.assert_frame_equal(
            nw.to_native(result),
            expected
        )


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

    @pytest.mark.parametrize(
        "op", ["contains", "starts_with", "ends_with"]
    )
    def test_invalid_string_operator_values(self, op: str):
        """Test with not allowed value for string operators."""
        with pytest.raises(TypeError):
            validate_operation(op, value=1)


class TestGetCondition:
    """Test the get_condition function."""

    @pytest.fixture(scope="class")
    def sample_df(self):
        """Create a sample dataframe for testing."""
        return nw.from_native(pd.DataFrame({
            "age": [15, 25, 35, None],
            "score": [50, 75, 85, 90],
            "name": ["Alice", "Bob", "Charlie", "Dave"],
            "status": ["active", "pending", "active", "inactive"],
            "salary": [30000.0, 50000.0, float('nan'), 70000.0]
        }))

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
        df = nw.from_native(pd.DataFrame({
            "a": [1, 2, 3, 4],
            "b": [1, 3, 3, 5]
        }))

        cond = get_condition("a", "eq", compare_col="b")
        result = df.filter(cond)
        assert len(result) == 2  # Rows where a==b (1==1 and 3==3)

    def test_gt_with_compare_col(self):
        """Test greater-than comparison between two columns."""
        df = nw.from_native(pd.DataFrame({
            "sales": [100, 200, 300],
            "target": [150, 150, 150]
        }))

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
        df = nw.from_native(pl.DataFrame({
            "value": [1.0, float('nan'), None, 4.0]
        }))

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
        df = nw.from_native(pd.DataFrame({
            "status": ["active", "pending", None, "inactive"]
        }))

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
        df = nw.from_native(pd.DataFrame({
            "value": [5, 10, 15, 20, 25]
        }))

        cond = get_condition("value", "is_between", value=[10, 20])
        result = df.filter(cond)
        assert len(result) == 3  # 10, 15, 20 (both boundaries included)

    # --- Edge Cases ---

    def test_comparison_with_null_propagates(self):
        """Test that comparisons with null values propagate null."""
        df = nw.from_native(pd.DataFrame({
            "a": [1, 2, None, 4]
        }))

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
        df = nw.from_native(pd.DataFrame({
            "name": ["Alice", "alice", "ALICE"]
        }))

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
        df = nw.from_native(pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-06-01", "2023-12-01"])
        }))
        cond = get_condition("date", "gt", value=pd.Timestamp("2023-05-01"))
        result = df.filter(cond)
        assert len(result) == 2


class TestGetConditionIntegration:
    """Integration tests combining multiple conditions."""

    def test_combining_conditions_with_and(self):
        """Test combining multiple conditions with AND logic."""
        df = nw.from_native(pd.DataFrame({
            "age": [15, 25, 35, 45],
            "status": ["active", "active", "inactive", "active"]
        }))

        cond1 = get_condition("age", "ge", value=25)
        cond2 = get_condition("status", "eq", value="active")

        result = df.filter(cond1 & cond2)
        assert len(result) == 2  # 25/active and 45/active

    def test_combining_conditions_with_or(self):
        """Test combining multiple conditions with OR logic."""
        df = nw.from_native(pd.DataFrame({
            "age": [15, 25, 35, 45],
            "status": ["pending", "active", "inactive", "active"]
        }))

        cond1 = get_condition("age", "lt", value=20)
        cond2 = get_condition("status", "eq", value="active")

        result = df.filter(cond1 | cond2)
        # 15/pending, 25/active, 45/active
        assert len(result) == 3

    def test_negating_condition(self):
        """Test negating a condition."""
        df = nw.from_native(pd.DataFrame({
            "status": ["active", "inactive", "pending"]
        }))

        cond = get_condition("status", "eq", value="active")
        result = df.filter(~cond)
        assert len(result) == 2  # inactive and pending

    def test_complex_filtering_scenario(self):
        """Test a realistic complex filtering scenario."""
        df = nw.from_native(pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "Dave", "Eve"],
            "age": [25, 35, 45, None, 55],
            "department": ["Sales", "Engineering", "Sales", "HR", "Engineering"],
            "salary": [50000, 80000, 60000, 55000, 90000]
        }))

        # Find: Engineering department, age >= 30, salary > 70000
        cond_dept = get_condition("department", "eq", value="Engineering")
        cond_age = get_condition("age", "ge", value=30)
        cond_salary = get_condition("salary", "gt", value=70000)

        result = df.filter(cond_dept & cond_age & cond_salary)

        assert len(result) == 2  # Bob and Eve
        result_native = nw.to_native(result)
        assert set(result_native["name"].values) == {"Bob", "Eve"}
