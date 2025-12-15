"""Unit-tests for 'filtering' transformers."""

import os

import narwhals as nw
import pandas as pd
import polars as pl
import pytest
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql.types import StringType, StructField, StructType

from nlsn.nebula.transformers import *


class TestDropNulls:
    """Test suite for DropNulls transformer."""

    @staticmethod
    @pytest.fixture(scope="class", name="df_input_spark")
    def _get_df_input(spark):
        fields = [
            StructField("a_1", StringType(), True),
            StructField("a_2", StringType(), True),
            StructField("a_3", StringType(), True),
            StructField("b_1", StringType(), True),
            StructField("b_2", StringType(), True),
            StructField("b_3", StringType(), True),
        ]
        data = [
            ("1", "11", None, "4", "41", "411"),
            ("1", "12", "120", "4", None, "412"),
            ("1", "12", "120", "4", "41", "412"),
            (None, None, None, None, None, None),
            ("1", "12", "120", "4", "41", None),
            (None, None, None, "4", "41", "412"),
        ]
        return spark.createDataFrame(data, schema=StructType(fields)).persist()

    @pytest.mark.skipif(os.environ.get("TESTS_NO_SPARK") == "true", reason="no spark")
    def test_spark_no_subset(self, df_input_spark):
        """Test DiscardNulls transformer w/o any subsets."""
        how = "any"
        t = DropNulls(how=how)
        df_chk = t.transform(df_input_spark)
        df_exp = df_input_spark.dropna(how=how)
        assert_df_equality(df_chk, df_exp, ignore_row_order=True)

    @pytest.mark.skipif(os.environ.get("TESTS_NO_SPARK") == "true", reason="no spark")
    @pytest.mark.parametrize("how", ["any", "all"])
    def test_spark_columns_subset(self, df_input_spark, how):
        """Test DiscardNulls transformer selecting specific columns."""
        subset = ["a_1", "b_1"]
        t = DropNulls(columns=subset, how=how)
        df_chk = t.transform(df_input_spark)
        df_exp = df_input_spark.dropna(subset=subset, how=how)
        assert_df_equality(df_chk, df_exp, ignore_row_order=True)

    def test_polars_any_all_columns(self):
        """Test dropping rows with any null/NaN across all columns."""
        df = pl.DataFrame({
            "user_id": [1, 2, 3, 4, 5],
            "age": [25.0, 30.0, 35.0, float('nan'), 45.0],
            "score": [100.0, 200.0, None, 400.0, 500.0],
        })

        t = DropNulls(how="any", drop_na=True)
        result = t.transform(df)

        # Rows 3 (null) and 4 (NaN) are dropped
        assert len(result) == 3
        assert result["user_id"].to_list() == [1, 2, 5]

    def test_polars_all_requires_all_missing(self):
        """Test dropping rows only when ALL values are null/NaN."""
        df = pl.DataFrame({
            "col_a": [1.0, None, None, 4.0],
            "col_b": [10.0, 20.0, None, 40.0],
            "col_c": [100.0, 200.0, None, 400.0],
        })

        t = DropNulls(how="all", drop_na=True)
        result = t.transform(df)

        # Only the third row has all nulls
        assert len(result) == 3
        assert result["col_a"][0] == 1.0
        assert result["col_a"][1] is None
        assert result["col_a"][2] == 4.0

    def test_polars_subset_columns(self):
        """Test dropping rows based on nulls in specific columns only."""
        df = pl.DataFrame({
            "user_id": [1, 2, 3, 4],
            "score_primary": [50.0, 100.0, 150.0, None],
            "score_secondary": [100.0, None, 300.0, None],
            "note": [None, "good", None, "excellent"],
        })

        # Only check score columns - ignore note nulls
        t = DropNulls(how="any", columns=["score_primary", "score_secondary"])
        result = t.transform(df)

        # Rows 1 and 3 have nulls in BOTH score columns
        # Rows 2 and 4 have at least one score populated
        assert len(result) == 2
        assert result["user_id"].to_list() == [1, 3]

    def test_polars_with_pattern_and_nan_handling(self):
        """Test dropping rows with NaN vs. null distinction."""
        df = pl.DataFrame({
            "user_id": [1, 2, 3, 4],
            "revenue_q1": [100.0, float('nan'), 300.0, 400.0],
            "revenue_q2": [200.0, 250.0, None, 450.0],
            "cost_q1": [50.0, 60.0, 70.0, None],
        })

        # Check revenue columns, including NaN
        t = DropNulls(how="any", glob="revenue_*", drop_na=True)
        result = t.transform(df)

        # Rows 2 (NaN) and 3 (null) have missing revenue values
        assert len(result) == 2
        assert result["user_id"].to_list() == [1, 4]

    def test_polars_ignore_nan(self):
        """Test ignoring NaN when drop_null=False."""
        df = pl.DataFrame({
            "user_id": [1, 2, 3],
            "value": [100.0, float('nan'), None],
        })

        # Only drop actual nulls, not NaN
        t = DropNulls(how="any", drop_na=False)
        result = t.transform(df)

        # Row 3 has null, row 2 has NaN (should remain)
        assert len(result) == 2
        assert result["user_id"].to_list() == [1, 2]
        # Verify NaN is still present
        assert result["value"][1] != result["value"][1]  # NaN != NaN

    @pytest.mark.parametrize("how", ["any", "all"])
    @pytest.mark.parametrize("columns", [None, "age"])
    def test_pandas_how(self, how: str, columns):
        """Test 'how' parameter in pandas."""
        df = pd.DataFrame({
            "user_id": [1, 2, 3, 4, 5],
            "age": [25.0, float('nan'), float('nan'), float('nan'), 45.0],
            "score": [100.0, 200.0, float('nan'), 400.0, 500.0],
        })
        t = DropNulls(how=how, columns=columns)
        df_chk = t.transform(df)
        df_exp = df.dropna(how=how, subset=columns)
        pd.testing.assert_frame_equal(df_chk, df_exp)

    @pytest.mark.parametrize("thresh", [1, 2])
    def test_pandas_thresh(self, thresh: int):
        """Test 'thresh' parameter in pandas."""
        df = pd.DataFrame({
            "user_id": [1, 2, 3, 4, 5],
            "age": [25.0, float('nan'), float('nan'), float('nan'), 45.0],
            "score": [100.0, 200.0, None, 400.0, 500.0],
        })
        t = DropNulls(thresh=thresh)
        df_chk = t.transform(df)
        df_exp = df.dropna(thresh=thresh)
        pd.testing.assert_frame_equal(df_chk, df_exp)


class TestFilterValidation:
    """Test Filter initialization validation."""

    def test_double_negation_is_not_in_rejected(self):
        """Test that perform='remove' + is_not_in raises error."""
        with pytest.raises(ValueError, match="double negation"):
            Filter(
                input_col="status",
                perform="remove",
                operator="is_not_in",
                value=["active"],
            )

    def test_double_negation_is_not_null_rejected(self):
        """Test that perform='remove' + is_not_null raises error."""
        with pytest.raises(ValueError, match="double negation"):
            Filter(
                input_col="age",
                perform="remove",
                operator="is_not_null",
            )

    def test_double_negation_is_not_nan_rejected(self):
        """Test that perform='remove' + is_not_nan raises error."""
        with pytest.raises(ValueError, match="double negation"):
            Filter(
                input_col="score",
                perform="remove",
                operator="is_not_nan",
            )

    def test_invalid_perform_value_rejected(self):
        """Test that invalid 'perform' values are rejected."""
        with pytest.raises((ValueError, AssertionError)):
            Filter(
                input_col="age",
                perform="maybe",  # Invalid
                operator="gt",
                value=18,
            )


class TestFilter:
    """Test Filter transformer."""

    # -------------- perform="keep" --------------

    @pytest.fixture(scope="class")
    def df(self):
        """Create a DataFrame with nulls and NaNs for testing."""
        return nw.from_native(
            pd.DataFrame({
                "age": [15, 25, 35, None],
                "score": [50.0, 75.0, float("nan"), 90.0],
                "name": ["Alice", "Bob", "Charlie", "Dave"],
                "status": ["active", "pending", "active", None],
            })
        )

    def test_keep_with_eq(self, df):
        """Test keep with equality - nulls are excluded."""
        t = Filter(input_col="status", perform="keep", operator="eq", value="active")
        result = t.transform(df)

        result_native = nw.to_native(result)
        assert len(result_native) == 2  # Alice and Charlie
        assert set(result_native["name"]) == {"Alice", "Charlie"}

    def test_keep_with_ne(self, df):
        """Test keep with not-equal - nulls are excluded."""
        t = Filter(input_col="status", perform="keep", operator="ne", value="active")
        result = t.transform(df)

        result_native = nw.to_native(result)
        # "pending" only - Dave's null is excluded (null != "active" → null → excluded)
        assert len(result_native) == 2
        assert result_native["name"].tolist() == ["Bob", "Dave"]

    def test_keep_with_gt_excludes_nulls(self, df):
        """Test that comparisons exclude nulls (null > 20 → null → excluded)."""
        t = Filter(input_col="age", perform="keep", operator="gt", value=20)
        result = t.transform(df)

        result_native = nw.to_native(result)
        # Bob (25) and Charlie (35) - Dave's null excluded
        assert len(result_native) == 2
        assert set(result_native["name"]) == {"Bob", "Charlie"}

    def test_keep_with_gt_excludes_nans(self, df):
        """Test that comparisons exclude NaNs (NaN > 60 → False)."""
        t = Filter(input_col="score", perform="keep", operator="gt", value=60)
        result = t.transform(df)

        result_native = nw.to_native(result)
        # Dave (90) + Charlie's NaN
        assert len(result_native) == 2
        assert result_native["name"].tolist() == ["Bob", "Dave"]

    def test_keep_with_is_null(self, df):
        """Test explicit null checking."""
        t = Filter(input_col="age", perform="keep", operator="is_null")
        result = t.transform(df)

        result_native = nw.to_native(result)
        assert len(result_native) == 1
        assert result_native["name"].iloc[0] == "Dave"

    def test_keep_with_is_not_null(self, df):
        """Test explicit not-null checking."""
        t = Filter(input_col="age", perform="keep", operator="is_not_null")
        result = t.transform(df)

        result_native = nw.to_native(result)
        assert len(result_native) == 3  # Alice, Bob, Charlie

    def test_keep_with_is_nan(self, df):
        """Test explicit NaN checking (distinct from null)."""
        t = Filter(input_col="score", perform="keep", operator="is_nan")
        result = t.transform(df)

        result_native = nw.to_native(result)
        assert len(result_native) == 1
        assert result_native["name"].iloc[0] == "Charlie"

    def test_keep_with_is_not_nan(self, df):
        """Test explicit not-NaN checking."""
        t = Filter(input_col="score", perform="keep", operator="is_not_nan")
        result = t.transform(df)

        result_native = nw.to_native(result)
        # Alice (50), Bob (75), Dave (90) - Charlie's NaN excluded
        assert len(result_native) == 3
        assert set(result_native["name"]) == {"Alice", "Bob", "Dave"}

    def test_keep_with_is_in(self, df):
        """Test set membership - nulls excluded."""
        t = Filter(
            input_col="status",
            perform="keep",
            operator="is_in",
            value=["active", "pending"],
        )
        result = t.transform(df)

        result_native = nw.to_native(result)
        # Alice (active), Bob (pending), Charlie (active) - Dave's null excluded
        assert len(result_native) == 3
        assert set(result_native["name"]) == {"Alice", "Bob", "Charlie"}

    # -------------- perform="remove" --------------

    def test_remove_with_eq_keeps_nulls(self, df):
        """Test remove with equality - nulls are KEPT (not removed)."""
        t = Filter(input_col="status", perform="remove", operator="eq", value="active")
        result = t.transform(df)

        result_native = nw.to_native(result)
        # Bob (pending) and Dave (null) - null != "active" so it's kept
        assert len(result_native) == 2
        assert set(result_native["name"]) == {"Bob", "Dave"}

    def test_remove_with_gt_keeps_nulls(self, df):
        """Test that remove with comparisons keeps nulls."""
        t = Filter(input_col="age", perform="remove", operator="gt", value=20)
        result = t.transform(df)

        result_native = nw.to_native(result)
        # Remove: Bob (25 > 20) and Charlie (35 > 20)
        # Keep: Alice (15 not > 20) and Dave (null, comparison is null → not removed)
        assert len(result_native) == 2
        assert set(result_native["name"]) == {"Alice", "Dave"}

    def test_remove_with_is_in_keeps_nulls(self, df):
        """Test that remove with is_in keeps nulls (important!)."""
        t = Filter(
            input_col="status",
            perform="remove",
            operator="is_in",
            value=["active"],
        )
        result = t.transform(df)

        result_native = nw.to_native(result)
        # Remove: Alice and Charlie (active is in [active])
        # Keep: Bob (pending not in [active]) and Dave (null not in [active])
        assert len(result_native) == 2
        assert set(result_native["name"]) == {"Bob", "Dave"}

    def test_remove_vs_keep_with_is_in_are_different(self, df):
        """Demonstrate that perform='remove' + is_in handles nulls differently than perform='keep' + is_not_in would."""
        # Remove rows where status is "active"
        t_remove = Filter(
            input_col="status",
            perform="remove",
            operator="is_in",
            value=["active"],
        )
        result_remove = t_remove.transform(df)

        # If we could use keep + is_not_in (which we banned), nulls would be excluded
        # But with remove + is_in, nulls are kept
        result_native = nw.to_native(result_remove)
        assert len(result_native) == 2
        # Critically: Dave (null) is KEPT
        assert "Dave" in set(result_native["name"])

    def test_remove_with_is_null_removes_nulls(self, df):
        """Test that remove + is_null removes null rows."""
        t = Filter(input_col="status", perform="remove", operator="is_null")
        result = t.transform(df)

        result_native = nw.to_native(result)
        # Remove Dave (null) - keep everyone else
        assert len(result_native) == 3
        assert set(result_native["name"]) == {"Alice", "Bob", "Charlie"}

    def test_remove_with_is_nan_removes_nans(self, df):
        """Test that remove + is_nan removes NaN rows."""
        t = Filter(input_col="score", perform="remove", operator="is_nan")
        result = t.transform(df)

        result_native = nw.to_native(result)
        # Remove Charlie (NaN) - keep everyone else
        assert len(result_native) == 3
        assert set(result_native["name"]) == {"Alice", "Bob", "Dave"}

    # -------------- test null and nan --------------

    @pytest.fixture(scope="class")
    def df_null_nan(self):
        """Create a DataFrame that clearly separates null and NaN."""
        return nw.from_native(
            pl.DataFrame({
                "value": [1.0, None, float("nan"), 4.0],
                "label": ["one", "null", "nan", "four"],
            })
        )

    def test_is_null_finds_only_null(self, df_null_nan):
        """Test that is_null finds None but not NaN."""
        t = Filter(input_col="value", perform="keep", operator="is_null")
        result = t.transform(df_null_nan)

        result_native = nw.to_native(result)
        assert len(result_native) == 1
        assert result_native["label"][0] == "null"

    def test_is_nan_finds_only_nan(self, df_null_nan):
        """Test that is_nan finds NaN but not None."""
        t = Filter(input_col="value", perform="keep", operator="is_nan")
        result = t.transform(df_null_nan)

        result_native = nw.to_native(result)
        assert len(result_native) == 1
        assert result_native["label"][0] == "nan"

    def test_is_not_null_excludes_only_null(self, df_null_nan):
        """Test that is_not_null keeps NaN but excludes None."""
        t = Filter(input_col="value", perform="keep", operator="is_not_null")
        result = t.transform(df_null_nan)

        result_native = nw.to_native(result)
        assert len(result_native) == 3  # one, nan, four
        assert set(result_native["label"]) == {"one", "nan", "four"}

    def test_is_not_nan_excludes_only_nan(self, df_null_nan):
        """Test that is_not_nan keeps None but excludes NaN."""
        t = Filter(input_col="value", perform="keep", operator="is_not_nan")
        result = t.transform(df_null_nan)

        result_native = nw.to_native(result)
        assert len(result_native) == 3  # one, null, four
        set_results = set(result_native["label"].to_list())
        assert set_results == {"one", "null", "four"}

    # -------------- test comparison column --------------

    @pytest.fixture(scope="class")
    def df_compare_col(self):
        """Create a DataFrame for column comparison tests."""
        return nw.from_native(
            pd.DataFrame({
                "sales": [100, 200, None, 400],
                "target": [150, 150, 150, 150],
                "name": ["Alice", "Bob", "Charlie", "Dave"],
            })
        )

    def test_keep_with_column_comparison(self, df_compare_col):
        """Test that column comparisons work with perform='keep'."""
        t = Filter(
            input_col="sales",
            perform="keep",
            operator="gt",
            compare_col="target",
        )
        result = t.transform(df_compare_col)

        result_native = nw.to_native(result)
        # Bob (200 > 150) and Dave (400 > 150)
        # Charlie's null is excluded (null > 150 → null → excluded)
        assert len(result_native) == 2
        assert set(result_native["name"]) == {"Bob", "Dave"}

    def test_remove_with_column_comparison_keeps_nulls(self, df_compare_col):
        """Test that remove with column comparison keeps nulls."""
        t = Filter(
            input_col="sales",
            perform="remove",
            operator="gt",
            compare_col="target",
        )
        result = t.transform(df_compare_col)

        result_native = nw.to_native(result)
        # Remove: Bob (200 > 150) and Dave (400 > 150)
        # Keep: Alice (100 not > 150) and Charlie (null, not removed)
        assert len(result_native) == 2
        assert set(result_native["name"]) == {"Alice", "Charlie"}

    # -------------- string operations --------------

    @pytest.fixture(scope="class")
    def df_strings(self):
        """Create a DataFrame with strings and nulls."""
        return nw.from_native(
            pd.DataFrame({
                "email": [
                    "alice@company.com",
                    "bob@external.org",
                    None,
                    "charlie@company.com",
                ],
                "name": ["Alice", "Bob", "Charlie", "Dave"],
            })
        )

    def test_keep_contains(self, df_strings):
        """Test string contains with keep."""
        t = Filter(
            input_col="email",
            perform="keep",
            operator="contains",
            value="company",
        )
        result = t.transform(df_strings.filter(~nw.col("email").is_null()))

        result_native = nw.to_native(result)
        assert len(result_native) == 2
        assert set(result_native["name"]) == {"Alice", "Dave"}

    def test_remove_contains_keeps_nulls(self, df_strings):
        """Test that 'remove' + 'contains' keep null values."""
        t = Filter(
            input_col="email",
            perform="remove",
            operator="contains",
            value="external",
        )
        result = t.transform(df_strings)

        result_native = nw.to_native(result)
        # Remove: Bob (@external)
        # Keep: Alice, Charlie (null), Dave
        assert len(result_native) == 3
        assert set(result_native["name"]) == {"Alice", "Charlie", "Dave"}

    def test_starts_with(self, df_strings):
        """Test string starts_with."""
        t = Filter(
            input_col="email",
            perform="keep",
            operator="starts_with",
            value="alice",
        )
        result = t.transform(df_strings.filter(~nw.col("email").is_null()))

        result_native = nw.to_native(result)
        assert len(result_native) == 1
        assert result_native["name"].iloc[0] == "Alice"
