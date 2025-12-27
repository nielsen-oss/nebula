"""Test 'apply_to_rows' pipeline functionality.

Tests the apply_to_rows feature which:
1. Splits a DataFrame based on a condition
2. Applies transformers only to matching rows
3. Optionally applies different transformers to non-matching rows (otherwise)
4. Merges results back (unless dead-end)
"""
import os

import narwhals as nw
import numpy as np
import polars as pl
import pytest

from nebula import load_pipeline
from nebula.storage import nebula_storage as ns
from .apply_to_rows_configs import *
from .auxiliaries import CallMe
from ..auxiliaries import pl_assert_equal


@pytest.fixture(scope="module")
def df_input() -> pl.DataFrame:
    """Standard test DataFrame.

    Structure:
    - idx: 0-12 (integers for easy gt/lt testing)
    - c1: mix of strings, empty strings, and nulls
    - c2: mix of strings, empty strings, and nulls

    Null positions in c1: idx 7, 12
    """
    return pl.DataFrame({
        "idx": list(range(13)),
        "c1": ["a", "a", "a", "a", "a", "", "", None, " ", "", "a", "a", None],
        "c2": ["b", "b", "b", "b", "b", "", "", None, None, None, None, "", "b"],
    })


class TestApplyToRowsBasic:
    """Basic apply_to_rows functionality."""

    def test_transforms_matching_rows_only(self, df_input):
        """Rows where idx > 5 get modified, others pass through unchanged."""
        pipe = pipe_apply_to_rows_basic()
        df_out = pipe.run(df_input, show_params=True, force_interleaved_transformer=CallMe())

        assert ns.get("_call_me_") == 2

        # Matching rows (idx > 5) should have c1 = "modified"
        df_matched = df_out.filter(pl.col("idx") > 5)
        assert df_matched["c1"].to_list() == ["modified"] * df_matched.height

        # Non-matching rows should be unchanged
        df_unmatched = df_out.filter(pl.col("idx") <= 5)
        df_expected = df_input.filter(pl.col("idx") <= 5)
        assert df_unmatched.sort("idx").equals(df_expected.sort("idx"))


class TestApplyToRowsDeadEnd:
    """Tests for dead-end behavior (matched rows not merged back)."""

    def test_dead_end_excludes_matched_rows(self, df_input):
        """Matched rows are stored but excluded from output."""
        ns.clear()

        pipe = pipe_apply_to_rows_dead_end()
        df_out = pipe.run(df_input)

        # Output should contain only non-null c1 rows
        assert df_out.filter(pl.col("c1").is_null()).shape[0] == 0

        # Original null count
        n_nulls = df_input.filter(pl.col("c1").is_null()).shape[0]
        assert df_out.shape[0] == df_input.shape[0] - n_nulls

    def test_dead_end_stores_matched_rows(self, df_input):
        """Matched rows should be stored with transformations applied."""
        ns.clear()

        pipe = pipe_apply_to_rows_dead_end()
        pipe.run(df_input)

        # Stored DataFrame should have the new column
        df_stored = ns.get("df_null_rows")
        assert "processed" in df_stored.columns
        assert df_stored["processed"].to_list() == [True] * df_stored.shape[0]

        # Should contain exactly the null rows
        n_nulls = df_input.filter(pl.col("c1").is_null()).shape[0]
        assert df_stored.shape[0] == n_nulls

        ns.clear()


class TestApplyToRowsOtherwise:
    """Tests for the 'otherwise' pipeline."""

    def test_otherwise_transforms_non_matching_rows(self, df_input):
        """Both branches apply their respective transforms."""
        n_rows = df_input.shape[0]
        pipe = pipe_apply_to_rows_otherwise()
        df_out = pipe.run(df_input)

        index = np.arange(n_rows)
        ar_exp = np.where(index > 5, "matched", "not_matched")
        df_exp = (
            df_input.drop("c1")
            .with_columns(pl.Series(name="c1", values=ar_exp))
            .select(df_input.columns)
        )

        pl_assert_equal(df_out, df_exp, sort=["idx"])


class TestApplyToRowsSkipIfEmpty:
    """Tests for skip_if_empty behavior."""

    def test_skip_when_no_rows_match(self, df_input):
        """When no rows match and skip_if_empty=True, output equals input."""
        pipe = pipe_apply_to_rows_skip_if_empty()
        df_out = pipe.run(df_input)
        pl_assert_equal(df_out, df_input)

        # Should be identical to input
        assert df_out.sort("idx").equals(df_input.sort("idx"))

        # The "should_not_appear" value should not exist
        assert "should_not_appear" not in df_out["c1"].to_list()


class TestApplyToRowsComparisonColumn:
    """Tests for column-to-column comparison."""

    def test_compare_two_columns(self, df_input):
        """Rows where c1 > c2 get the new column."""
        pipe = pipe_apply_to_rows_comparison_column()
        df_out = pipe.run(df_input)

        # New column should exist
        assert "result" in df_out.columns

        # Rows where c1 > c2 should have the result value
        # Note: string comparison, null handling may vary
        df_with_result = df_out.filter(pl.col("result") == "c1_gt_c2")
        assert df_with_result.shape[0] > 0  # At least some rows matched


class TestApplyToRowsErrors:
    """Error cases."""

    def test_missing_columns_raises_without_allow(self, df_input):
        """Adding a column in branch without allow_missing_columns should fail."""
        pipe = pipe_apply_to_rows_missing_cols_error()

        with pytest.raises(ValueError):
            pipe.run(df_input)


@pytest.mark.skipif(os.environ.get("TESTS_NO_SPARK") == "true", reason="no spark")
class TestSparkCoalesceRepartitionToOriginal:
    """Test 'coalesce' and 'repartition' options for spark."""

    @staticmethod
    @pytest.fixture(scope="class", name="df_input_spark")
    def _get_df_spark(spark):
        from pyspark.sql.types import IntegerType, StructField, StructType
        fields = [StructField("idx", IntegerType(), True)]
        data = np.arange(100).reshape(-1, 1).tolist()
        return spark.createDataFrame(data, schema=StructType(fields)).coalesce(2)

    @pytest.mark.parametrize("to_nw", (True, False))
    @pytest.mark.parametrize("repartition, coalesce", ([True, False], [False, True]))
    def test(self, df_input_spark, to_nw: bool, repartition: bool, coalesce: bool):
        ns.clear()
        data = {
            "pipeline": [
                {
                    "apply_to_rows": {
                        "input_col": "idx",
                        "operator": "gt",
                        "value": 5,
                    },
                    "repartition_output_to_original": repartition,
                    "coalesce_output_to_original": coalesce,
                    "pipeline": [
                        {
                            "transformer": "Repartition",
                            "params": {"num_partitions": 10}
                        }
                    ]
                }
            ]
        }
        n_exp = df_input_spark.rdd.getNumPartitions()

        pipeline = load_pipeline(data)
        pipeline.show(add_params=True)

        df_out = pipeline.run(nw.from_native(df_input_spark) if to_nw else df_input_spark)
        n_chk = df_out.rdd.getNumPartitions()
        assert n_chk == n_exp

        ns.clear()
