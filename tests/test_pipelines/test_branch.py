"""Test 'branch' pipeline functionality.

Tests the branch feature which:
1. Creates a secondary pipeline (from main DataFrame or storage)
2. Transforms the branched data
3. Merges back via: append (union), join, or dead-end (no merge)
4. Optionally applies a different transform to the main DataFrame (otherwise)
"""

import os

import numpy as np
import polars as pl
import pytest

from nebula.pipelines.pipeline_loader import load_pipeline
from nebula.storage import nebula_storage as ns
from .branch_configs import *


@pytest.fixture
def df_input() -> pl.DataFrame:
    """Standard test DataFrame.

    Structure:
    - idx: 0-5 (integers, unique, for join testing)
    - c1, c2: simple string columns
    """
    return pl.DataFrame({
        "idx": [0, 1, 2, 3, 4, 5],
        "c1": ["a", "b", "c", "d", "e", "f"],
        "c2": ["x", "y", "z", "w", "v", "u"],
    })


class TestBranchDeadEnd:
    """Dead-end: branch runs but result is not merged back."""

    def test_dead_end_passes_through_unchanged(self, df_input):
        """Main DataFrame should pass through unchanged."""
        ns.clear()

        pipe = pipe_branch_dead_end()
        df_out = pipe.run(df_input)

        # Output equals input (branch result discarded)
        assert df_out.sort("idx").equals(df_input.sort("idx"))

        # branch_col should NOT be in output
        assert "branch_col" not in df_out.columns

    def test_dead_end_stores_branch_result(self, df_input):
        """Branch result should be stored for inspection."""
        ns.clear()

        pipe = pipe_branch_dead_end()
        pipe.run(df_input)

        # Branch result should be in storage
        df_stored = ns.get("df_branch_result")
        assert "branch_col" in df_stored.columns
        assert df_stored["branch_col"].to_list() == ["from_branch"] * df_input.height

        ns.clear()

    def test_dead_end_from_storage(self, df_input):
        """Branch can read from storage instead of main DataFrame."""
        ns.clear()

        # Store a different DataFrame for the branch to use
        df_source = pl.DataFrame({
            "idx": [100, 101],
            "c1": ["stored_a", "stored_b"],
            "c2": ["stored_x", "stored_y"],
        })
        ns.set("df_source", df_source)

        pipe = pipe_branch_dead_end_from_storage()
        df_out = pipe.run(df_input)

        # Main DataFrame passes through unchanged
        assert df_out.sort("idx").equals(df_input.sort("idx"))

        # Branch processed the stored DataFrame
        df_stored = ns.get("df_branch_result")
        assert df_stored.height == 2  # From df_source, not df_input
        assert "branch_col" in df_stored.columns

        ns.clear()


class TestBranchAppend:
    """Append: branch result is appended to main DataFrame."""

    def test_append_doubles_rows(self, df_input):
        """Output should have 2x rows (original + branch)."""
        pipe = pipe_branch_append()
        df_out = pipe.run(df_input)

        assert df_out.height == df_input.height * 2

    def test_append_has_both_versions(self, df_input):
        """Output should contain original c1 values and branch c1 values."""
        pipe = pipe_branch_append()
        df_out = pipe.run(df_input)

        c1_values = set(df_out["c1"].to_list())

        # Original values
        assert "a" in c1_values
        # Branch value
        assert "from_branch" in c1_values

    def test_append_new_column_with_allow_missing(self, df_input):
        """Branch can add new column when allow_missing_columns=True."""
        pipe = pipe_branch_append_new_column()
        df_out = pipe.run(df_input)

        assert "new_col" in df_out.columns
        assert df_out.height == df_input.height * 2

        # Original rows should have null in new_col
        # Branch rows should have "branch_value"
        new_col_values = df_out["new_col"].to_list()
        assert "branch_value" in new_col_values
        assert None in new_col_values

    def test_append_missing_cols_raises_without_allow(self, df_input):
        """Adding column without allow_missing_columns should fail."""
        pipe = pipe_branch_append_missing_cols_error()

        with pytest.raises(ValueError):
            pipe.run(df_input)

    def test_append_from_storage(self, df_input):
        """Branch can read from storage and append to main."""
        ns.clear()

        df_source = pl.DataFrame({
            "idx": [100, 101],
            "c1": ["stored_a", "stored_b"],
            "c2": ["stored_x", "stored_y"],
        })
        ns.set("df_source", df_source)

        pipe = pipe_branch_append_from_storage()
        df_out = pipe.run(df_input)

        # Output has main + branch rows
        assert df_out.height == df_input.height + df_source.height

        # Branch rows have modified c1
        assert "from_storage_branch" in df_out["c1"].to_list()

        ns.clear()


class TestBranchJoin:
    """Join: branch result is joined to main DataFrame."""

    def test_join_adds_column(self, df_input):
        """Join should add new_col from branch."""
        pipe = pipe_branch_join()
        df_out = pipe.run(df_input)

        assert "new_col" in df_out.columns
        assert df_out["new_col"].to_list() == ["joined"] * df_input.height

    def test_join_preserves_row_count(self, df_input):
        """Inner join on same data should preserve row count."""
        pipe = pipe_branch_join()
        df_out = pipe.run(df_input)

        assert df_out.height == df_input.height

    def test_join_from_storage(self, df_input):
        """Branch can read from storage and join to main."""
        ns.clear()

        # Store DataFrame with subset of idx values
        df_source = pl.DataFrame({
            "idx": [0, 1, 2],  # Only 3 of 6 idx values
            "c1": ["x", "y", "z"],
            "c2": ["a", "b", "c"],
        })
        ns.set("df_source", df_source)

        pipe = pipe_branch_join_from_storage()
        df_out = pipe.run(df_input)

        # Inner join: only rows with matching idx
        assert df_out.height == 3
        assert "new_col" in df_out.columns

        ns.clear()


class TestBranchOtherwise:
    """Otherwise: separate transform for main DataFrame."""

    def test_append_otherwise_transforms_both(self, df_input):
        """Both main and branch should be transformed differently."""
        pipe = pipe_branch_append_otherwise()
        df_out = pipe.run(df_input)

        c1_values = set(df_out["c1"].to_list())

        # Main DataFrame transformed
        assert "main_transformed" in c1_values
        # Branch transformed
        assert "branch_transformed" in c1_values
        # Original values should NOT be present
        assert "a" not in c1_values

    def test_join_otherwise_transforms_main(self, df_input):
        """Main DataFrame should have otherwise transform applied before join."""
        pipe = pipe_branch_join_otherwise()
        df_out = pipe.run(df_input)

        # Main got other_col from otherwise
        assert "other_col" in df_out.columns
        assert df_out["other_col"].to_list() == ["main_marker"] * df_input.height

        # Branch provided new_col via join
        assert "new_col" in df_out.columns
        assert df_out["new_col"].to_list() == ["joined"] * df_input.height


class TestBranchSkip:
    """Skip/Perform: conditionally disable the branch."""

    @pytest.mark.parametrize("pipe", [pipe_branch_skip, pipe_branch_not_perform])
    def test_skip_passes_through_unchanged(self, df_input, pipe):
        """When skip=True / perform=False, main DataFrame passes through unchanged."""
        df_out = pipe().run(df_input)

        assert df_out.sort("idx").equals(df_input.sort("idx"))

        # Branch transform should NOT be applied
        assert "should_not_appear" not in df_out["c1"].to_list()

    def test_skip_otherwise_still_runs(self, df_input):
        """When skip=True, otherwise pipeline should still run."""
        pipe = pipe_branch_skip_otherwise()
        df_out = pipe.run(df_input)

        # Otherwise transform applied
        assert df_out["c1"].to_list() == ["otherwise_applied"] * df_input.height

        # Branch transform NOT applied
        assert "should_not_appear" not in df_out["c1"].to_list()


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

    @pytest.mark.parametrize("repartition, coalesce", ([True, False], [False, True]))
    def test(self, df_input_spark, repartition: bool, coalesce: bool):
        ns.clear()
        data = {
            "pipeline": [
                {
                    "branch": {"end": "append"},
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
        pipeline.show()

        df_out = pipeline.run(df_input_spark)
        n_chk = df_out.rdd.getNumPartitions()
        assert n_chk == n_exp

        ns.clear()
