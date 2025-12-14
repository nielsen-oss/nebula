"""Test split pipeline functionalities using Polars."""
import os

import polars as pl
import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from nlsn.nebula.pipelines.pipelines import TransformerPipeline
from nlsn.nebula.storage import nebula_storage as ns
from nlsn.nebula.transformers import AddLiterals, AssertNotEmpty, Cast
from .auxiliaries import *
from ..auxiliaries import to_polars

_nan = float("nan")

_DATA = [
    [0.1234, "a", "b"],
    [0.1234, "a", "b"],
    [0.1234, "a", "b"],
    [1.1234, "a", "  b"],
    [2.1234, "  a  ", "  b  "],
    [3.1234, "", ""],
    [4.1234, "   ", "   "],
    [5.1234, None, None],
    [6.1234, " ", None],
    [6.1234, " ", None],
    [6.1234, " ", None],
    [7.1234, "", None],
    [8.1234, "a", None],
    [8.1234, "a", None],
    [8.1234, "a", None],
    [9.1234, "a", ""],
    [9.1234, "a", ""],
    [9.1234, "a", ""],
    [9.1234, "a", ""],
    [10.1234, "   ", "b"],
    [10.1234, "   ", "b"],
    [11.1234, "a", None],
    [11.1234, "a", None],
    [11.1234, "a", None],
    [11.1234, "a", None],
    [11.1234, "a", None],
    [12.1234, None, "b"],
    [12.1234, None, "b"],
    [13.1234, None, "b"],
    [13.1234, None, "b"],
    [13.1234, None, "b"],
    [14.1234, None, None],
    [14.1234, None, None],
    [15.1234, None, None],
    [15.1234, None, None],
    [None, "a", "b"],
    [None, "c", "d"],
    [_nan, "c", "d"],
    [_nan, "c", "d"],
]

# Transformer lists for split tests
_trf_low: list = [RoundValues(input_columns="c1", precision=3)]
_trf_hi: list = [
    RoundValues(input_columns="c1", precision=1),
    Cast(cast={"c1": "float64"}),
]


# =============================================================================
# Fixtures and Helper Functions
# =============================================================================


@pytest.fixture(scope="module")
def df_input() -> pl.DataFrame:
    """Get the input dataframe."""
    schema = {"c1": pl.Float64, "c2": pl.String, "c3": pl.String}
    return pl.DataFrame(_DATA, schema=schema, orient="row")


def _split_function(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Split dataframe into 'low' (c1 < 10) and 'hi' (c1 >= 10) subsets."""
    cond = pl.col("c1") < 10
    # Handle nulls: rows with null in c1 go to neither split
    cond_not_null = pl.col("c1").is_not_null() & pl.col("c1").is_not_nan()
    return {
        "low": df.filter(cond & cond_not_null),
        "hi": df.filter(~cond & cond_not_null),
    }


def _split_function_with_null(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Split dataframe into 'low', 'hi', and 'null' subsets."""
    ret = _split_function(df)
    # Include both actual nulls and NaN values in the 'null' split
    cond_null = pl.col("c1").is_null() | pl.col("c1").is_nan()
    return {**ret, "null": df.filter(cond_null)}


# =============================================================================
# Test Classes
# =============================================================================


class TestSplitPipeline:
    """Generic tests for split pipelines."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Clear storage before and after each test."""
        ns.clear()
        yield
        ns.clear()

    @pytest.mark.parametrize("name", [None, "name_01"])
    def test_basic(
            self,
            df_input: pl.DataFrame,
            name: str | None,
    ):
        """Test with various configurations."""
        dict_splits = {"low": _trf_low, "hi": _trf_hi}

        pipe = TransformerPipeline(
            dict_splits,
            split_function=_split_function,
            interleaved=[AssertNotEmpty()],  # (no-op, just for testing flow)
            name=name,
        )

        pipe.show_pipeline()
        df_chk = pipe.run(df_input)

        # Create the expected DF
        split_dfs = _split_function(df_input)
        df_low = split_dfs["low"]
        df_hi = split_dfs["hi"]

        # Apply transformers to each split
        for t in _trf_low:
            df_low = t.transform(df_low)

        for t in _trf_hi:
            df_hi = t.transform(df_hi)

        df_exp = pl.concat([df_low, df_hi], how="vertical")

        pl_assert_equal(df_chk.sort(df_chk.columns), df_exp.sort(df_exp.columns))

    def test_cast_subset_to_input_schema(self, df_input):
        """Test with various configurations."""
        # cast c1 to float32 ...
        dict_splits = {"low": [], "hi": [Cast(cast={"c1": "float32"})], "null": []}

        pipe = TransformerPipeline(
            dict_splits,
            split_function=_split_function_with_null,
            cast_subset_to_input_schema=True,
        )

        pipe.show_pipeline()
        df_chk = pipe.run(df_input)
        df_chk = to_polars(df_chk)
        # ... and ensure it is converted back to float64 at the end
        assert df_chk["c1"].dtype == pl.Float64
        pl_assert_equal(df_chk.sort(df_chk.columns), df_input.sort(df_input.columns))

    def test_allow_missing_columns(self, df_input: pl.DataFrame):
        """Test with allow_missing_columns=True."""
        t_new_col = AddLiterals(data=[{"value": "hello", "alias": "new_column"}])
        dict_splits = {"low": [], "hi": [t_new_col]}

        pipe = TransformerPipeline(
            dict_splits,
            split_function=_split_function,
            allow_missing_columns=True,
        )
        df_chk = pipe.run(df_input)

        # Build expected result
        dict_df = _split_function(df_input)
        df_low = dict_df["low"]
        df_hi = dict_df["hi"].with_columns(pl.lit("hello").alias("new_column"))
        df_exp = pl.concat([df_low, df_hi], how="diagonal")
        pl_assert_equal(df_chk.sort(df_chk.columns), df_exp.sort(df_exp.columns))

    @pytest.mark.parametrize(
        "splits_skip_if_empty", [None, "hi", {"hi"}, {"hi", "low"}]
    )
    def test_skip_if_empty(self, df_input: pl.DataFrame, splits_skip_if_empty):
        """Test with splits_skip_if_empty argument."""

        def _mock_split(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
            return {"low": df, "hi": df.head(0)}  # Empty 'hi' split

        pipe = TransformerPipeline(
            {"low": [], "hi": [ThisTransformerIsBroken()]},
            split_function=_mock_split,
            splits_skip_if_empty=splits_skip_if_empty,
        )

        if splits_skip_if_empty in {None, "low"}:
            with pytest.raises(ValueError):
                pipe.run(df_input)
            return

        df_chk = pipe.run(df_input)
        pl_assert_equal(df_chk.sort(df_chk.columns), df_input.sort(df_input.columns))

    @pytest.mark.parametrize("splits_skip_if_empty", ["wrong", {"wrong"}, ["wrong"]])
    def test_splits_skip_if_empty_wrong_split(self, splits_skip_if_empty):
        """Test with splits_skip_if_empty and a wrong split name."""
        with pytest.raises(KeyError):
            TransformerPipeline(
                {"low": [], "hi": []},
                split_function=lambda x: x,
                splits_skip_if_empty=splits_skip_if_empty,
            )

    @pytest.mark.parametrize("interleaved", [None, [CallMe()], CallMe()])
    @pytest.mark.parametrize("prepend", [True, False])
    @pytest.mark.parametrize("append", [True, False])
    def test_interleaved(self, df_input: pl.DataFrame, interleaved, prepend, append):
        """Test the interleaved capability."""
        ns.clear()
        dict_splits = {"low": [Distinct(), Distinct()], "hi": Distinct()}

        pipe = TransformerPipeline(
            dict_splits,
            split_function=_split_function,
            interleaved=interleaved,
            prepend_interleaved=prepend,
            append_interleaved=append,
            allow_missing_columns=True,
        )

        pipe.show_pipeline()
        df_chk = pipe.run(df_input)

        df_exp = df_input.filter(pl.col("c1").is_not_null() & pl.col("c1").is_not_nan()).unique()

        if interleaved:
            n_chk = ns.get("_call_me_")
            n_exp = 1
            if prepend:
                n_exp += 2  # -> 2 split, run twice
            if append:
                n_exp += 2  # -> 2 split, run twice
            assert n_chk == n_exp
        else:
            assert "_call_me_" not in ns.list_keys()

        pl_assert_equal(df_chk.sort(df_chk.columns), df_exp.sort(df_chk.columns))
        ns.clear()


class TestSplitPipelineApplyTransformerBeforeAndAfter:
    """Test adding transformers after splitting or before appending."""

    @pytest.mark.parametrize(
        "where, transformer",
        [
            ("split_apply_before_appending", Distinct()),
            ("split_apply_before_appending", [Distinct()]),
            ("split_apply_after_splitting", (Distinct(),)),
        ],
    )
    def test_before_and_after(self, df_input: pl.DataFrame, where: str, transformer):
        """Add a transformer after splitting or before appending.

        The 'Distinct' transformer is applied only in the specified location.
        """
        dict_transformers = {"low": AssertNotEmpty(), "hi": AssertNotEmpty()}

        pipe = TransformerPipeline(
            dict_transformers,
            split_function=_split_function,
            cast_subset_to_input_schema=True,
            **{where: transformer}
        )

        pipe.show_pipeline()

        df_chk = pipe.run(df_input)

        # The expected result is distinct rows from the splits
        split_dfs = _split_function(df_input)
        df_exp = pl.concat(
            [split_dfs["low"].unique(), split_dfs["hi"].unique()],
            how="diagonal"
        )
        pl_assert_equal(df_chk.sort(df_chk.columns), df_exp.sort(df_exp.columns))

    @pytest.mark.parametrize(
        "where",
        [
            "split_apply_before_appending",
            "split_apply_after_splitting",
        ],
    )
    def test_error_pipeline_instead_of_transformer(self, where: str):
        """Pass a TransformerPipeline instead of a transformer - should raise."""
        dict_transformers = {"low": AssertNotEmpty(), "hi": AssertNotEmpty()}
        with pytest.raises(ValueError):
            TransformerPipeline(
                dict_transformers,
                split_function=_split_function,
                **{where: TransformerPipeline([AssertNotEmpty()])}
            )


class TestSplitPipelineDeadEnd:
    """Test a split pipeline with dead-end splits."""

    @pytest.mark.parametrize("splits_no_merge", [("no", "hi"), "no", ["no"]])
    def test_dead_end_error(self, splits_no_merge):
        """Test with wrong 'splits_no_merge' keys."""
        with pytest.raises(KeyError):
            TransformerPipeline(
                {"low": [], "hi": []},
                split_function=_split_function,
                splits_no_merge=splits_no_merge,
            )

    @pytest.mark.parametrize("splits_no_merge", ["hi", ["hi"], ("hi",), {"hi"}])
    def test_dead_end_different_types(self, splits_no_merge):
        """Test with different 'splits_no_merge' types."""
        dict_transformers = {"low": [], "hi": []}

        pipe = TransformerPipeline(
            dict_transformers,
            split_function=_split_function_with_null,
            splits_no_merge=splits_no_merge,
        )

        assert pipe.splits_no_merge == {"hi"}
        pipe.show_pipeline()

    @pytest.mark.parametrize(
        "splits_no_merge", [["hi"], ["hi", "null"], ["low", "hi", "null"]]
    )
    def test_dead_end(self, df_input: pl.DataFrame, splits_no_merge: list):
        """Test with 'splits_no_merge'."""
        ns.clear()

        # Create 3 splits w/o any transformer, just store the dead-end ones
        # in nebula storage.
        dict_transformers = {}
        li_split_names = ["low", "hi", "null"]
        list_splits_to_merge = []

        for split_name in li_split_names:
            if split_name in splits_no_merge:
                li = [{"store": split_name}]
            else:
                li = []
                list_splits_to_merge.append(split_name)
            dict_transformers[split_name] = li

        pipe = TransformerPipeline(
            dict_transformers,
            split_function=_split_function_with_null,
            splits_no_merge=splits_no_merge,
        )

        pipe.show_pipeline()
        df_out = pipe.run(df_input)

        # Check the df output
        full_splits = _split_function_with_null(df_input)
        li_exp_df = [full_splits[k] for k in list_splits_to_merge]
        if li_exp_df:
            df_out_exp = pl.concat(li_exp_df, how="diagonal")
            pl_assert_equal(df_out.sort(df_out.columns), df_out_exp.sort(df_out_exp.columns))

        # Check the dead-end splits that are stored in nebula storage.
        for dead_end_split in splits_no_merge:
            df_chk = ns.get(dead_end_split)
            df_exp = full_splits[dead_end_split]
            pl_assert_equal(df_chk.sort(df_chk.columns), df_exp.sort(df_exp.columns))


class TestSplitPipelineSplitOrder:
    """Test split order functionality."""

    def test_type_error(self):
        """Test that non-string split_order raises TypeError."""
        with pytest.raises(TypeError):
            TransformerPipeline(
                {"a": [], "b": []},
                split_function=lambda df: {"a": df, "b": df},
                split_order=[1, 2],  # Should be strings
            )

    def test_key_error(self):
        """Test that mismatched split_order keys raises KeyError."""
        with pytest.raises(KeyError):
            TransformerPipeline(
                {"a": [], "b": []},
                split_function=lambda df: {"a": df, "b": df},
                split_order=["a", "c"],  # 'c' doesn't exist
            )

    @pytest.mark.parametrize("split_order", [["low", "hi"], ["hi", "low"]])
    def test_basic(self, df_input: pl.DataFrame, split_order: list[str]):
        """Test the split order."""
        dict_splits = {"low": [], "hi": []}

        pipe = TransformerPipeline(
            dict_splits,
            split_function=_split_function,
            split_order=split_order,
        )

        pipe.show_pipeline()
        df_chk = pipe.run(df_input)
        splits = _split_function(df_input)
        df_hi = splits["hi"]
        df_low = splits["low"]

        if split_order[0] == "hi":
            concat = [df_hi, df_low]
        else:
            concat = [df_low, df_hi]

        df_exp = pl.concat(concat, how="vertical")
        pl_assert_equal(df_chk, df_exp)


class TestSplitPipelineMutuallyExclusive:
    """Test mutually exclusive parameters."""

    def test_cast_and_allow_missing_mutually_exclusive(self):
        """Test that cast_subset_to_input_schema and allow_missing_columns are mutually exclusive."""
        with pytest.raises(AssertionError):
            TransformerPipeline(
                {"a": [], "b": []},
                split_function=lambda df: {"a": df, "b": df},
                cast_subset_to_input_schema=True,
                allow_missing_columns=True,
            )


class TestSplitPipelineSingleSplitDictionary:
    """Test single-split dictionary behavior."""

    def test_single_split_becomes_linear(self, df_input: pl.DataFrame):
        """Test that a single-split dictionary becomes a linear pipeline."""
        # When dict has only one key, it should be treated as linear pipeline
        # and split_function should be ignored
        pipe = TransformerPipeline(
            {"only": [Distinct()]},
            split_function=None,  # Can be None for single-split
        )

        df_chk = pipe.run(df_input)
        df_exp = df_input.unique()

        pl_assert_equal(df_chk.sort(df_chk.columns), df_exp.sort(df_exp.columns))


class TestSplitFunctionKeyMismatch:
    """Test split function key mismatch handling."""

    def test_key_mismatch_raises(self, df_input: pl.DataFrame):
        """Test that mismatched keys between split_function and dict_transformers raises KeyError."""

        def _bad_split(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
            return {"a": df, "b": df}  # Keys don't match dict_transformers

        pipe = TransformerPipeline(
            {"x": [], "y": []},  # Different keys
            split_function=_bad_split,
        )

        with pytest.raises(KeyError):
            pipe.run(df_input)


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


class TestSplitPipelineEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_transformers_list(self, df_input: pl.DataFrame):
        """Test split pipeline with empty transformer lists."""
        pipe = TransformerPipeline(
            {"low": [], "hi": []},
            split_function=_split_function,
        )

        df_chk = pipe.run(df_input)

        # Should return the union of both splits unchanged
        split_dfs = _split_function(df_input)
        df_exp = pl.concat([split_dfs["low"], split_dfs["hi"]], how="diagonal")

        pl_assert_equal(df_chk.sort(df_chk.columns), df_exp.sort(df_exp.columns))

    def test_nested_pipeline_in_split(self, df_input: pl.DataFrame):
        """Test that a nested TransformerPipeline can be used within a split."""
        nested_pipe = TransformerPipeline([Distinct()], name="nested")

        pipe = TransformerPipeline(
            {"low": [nested_pipe], "hi": []},
            split_function=_split_function,
        )

        df_chk = pipe.run(df_input)

        # Expected: low split is distinct, hi split unchanged
        split_dfs = _split_function(df_input)
        df_exp = pl.concat(
            [split_dfs["low"].unique(), split_dfs["hi"]],
            how="diagonal"
        )

        pl_assert_equal(df_chk.sort(df_chk.columns), df_exp.sort(df_exp.columns))


@pytest.mark.skipif(os.environ.get("TESTS_NO_SPARK") == "true", reason="no spark")
class TestSpark:

    @staticmethod
    @pytest.fixture(scope="class")
    def df_input_spark(spark):
        fields = [
            StructField("c1", DoubleType(), True),
            StructField("c2", StringType(), True),
            StructField("c3", StringType(), True),
        ]
        return spark.createDataFrame(_DATA, schema=StructType(fields)).persist()

    @pytest.mark.parametrize("repartition", [True, False])
    @pytest.mark.parametrize("coalesce", [True, False])
    def test_repartition_coalesce(self, df_input_spark, repartition, coalesce):
        """Test with repartition / coalesce to original."""
        if repartition and coalesce:  # It raises a (tested) error
            return

        def _c2_null(_df):
            cond = F.col("c2").isNull()
            return {"low": _df.filter(cond), "hi": _df.filter(~cond)}

        df_input_spark = df_input_spark.coalesce(1)
        dict_splits = {"low": [], "hi": []}

        n_part_orig = df_input_spark.rdd.getNumPartitions()

        pipe = TransformerPipeline(
            dict_splits,
            split_function=_c2_null,
            repartition_output_to_original=repartition,
            coalesce_output_to_original=coalesce,
        )

        df_chk = pipe.run(df_input_spark)

        n_part_new = df_chk.rdd.getNumPartitions()

        if repartition or coalesce:
            assert n_part_new == n_part_orig
        else:
            assert n_part_new > n_part_orig
