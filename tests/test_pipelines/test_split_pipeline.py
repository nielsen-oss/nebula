"""Test split pipeline functionalities using Polars."""

import os

import polars as pl
import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from nebula import TransformerPipeline
from nebula.storage import nebula_storage as ns
from nebula.transformers import AddLiterals, AssertNotEmpty, Cast

from ..auxiliaries import pl_assert_equal
from .auxiliaries import CallMe, Distinct, RoundValues, ThisTransformerIsBroken

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

_trf_low: list = [RoundValues(column="c1", precision=3)]
_trf_hi: list = [
    RoundValues(column="c1", precision=1),
    Cast(cast={"c1": "float64"}),
]


@pytest.fixture(scope="module")
def df_input() -> pl.DataFrame:
    schema = {"c1": pl.Float64, "c2": pl.String, "c3": pl.String}
    return pl.DataFrame(_DATA, schema=schema, orient="row")


def _split_function(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Split dataframe into 'low' (c1 < 10) and 'hi' (c1 >= 10) subsets."""
    cond = pl.col("c1") < 10
    cond_not_null = pl.col("c1").is_not_null() & pl.col("c1").is_not_nan()
    return {
        "low": df.filter(cond & cond_not_null),
        "hi": df.filter(~cond & cond_not_null),
    }


def _split_function_with_null(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Split dataframe into 'low', 'hi', and 'null' subsets."""
    ret = _split_function(df)
    cond_null = pl.col("c1").is_null() | pl.col("c1").is_nan()
    return {**ret, "null": df.filter(cond_null)}


class TestSplitPipeline:
    """Generic tests for split pipelines."""

    @pytest.mark.parametrize("name", [None, "name_01"])
    def test_basic(self, df_input: pl.DataFrame, name: str | None):
        dict_splits = {"low": _trf_low, "hi": _trf_hi}

        pipe = TransformerPipeline(
            dict_splits,
            split_function=_split_function,
            interleaved=[AssertNotEmpty()],
            name=name,
        )

        df_chk = pipe.run(df_input)

        split_dfs = _split_function(df_input)
        df_low = split_dfs["low"]
        df_hi = split_dfs["hi"]

        for t in _trf_low:
            df_low = t.transform(df_low)
        for t in _trf_hi:
            df_hi = t.transform(df_hi)

        df_exp = pl.concat([df_low, df_hi], how="vertical")
        pl_assert_equal(df_chk.sort(df_chk.columns), df_exp.sort(df_exp.columns))

    def test_cast_subsets_to_input_schema(self, df_input):
        dict_splits = {"low": [], "hi": [Cast(cast={"c1": "float32"})], "null": []}

        pipe = TransformerPipeline(
            dict_splits,
            split_function=_split_function_with_null,
            cast_subsets_to_input_schema=True,
        )

        df_chk = pipe.run(df_input)
        assert df_chk["c1"].dtype == pl.Float64
        pl_assert_equal(df_chk.sort(df_chk.columns), df_input.sort(df_input.columns))

    def test_allow_missing_columns(self, df_input: pl.DataFrame):
        t_new_col = AddLiterals(data=[{"value": "hello", "alias": "new_column"}])
        dict_splits = {"low": [], "hi": [t_new_col]}

        pipe = TransformerPipeline(
            dict_splits,
            split_function=_split_function,
            allow_missing_columns=True,
        )
        df_chk = pipe.run(df_input)

        dict_df = _split_function(df_input)
        df_low = dict_df["low"]
        df_hi = dict_df["hi"].with_columns(pl.lit("hello").alias("new_column"))
        df_exp = pl.concat([df_low, df_hi], how="diagonal")
        pl_assert_equal(df_chk.sort(df_chk.columns), df_exp.sort(df_exp.columns))

    @pytest.mark.parametrize("splits_skip_if_empty", [None, "hi", {"hi"}, {"hi", "low"}])
    def test_skip_if_empty(self, df_input: pl.DataFrame, splits_skip_if_empty):
        def _mock_split(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
            return {"low": df, "hi": df.head(0)}

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
        dict_splits = {"low": [Distinct(), Distinct()], "hi": Distinct()}

        pipe = TransformerPipeline(
            dict_splits,
            split_function=_split_function,
            interleaved=interleaved,
            prepend_interleaved=prepend,
            append_interleaved=append,
            allow_missing_columns=True,
        )

        df_chk = pipe.run(df_input)

        df_exp = df_input.filter(pl.col("c1").is_not_null() & pl.col("c1").is_not_nan()).unique()

        if interleaved:
            n_chk = ns.get("_call_me_")
            n_exp = 1
            if prepend:
                n_exp += 2
            if append:
                n_exp += 2
            assert n_chk == n_exp
        else:
            assert "_call_me_" not in ns.list_keys()

        pl_assert_equal(df_chk.sort(df_chk.columns), df_exp.sort(df_chk.columns))


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
        dict_transformers = {"low": AssertNotEmpty(), "hi": AssertNotEmpty()}

        pipe = TransformerPipeline(
            dict_transformers,
            split_function=_split_function,
            cast_subsets_to_input_schema=True,
            **{where: transformer},
        )

        df_chk = pipe.run(df_input)

        split_dfs = _split_function(df_input)
        df_exp = pl.concat([split_dfs["low"].unique(), split_dfs["hi"].unique()], how="diagonal")
        pl_assert_equal(df_chk.sort(df_chk.columns), df_exp.sort(df_exp.columns))

    @pytest.mark.parametrize(
        "where",
        ["split_apply_before_appending", "split_apply_after_splitting"],
    )
    def test_error_pipeline_instead_of_transformer(self, where: str):
        dict_transformers = {"low": AssertNotEmpty(), "hi": AssertNotEmpty()}
        with pytest.raises(TypeError):
            TransformerPipeline(
                dict_transformers,
                split_function=_split_function,
                **{where: TransformerPipeline([AssertNotEmpty()])},
            )


class TestSplitPipelineDeadEnd:
    """Test a split pipeline with dead-end splits."""

    @pytest.mark.parametrize("splits_no_merge", [("no", "hi"), "no", ["no"]])
    def test_dead_end_error(self, splits_no_merge):
        with pytest.raises(KeyError):
            TransformerPipeline(
                {"low": [], "hi": []},
                split_function=_split_function,
                splits_no_merge=splits_no_merge,
            )

    @pytest.mark.parametrize("splits_no_merge", ["hi", ["hi"], ("hi",), {"hi"}])
    def test_splits_no_merge(self, df_input, splits_no_merge):
        dict_transformers = {"low": [], "hi": [], "null": []}

        pipe = TransformerPipeline(
            dict_transformers,
            split_function=_split_function_with_null,
            splits_no_merge=splits_no_merge,
        )

        df_chk = pipe.run(df_input)
        df_exp = df_input.filter((pl.col("c1") < 10) | pl.col("c1").is_null() | pl.col("c1").is_nan())
        pl_assert_equal(df_chk, df_exp, ["c1"])

    @pytest.mark.parametrize("splits_no_merge", [["hi"], ["hi", "null"], ["low", "hi", "null"]])
    def test_dead_end(self, df_input: pl.DataFrame, splits_no_merge: list):
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

        df_out = pipe.run(df_input)

        full_splits = _split_function_with_null(df_input)
        li_exp_df = [full_splits[k] for k in list_splits_to_merge]
        if li_exp_df:
            df_out_exp = pl.concat(li_exp_df, how="diagonal")
            pl_assert_equal(df_out.sort(df_out.columns), df_out_exp.sort(df_out_exp.columns))

        for dead_end_split in splits_no_merge:
            df_chk = ns.get(dead_end_split)
            df_exp = full_splits[dead_end_split]
            pl_assert_equal(df_chk.sort(df_chk.columns), df_exp.sort(df_exp.columns))


class TestSplitPipelineSplitOrder:
    """Test split order functionality."""

    def test_type_error(self):
        with pytest.raises(TypeError):
            TransformerPipeline(
                {"a": [], "b": []},
                split_function=lambda df: {"a": df, "b": df},
                split_order=[1, 2],
            )

    def test_key_error(self):
        with pytest.raises(KeyError):
            TransformerPipeline(
                {"a": [], "b": []},
                split_function=lambda df: {"a": df, "b": df},
                split_order=["a", "c"],
            )

    @pytest.mark.parametrize("split_order", [["low", "hi"], ["hi", "low"]])
    def test_basic(self, df_input: pl.DataFrame, split_order: list[str]):
        dict_splits = {"low": [], "hi": []}

        pipe = TransformerPipeline(
            dict_splits,
            split_function=_split_function,
            split_order=split_order,
        )

        df_chk = pipe.run(df_input)
        splits = _split_function(df_input)

        if split_order[0] == "hi":
            concat = [splits["hi"], splits["low"]]
        else:
            concat = [splits["low"], splits["hi"]]

        df_exp = pl.concat(concat, how="vertical")
        pl_assert_equal(df_chk, df_exp)


def test_cast_and_allow_missing_mutually_exclusive():
    """Test that cast_subsets_to_input_schema and allow_missing_columns are mutually exclusive."""
    with pytest.raises(AssertionError):
        TransformerPipeline(
            {"a": [], "b": []},
            split_function=lambda df: {"a": df, "b": df},
            cast_subsets_to_input_schema=True,
            allow_missing_columns=True,
        )


def test_invalid_single_split(df_input: pl.DataFrame):
    """Ensure that single-split pipelines are rejected."""
    with pytest.raises(ValueError):
        TransformerPipeline({"x": Distinct()}, split_function=lambda x: x)


def test_key_mismatch_raises(df_input: pl.DataFrame):
    """Test that mismatched keys between split_function and dict_transformers raises KeyError."""

    def _bad_split(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
        return {"a": df, "b": df}

    pipe = TransformerPipeline(
        {"x": [], "y": []},
        split_function=_bad_split,
    )

    with pytest.raises(KeyError):
        pipe.run(df_input)


class TestSplitPipelineEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_transformers_list(self, df_input: pl.DataFrame):
        pipe = TransformerPipeline(
            {"low": [], "hi": []},
            split_function=_split_function,
        )

        df_chk = pipe.run(df_input)

        split_dfs = _split_function(df_input)
        df_exp = pl.concat([split_dfs["low"], split_dfs["hi"]], how="diagonal")
        pl_assert_equal(df_chk.sort(df_chk.columns), df_exp.sort(df_exp.columns))

    def test_nested_pipeline_in_split(self, df_input: pl.DataFrame):
        nested_pipe = TransformerPipeline([Distinct(), [Distinct()]], name="nested")

        pipe = TransformerPipeline(
            {"low": [nested_pipe], "hi": []},
            split_function=_split_function,
        )

        df_chk = pipe.run(df_input)

        split_dfs = _split_function(df_input)
        df_exp = pl.concat([split_dfs["low"].unique(), split_dfs["hi"]], how="diagonal")
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
        if repartition and coalesce:
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
