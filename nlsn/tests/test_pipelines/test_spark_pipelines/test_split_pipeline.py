"""Test a complex pipeline using the functionality 'split_apply_before_appending'."""

import os
from functools import reduce

import pytest
from chispa import assert_df_equality
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from nlsn.nebula import nebula_storage as ns
from nlsn.nebula.pipelines.pipelines import TransformerPipeline
from nlsn.nebula.shared_transformers import RoundValues, WithColumn
from nlsn.nebula.spark_transformers import Cache, Cast, Distinct, LogDataSkew, NanToNull
from nlsn.nebula.spark_util import null_cond_to_false
from nlsn.tests.test_pipelines._shared import RuntimeErrorTransformer

_trf_low: list = [RoundValues(input_columns="c1", precision=3)]
_trf_hi: list = [
    RoundValues(input_columns="c1", precision=1),
    NanToNull(glob="*"),
    Cast(cast={"c1": "float"}),
]

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


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    """Get input dataframe."""
    fields = [
        StructField("c1", DoubleType(), True),
        StructField("c2", StringType(), True),
        StructField("c3", StringType(), True),
    ]
    return spark.createDataFrame(_DATA, schema=StructType(fields)).persist()


def _get_cond():
    return F.col("c1") < 10


def _split_function(df):
    cond = _get_cond()
    return {
        "low": df.filter(cond),
        "hi": df.filter(~null_cond_to_false(cond)),
    }


def _split_function_with_null(df):
    ret = _split_function(df)
    return {**ret, **{"null": df.filter(F.col("c1").isNull())}}


class TestSplitPipeline:
    """Generic tests for split pipelines."""

    _INTERLEAVED = [Cache(), LogDataSkew()]
    _APPLY_BEFORE_APPENDING = Distinct()

    @pytest.fixture(scope="class", name="df_exp")
    def _get_df_exp(self, df_input):
        """Get the expected dataframe."""
        _cond = _get_cond()
        df_low = df_input.filter(_cond)
        df_hi = df_input.filter(~null_cond_to_false(_cond))

        for t in _trf_low + [self._APPLY_BEFORE_APPENDING]:
            df_low = t.transform(df_low)

        for t in _trf_hi + [self._APPLY_BEFORE_APPENDING]:
            df_hi = t.transform(df_hi)

        df_ret = df_low.unionByName(df_hi).cache()
        # "interleaved" transformers are just for log in this unit-test.
        return df_ret

    @pytest.mark.parametrize(
        "name, split_order, interleaved, prepend_interleaved, append_interleaved, cast_subset_to_input_schema",
        [
            (None, None, None, True, False, False),
            (None, [], [], True, False, True),
            ("name_01", None, _INTERLEAVED, True, False, False),
            ("name_02", ["low", "hi"], _INTERLEAVED, True, True, True),
            ("name_03", ["hi", "low"], _INTERLEAVED, True, True, True),
        ],
    )
    def test_split_pipeline(
        self,
        df_input,
        df_exp,
        name: str,
        split_order,
        interleaved: list,
        prepend_interleaved: bool,
        append_interleaved: bool,
        cast_subset_to_input_schema,
    ):
        """Test TransformerPipeline pipeline."""
        dict_splits = {"low": _trf_low, "hi": _trf_hi}

        pipe = TransformerPipeline(
            dict_splits,
            split_function=_split_function,
            name=name,
            split_order=split_order,
            interleaved=interleaved,
            prepend_interleaved=prepend_interleaved,
            append_interleaved=append_interleaved,
            split_apply_before_appending=self._APPLY_BEFORE_APPENDING,
            cast_subset_to_input_schema=cast_subset_to_input_schema,
        )

        pipe.show_pipeline()

        df_chk = pipe.run(df_input)

        if cast_subset_to_input_schema:
            # df_exp.c1 became <double> after the UnionByName.
            df_exp = df_exp.withColumn("c1", F.col("c1").cast(DoubleType()))

        assert_df_equality(
            df_chk, df_exp, ignore_row_order=True, allow_nan_equality=True
        )

    @pytest.mark.parametrize("repartition", [True, False])
    @pytest.mark.parametrize("coalesce", [True, False])
    def test_repartition_coalesce(self, df_input, df_exp, repartition, coalesce):
        """Test TransformerPipeline pipeline with repartition / coalesce to original."""
        if repartition and coalesce:
            # It raises a (tested) error
            return

        df_input = df_input.coalesce(1)
        dict_splits = {"low": _trf_low, "hi": _trf_hi}

        n_part_orig = df_input.rdd.getNumPartitions()

        pipe = TransformerPipeline(
            dict_splits,
            split_function=_split_function,
            split_apply_before_appending=self._APPLY_BEFORE_APPENDING,
            repartition_output_to_original=repartition,
            coalesce_output_to_original=coalesce,
        )

        df_chk = pipe.run(df_input)

        # Both dfs have c1 <double>
        assert_df_equality(
            df_chk, df_exp, ignore_row_order=True, allow_nan_equality=True
        )

        n_part_new = df_chk.rdd.getNumPartitions()

        if repartition or coalesce:
            assert n_part_new == n_part_orig
        else:
            assert n_part_new > n_part_orig

    @staticmethod
    def test_allow_missing_columns(df_input):
        """Test TransformerPipeline with allow_missing_columns=True."""
        t_new_col = WithColumn(column_name="new_column", value="new_value")
        dict_splits = {"low": [], "hi": [t_new_col]}

        pipe = TransformerPipeline(
            dict_splits, split_function=_split_function, allow_missing_columns=True
        )
        df_chk = pipe.run(df_input)

        dict_df = _split_function(df_input)
        df_low = dict_df["low"]
        df_hi = dict_df["hi"].withColumn("new_column", F.lit("new_value"))
        df_exp = df_low.unionByName(df_hi, allowMissingColumns=True)

        assert_df_equality(
            df_chk, df_exp, ignore_row_order=True, allow_nan_equality=True
        )

    @pytest.mark.parametrize(
        "splits_skip_if_empty", [None, "hi", {"hi"}, {"hi", "low"}]
    )
    def test_splits_skip_if_empty(self, df_input, splits_skip_if_empty):
        """Test TransformerPipeline pipeline with splits_skip_if_empty argument."""

        def _mock_split(df):
            return {"low": df, "hi": df.limit(0)}

        pipe = TransformerPipeline(
            {"low": [], "hi": [RuntimeErrorTransformer()]},
            split_function=_mock_split,
            splits_skip_if_empty=splits_skip_if_empty,
        )

        if splits_skip_if_empty in {None, "low"}:
            with pytest.raises(RuntimeError):
                pipe.run(df_input)
            return

        df_chk = pipe.run(df_input)
        assert_df_equality(
            df_chk, df_input, ignore_row_order=True, allow_nan_equality=True
        )


@pytest.mark.skipif(os.getenv("full_nebula_test") != "true", reason="tested in pandas")
class TestSplitPipelineApplyTransformerBeforeAndAfter:
    """Test adding transformers after splitting or before appending."""

    @staticmethod
    @pytest.mark.parametrize(
        "where, transformer",
        [
            ("split_apply_before_appending", Distinct()),
            ("split_apply_before_appending", [Distinct()]),
            ("split_apply_after_splitting", (Distinct(),)),
        ],
    )
    def test_before_and_after(df_input, where: str, transformer):
        """Add a transformer after splitting or before appending.

        The 'Distinct' transformer is applied only in the
        'split_apply_before_appending' or 'split_apply_after_splitting' part.
        """
        dict_transformers = {"low": Cache(), "hi": Cache()}

        pipe = TransformerPipeline(
            dict_transformers,
            split_function=_split_function,
            cast_subset_to_input_schema=True,
            **{where: transformer}
        )

        pipe.show_pipeline()

        df_chk = pipe.run(df_input)
        assert_df_equality(
            df_chk, df_input.distinct(), ignore_row_order=True, allow_nan_equality=True
        )


@pytest.mark.skipif(os.getenv("full_nebula_test") != "true", reason="tested in pandas")
class TestSplitPipelineDeadEnd:
    """Test a split pipeline with dead-end splits."""

    @staticmethod
    @pytest.mark.parametrize("splits_no_merge", ["hi", ["hi"], ("hi",), {"hi"}])
    def test_dead_end_different_types(splits_no_merge):
        """Test TransformerPipeline pipeline with different 'splits_no_merge' types."""
        dict_transformers = {"low": [], "hi": []}

        pipe = TransformerPipeline(
            dict_transformers,
            split_function=_split_function_with_null,
            splits_no_merge=splits_no_merge,
        )

        assert pipe.splits_no_merge == {"hi"}
        pipe.show_pipeline()

    @staticmethod
    @pytest.fixture(scope="module", name="full_splits")
    def _get_full_splits(df_input) -> dict:
        return _split_function_with_null(df_input)

    @staticmethod
    @pytest.mark.parametrize(
        "splits_no_merge", [["hi"], ["hi", "null"], ["low", "hi", "null"]]
    )
    def test_dead_end(df_input, full_splits, splits_no_merge: list):
        """Test TransformerPipeline pipeline with 'splits_no_merge'."""
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

        # To catch any error
        pipe.show_pipeline()

        df_out = pipe.run(df_input)

        # Check the df output
        li_exp_df = [full_splits[k] for k in list_splits_to_merge]
        if li_exp_df:
            df_out_exp = reduce(lambda df1, df2: df1.unionByName(df2), li_exp_df)
            assert_df_equality(
                df_out, df_out_exp, ignore_row_order=True, allow_nan_equality=True
            )

        # Check the dead-end splits that are stored in nebula storage.
        for dead_end_split in splits_no_merge:
            df_chk = ns.get(dead_end_split)
            df_exp = full_splits[dead_end_split]
            assert_df_equality(
                df_chk, df_exp, ignore_row_order=True, allow_nan_equality=True
            )
