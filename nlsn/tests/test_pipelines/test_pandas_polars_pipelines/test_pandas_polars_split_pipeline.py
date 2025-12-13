"""Test some pipeline functionalities."""

import pandas as pd
import pytest

from nlsn.nebula import nebula_storage as ns
from nlsn.nebula.pipelines.pipelines import TransformerPipeline
from nlsn.nebula.shared_transformers import Count, Distinct

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
    [_nan, "a", "b"],
    [_nan, "c", "d"],
    [_nan, "c", "d"],
    [_nan, "c", "d"],
]


def _get_df_input():
    """Get input dataframe."""
    return pd.DataFrame(_DATA, columns=["c1", "c2", "c3"])


def _get_cond(df):
    return df["c1"] < 10


def _split_function(df):
    mask = _get_cond(df)
    return {
        "low": df[mask].copy(),
        "hi": df[~mask].copy(),
    }


def _split_function_with_null(df):
    ret = _split_function(df)
    return {**ret, **{"null": df[df["c1"].isna()].copy()}}


class TestSpitPipelineApplyTransformerBeforeAndAfter:
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
    def test_before_and_after(where: str, transformer):
        """Add a transformer after splitting or before appending.

        The 'Distinct' transformer is applied only in the
        'split_apply_before_appending' or 'split_apply_after_splitting' part.
        """
        df_input = _get_df_input()
        dict_transformers = {"low": Count(), "hi": Count()}

        pipe = TransformerPipeline(
            dict_transformers,
            split_function=_split_function,
            cast_subset_to_input_schema=True,
            **{where: transformer}
        )

        pipe.show_pipeline()

        df_chk = pipe.run(df_input)
        df_chk = df_chk.sort_index()
        pd.testing.assert_frame_equal(df_chk, df_input.drop_duplicates())

    @staticmethod
    @pytest.mark.parametrize(
        "where",
        [
            "split_apply_before_appending",
            "split_apply_after_splitting",
        ],
    )
    def test_error(where: str):
        """Pass a TransformerPipeline instead of a transformer."""
        dict_transformers = {"low": Count(), "hi": Count()}
        with pytest.raises(ValueError):
            TransformerPipeline(
                dict_transformers,
                split_function=_split_function,
                **{where: TransformerPipeline([Distinct()])}
            )


class TestSpitPipelineDeadEnd:
    """Test a split pipeline with dead-end splits."""

    @staticmethod
    @pytest.mark.parametrize("splits_no_merge", [("no", "hi"), "no", ["no"]])
    def test_dead_end_error(splits_no_merge):
        """Test TransformerPipeline pipeline with wrong 'splits_no_merge' keys."""
        with pytest.raises(KeyError):
            TransformerPipeline(
                {"low": [], "hi": []},
                split_function=_split_function,
                splits_no_merge=splits_no_merge,
            )

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
    @pytest.mark.parametrize(
        "splits_no_merge", [["hi"], ["hi", "null"], ["low", "hi", "null"]]
    )
    def test_dead_end(splits_no_merge: list):
        """Test TransformerPipeline pipeline with 'splits_no_merge'."""
        ns.clear()

        df_input = _get_df_input()
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
        full_splits = _split_function_with_null(_get_df_input())
        li_exp_df = [full_splits[k] for k in list_splits_to_merge]
        if li_exp_df:
            df_out_exp = pd.concat(li_exp_df, axis=0)
            pd.testing.assert_frame_equal(df_out, df_out_exp)

        # Check the dead-end splits that are stored in nebula storage.
        for dead_end_split in splits_no_merge:
            df_chk = ns.get(dead_end_split)
            df_exp = full_splits[dead_end_split]
            pd.testing.assert_frame_equal(df_chk, df_exp)
