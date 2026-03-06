"""Miscellaneous tests."""

import polars as pl
import pytest

from nebula import TransformerPipeline
from nebula import nebula_storage as ns
from nebula.transformers import AssertNotEmpty, DropNulls, SelectColumns

from ..auxiliaries import pl_assert_equal
from .auxiliaries import CallMe, NoParentClass, ThisTransformerIsBroken


@pytest.fixture(scope="module", name="df_input")
def _get_df_input():
    data = [
        [0, "a0", "b0"],
        [1, "a1", "b1"],
        [2, "a2", "b2"],
    ]
    return pl.DataFrame(data, schema=["idx", "c1", "c2"], orient="row")


def test_transformer_no_parent_class_in_pipeline(df_input):
    """Test a pipeline with a transformer without a known parent class."""
    pipe = TransformerPipeline(NoParentClass())

    df_chk = pipe.run(df_input)
    pl_assert_equal(df_chk, df_input)


def test_after_each_step_with_transformer(df_input):
    """Ensure after_each_step transformer is executed after each step."""
    list_trf_1 = [SelectColumns(glob="*")]
    list_trf_2 = [AssertNotEmpty(), DropNulls()]

    pipe_1 = TransformerPipeline(list_trf_1)
    pipe_2 = TransformerPipeline(list_trf_2)
    pipe = TransformerPipeline([pipe_1, pipe_2])

    df_chk = pipe.run(df_input, after_each_step=CallMe())
    n: int = ns.get("_call_me_")
    assert n == len(list_trf_1) + len(list_trf_2)
    pl_assert_equal(df_chk, df_input)


def test_after_each_step_with_function(df_input):
    """Ensure after_each_step works with a plain function."""
    call_count = {"n": 0}

    def count_calls(df):
        call_count["n"] += 1
        return df

    list_trf = [SelectColumns(glob="*"), AssertNotEmpty(), DropNulls()]
    pipe = TransformerPipeline(list_trf)

    df_chk = pipe.run(df_input, after_each_step=count_calls)

    assert call_count["n"] == len(list_trf)
    pl_assert_equal(df_chk, df_input)


def test_skip_pipeline(df_input):
    """Ensure the skipped pipeline is not executed."""
    pipe_1 = TransformerPipeline(SelectColumns(glob="c*"))
    pipe_2 = TransformerPipeline(ThisTransformerIsBroken, skip=True)
    pipe = TransformerPipeline([pipe_1, pipe_2])
    df_chk = pipe.run(df_input, after_each_step=CallMe())
    pl_assert_equal(df_chk, df_input.drop("idx"))


# ---------------------------------------------------------------------------
# to_string / printer / hooks coverage
# ---------------------------------------------------------------------------


def _dummy_fn(df):
    return df


def _fn_with_args(df, x, y, z=10):
    return df


class TestToString:
    """Test pipeline to_string and show, covering printer visit paths."""

    def test_flat_pipeline_to_string(self, df_input):
        pipe = TransformerPipeline(
            [SelectColumns(glob="*"), AssertNotEmpty()],
            name="my-flat",
        )
        s = pipe.to_string()
        assert "my-flat" in s
        assert "SelectColumns" in s

    def test_flat_pipeline_to_string_with_params(self, df_input):
        pipe = TransformerPipeline(
            [SelectColumns(glob="*"), DropNulls()],
            name="flat-params",
        )
        s = pipe.to_string(add_params=True)
        assert "flat-params" in s
        assert "PARAMS:" in s

    def test_pipeline_with_functions_to_string(self):
        pipe = TransformerPipeline(
            [
                _dummy_fn,
                (_fn_with_args, [1, 2], {"z": 99}, "my description"),
            ]
        )
        s = pipe.to_string(add_params=True)
        assert "_dummy_fn" in s
        assert "_fn_with_args" in s
        assert "Description" in s

    def test_pipeline_with_storage_to_string(self):
        pipe = TransformerPipeline(
            [
                AssertNotEmpty(),
                {"store": "test_key"},
            ]
        )
        s = pipe.to_string()
        assert "Store" in s

    def test_pipeline_with_conversion_to_string(self):
        pipe = TransformerPipeline(["to_native", "from_native", AssertNotEmpty()])
        s = pipe.to_string()
        assert "native" in s.lower()

    def test_conversion_node_display_messages(self):
        from nebula.pipelines.ir.nodes import ConversionNode

        assert "Collect" in ConversionNode(operation="collect").display_message
        assert "lazy" in ConversionNode(operation="to_lazy").display_message
        assert "native" in ConversionNode(operation="to_native").display_message.lower()
        assert "Narwhals" in ConversionNode(operation="from_native").display_message

    def test_nested_pipeline_to_string(self, df_input):
        inner = TransformerPipeline([AssertNotEmpty()], name="inner")
        outer = TransformerPipeline([inner, SelectColumns(glob="*")], name="outer")
        s = outer.to_string()
        assert "outer" in s
        assert "AssertNotEmpty" in s

    def test_split_pipeline_to_string(self):
        import narwhals as nw

        from nebula.nw_util import null_cond_to_false

        def my_split(df):
            cond = nw.col("c1") < 5
            return {"low": df.filter(cond), "hi": df.filter(~null_cond_to_false(cond))}

        pipe = TransformerPipeline(
            {"low": [AssertNotEmpty()], "hi": [AssertNotEmpty()]},
            split_function=my_split,
            name="split-pipe",
        )
        s = pipe.to_string(add_params=True)
        assert "SPLIT" in s
        assert "low" in s
        assert "hi" in s

    def test_apply_to_rows_to_string(self):
        from nebula.transformers import AddLiterals

        pipe = TransformerPipeline(
            [AddLiterals(data=[{"value": "x", "alias": "c_new"}])],
            apply_to_rows={"input_col": "idx", "operator": "gt", "value": 1},
        )
        s = pipe.to_string()
        assert "APPLY TO ROWS" in s

    def test_apply_to_rows_otherwise_to_string(self):
        from nebula.transformers import AddLiterals

        pipe = TransformerPipeline(
            [AddLiterals(data=[{"value": "matched", "alias": "tag"}])],
            apply_to_rows={"input_col": "idx", "operator": "gt", "value": 1},
            otherwise=AddLiterals(data=[{"value": "other", "alias": "tag"}]),
        )
        s = pipe.to_string()
        assert "Otherwise" in s

    def test_dead_end_to_string(self):
        from nebula.transformers import AddLiterals

        pipe = TransformerPipeline(
            [AddLiterals(data=[{"value": True, "alias": "flag"}])],
            apply_to_rows={"input_col": "idx", "operator": "gt", "value": 1, "dead-end": True},
        )
        s = pipe.to_string()
        assert "Dead End" in s

    def test_branch_from_storage_to_string(self):
        pipe = TransformerPipeline(
            [AssertNotEmpty()],
            branch={"storage": "my_stored_df", "end": "dead-end"},
        )
        s = pipe.to_string()
        assert "BRANCH" in s
        assert "storage" in s.lower()

    def test_branch_join_to_string(self):
        pipe = TransformerPipeline(
            [AssertNotEmpty()],
            branch={"end": "join", "on": "idx", "how": "left"},
        )
        s = pipe.to_string()
        assert "Join" in s

    def test_transformer_description_to_string(self):
        pipe = TransformerPipeline([(AssertNotEmpty(), "Ensure we have data")])
        s = pipe.to_string()
        assert "Description" in s

    def test_merge_config_params_to_string(self):
        pipe = TransformerPipeline(
            [AssertNotEmpty()],
            branch={"end": "join", "on": "idx", "how": "left"},
        )
        s = pipe.to_string(add_params=True)
        assert "Join" in s


class TestShowParamsHooks:
    """Run with show_params=True to exercise hooks on_node_start param paths."""

    def test_show_params_transformer(self, df_input):
        pipe = TransformerPipeline([SelectColumns(glob="*")])
        df_chk = pipe.run(df_input, show_params=True)
        pl_assert_equal(df_chk, df_input)

    def test_show_params_function_with_args(self, df_input):
        pipe = TransformerPipeline(
            [
                (_fn_with_args, [1, 2], {"z": 5}, "test fn"),
            ]
        )
        df_chk = pipe.run(df_input, show_params=True)
        pl_assert_equal(df_chk, df_input)
