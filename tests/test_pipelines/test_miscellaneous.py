"""Miscellaneous tests."""

import polars as pl
import pytest

from nebula import TransformerPipeline
from nebula import nebula_storage as ns
from nebula.transformers import SelectColumns, AssertNotEmpty, DropNulls
from .auxiliaries import NoParentClass, CallMe, ThisTransformerIsBroken
from ..auxiliaries import pl_assert_equal


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
    pipe.show()

    df_chk = pipe.run(df_input)
    pl_assert_equal(df_chk, df_input)


def test_forced_transformer(df_input):
    """Ensure the forced transformer is executed after each transformer."""
    list_trf_1 = [SelectColumns(glob="*")]
    list_trf_2 = [AssertNotEmpty(), DropNulls()]

    pipe_1 = TransformerPipeline(list_trf_1)
    pipe_2 = TransformerPipeline(list_trf_2)
    pipe = TransformerPipeline([pipe_1, pipe_2])
    pipe.show()

    ns.clear()
    df_chk = pipe.run(df_input, force_interleaved_transformer=CallMe())
    n: int = ns.get("_call_me_")
    exp_n = len(list_trf_1) + len(list_trf_2)
    assert n == exp_n
    ns.clear()

    pl_assert_equal(df_chk, df_input)


def test_skip_pipeline(df_input):
    """Ensure the skipped pipeline is not executed."""
    pipe_1 = TransformerPipeline(SelectColumns(glob="c*"))
    pipe_2 = TransformerPipeline(ThisTransformerIsBroken, skip=True)
    pipe = TransformerPipeline([pipe_1, pipe_2])
    pipe.show(add_params=True)
    df_chk = pipe.run(df_input, force_interleaved_transformer=CallMe())
    pl_assert_equal(df_chk, df_input.drop("idx"))
