"""Miscellaneous tests."""

import polars as pl
import pytest

from nlsn.nebula import nebula_storage as ns
from nlsn.nebula.pipelines.pipelines import TransformerPipeline
from nlsn.nebula.transformers import SelectColumns, AssertNotEmpty, DropNulls
from nlsn.tests.test_pipelines.auxiliaries import NoParentClass, pl_assert_equal, CallMe


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
    pipe.show_pipeline()

    df_chk = pipe.run(df_input)
    pl_assert_equal(df_chk, df_input)


def test_forced_transformer(df_input):
    """Ensure the forced transformer is executed after each transformer."""
    list_trf_1 = [SelectColumns(glob="*")]
    list_trf_2 = [AssertNotEmpty(), DropNulls()]

    pipe_1 = TransformerPipeline(list_trf_1)
    pipe_2 = TransformerPipeline(list_trf_2)
    pipe = TransformerPipeline([pipe_1, pipe_2])
    pipe.show_pipeline()

    ns.clear()
    df_chk = pipe.run(df_input, CallMe())
    n: int = ns.get("_call_me_")
    exp_n = len(list_trf_1) + len(list_trf_2)
    assert n == exp_n
    ns.clear()

    pl_assert_equal(df_chk, df_input)
