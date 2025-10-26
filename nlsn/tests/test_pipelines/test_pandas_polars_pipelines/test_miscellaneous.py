"""Miscellaneous pandas pipelines tests."""

from typing import Optional

import pandas as pd
import pytest

from nlsn.nebula import nebula_storage as ns
from nlsn.nebula.base import Transformer
from nlsn.nebula.pipelines.pipelines import TransformerPipeline
from nlsn.nebula.shared_transformers import Count, PrintSchema, SelectColumns


class AddOne:
    def __init__(self, *, column: str):  # noqa: D107
        self._col: str = column

    def transform(self, df):  # noqa: D102
        df[self._col] += 1
        return df


class ForcedTransformer(Transformer):
    def __init__(self):
        """Force interleaved transformer."""
        super().__init__()

    def _transform(self, df):
        n = ns.get("num_forced_transformer")
        n += 1
        ns.set("num_forced_transformer", n)
        return df


@pytest.fixture(scope="module", name="df_input")
def _get_df_input():
    data = [
        [0, "a0", "b0"],
        [1, "a1", "b1"],
        [2, "a2", "b2"],
    ]
    return pd.DataFrame(data, columns=["idx", "c1", "c2"])


def test_transformer_no_parent_class_in_pipeline(df_input):
    """Test a pipeline with a transformer without a known parent class."""
    df_exp = df_input.copy()
    df_exp["idx"] += 1

    pipe = TransformerPipeline(AddOne(column="idx"))
    pipe.show_pipeline()
    pipe._print_dag()

    df_chk = pipe.run(df_input)
    pd.testing.assert_frame_equal(df_exp, df_chk)


@pytest.mark.parametrize("backend", ["pandas", None])
def test_backend_pandas(df_input, backend: Optional[str]):
    """Test explicit and implicit pandas backend."""
    pipe = TransformerPipeline(SelectColumns(glob="c*"), backend=backend)
    pipe.show_pipeline()
    pipe._print_dag()

    df_chk = pipe.run(df_input)
    df_exp = df_input[["c1", "c2"]]
    pd.testing.assert_frame_equal(df_chk, df_exp)


def test_forced_transformer(df_input):
    """Ensure the forced transformer is executed after each transformer."""
    list_trf_1 = [SelectColumns(glob="*")]
    list_trf_2 = [Count(), PrintSchema()]

    pipe_1 = TransformerPipeline(list_trf_1, backend="pandas")
    pipe_2 = TransformerPipeline(list_trf_2)
    pipe = TransformerPipeline([pipe_1, pipe_2], backend="pandas")
    pipe.show_pipeline()

    ns.clear()
    ns.set("num_forced_transformer", 0)
    df_chk = pipe.run(df_input, ForcedTransformer())
    n: int = ns.get("num_forced_transformer")
    exp_n = len(list_trf_1) + len(list_trf_2)
    assert n == exp_n
    ns.clear()

    pd.testing.assert_frame_equal(df_chk, df_input)
