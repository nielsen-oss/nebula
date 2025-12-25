"""Test a simple flat pipeline."""

import polars as pl
import pytest

from nebula import TransformerPipeline
from nebula.transformers import *
from ..auxiliaries import pl_assert_equal

_TRANSFORMERS = [
    SelectColumns(glob="*"),
    DropNulls(glob="*", drop_na=True),
]

_INTERLEAVED = [
    AssertContainsColumns(columns="idx"),
    AssertNotEmpty(),
]

_nan = float("nan")


@pytest.fixture(scope="module", name="df_input")
def _get_df_input():
    data = [
        [0, "a", "b"],
        [1, "a", "  b"],
        [2, "  a  ", "  b  "],
        [3, "", ""],
        [4, "   ", "   "],
        [5, None, None],
        [6, " ", None],
        [7, "", None],
        [8, "a", None],
        [9, "a", ""],
        [10, "   ", "b"],
        [11, "a", None],
        [12, None, "b"],
        [13, _nan, "b"],
        [14, _nan, None],
        [15, _nan, _nan],
    ]
    return pl.DataFrame(data, schema=["idx", "c1", "c2"])


@pytest.fixture(scope="module", name="df_exp")
def _get_df_exp(df_input):
    """Get the expected dataframe."""
    df_ret = df_input
    for t in _TRANSFORMERS:
        df_ret = t.transform(df_ret)
    # "interleaved" transformers are just for log in this unit-test.
    return df_ret.persist()


@pytest.mark.parametrize(
    "interleaved, prepend_interleaved, append_interleaved, name",
    [
        (None, True, False, None),
        ([], True, False, None),
        (_INTERLEAVED, True, False, "name_01"),
        (_INTERLEAVED, True, True, "name_02"),
        (_INTERLEAVED[0], False, False, "name_03"),
    ],
)
def test_pipeline_flat_list_transformers(
        df_input: pl.DataFrame,
        interleaved: list,
        prepend_interleaved: bool,
        append_interleaved: bool,
        name: str,
):
    """Test TransformerPipeline pipeline w/ list of transformers."""
    df_exp = df_input.drop_nulls().drop_nulls()
    pipe = TransformerPipeline(
        _TRANSFORMERS,
        interleaved=interleaved,
        prepend_interleaved=prepend_interleaved,
        append_interleaved=append_interleaved,
        name=name,
    )
    pipe.show()
    df_chk = pipe.run(df_input)
    pl_assert_equal(df_chk, df_exp)
