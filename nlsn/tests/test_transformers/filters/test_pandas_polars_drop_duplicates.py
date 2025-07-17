"""Unit-test for Distinct."""

import pandas as pd
import polars as pl
import pytest

from nlsn.nebula.shared_transformers import Distinct
from nlsn.tests.auxiliaries import assert_pandas_polars_frame_equal


@pytest.fixture(scope="module", name="df_input")
def _get_df_input():
    # fmt: off
    data = [
        ["a1", "b1", "c1"],
        ["a1", "b1", "c1"],
        ["a1", "b1", None],
        ["a1", "b2", "c2"],
        ["a1", "b2", "c2"],
        ["a1", None, "c2"],
    ]
    # fmt: on
    return pd.DataFrame(data, columns=["a", "b", "c"])


def test_pandas_distinct(df_input):
    """Test Distinct transformer with a Pandas dataframe."""
    t = Distinct()
    df_exp = df_input.copy().drop_duplicates().reset_index(drop=True)
    df_chk = t.transform(df_input).reset_index(drop=True)
    assert_pandas_polars_frame_equal("pandas", df_exp, df_chk)


@pytest.mark.parametrize("maintain_order", [False])
def test_polars_distinct(df_input, maintain_order: bool):
    """Test Distinct transformer with a Polars dataframe."""
    df_input = pl.from_pandas(df_input)
    df_exp = df_input.unique()

    t = Distinct(maintain_order=maintain_order)
    df_chk = t.transform(df_input)

    if not maintain_order:
        sort_cols = ["c", "b", "a"]
        df_exp = df_exp.sort(sort_cols)
        df_chk = df_chk.sort(sort_cols)

    assert_pandas_polars_frame_equal("polars", df_exp, df_chk)
