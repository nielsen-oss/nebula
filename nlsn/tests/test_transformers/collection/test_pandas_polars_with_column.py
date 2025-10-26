"""Unit-test for Pandas / Polars WithColumn."""

import pandas as pd
import polars as pl
import pytest

from nlsn.nebula.shared_transformers import WithColumn


@pytest.fixture(scope="module", name="df_input")
def _get_df_input():
    return pd.DataFrame([["a"], ["b"]], columns=["col_1"])


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize(
    "value, cast",
    [
        (1, None),
        (None, None),
        (1, "float64"),
        ("1", None),
    ],
)
def test_with_column(df_input, backend, value, cast):
    """Test WithColumn transformer."""
    column_name = "col_2"

    df_exp = df_input.copy()
    df_exp[column_name] = value
    if cast is not None:
        df_exp[column_name] = df_exp[column_name].astype(cast)

    d_polars_cast = {"float64": pl.Float64}

    check_dtype = True
    if backend == "polars":
        check_dtype = False
        df_input = pl.from_pandas(df_input)
        if cast is not None:
            check_dtype = True
            cast = d_polars_cast[cast]

    t = WithColumn(column_name=column_name, value=value, cast=cast)

    df_out = t.transform(df_input)

    if backend == "polars":
        df_out = df_out.to_pandas()

    pd.testing.assert_frame_equal(df_exp, df_out, check_dtype=check_dtype)
