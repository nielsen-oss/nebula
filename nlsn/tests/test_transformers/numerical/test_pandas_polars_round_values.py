"""Unit-test for RoundValues."""

import numpy as np
import pandas as pd
import pytest

from nlsn.nebula.shared_transformers import RoundValues
from nlsn.tests.auxiliaries import assert_pandas_polars_frame_equal, pandas_to_polars

# fmt: off
_params = [
    ("c_int", 0, None),
    ("c_double", 3, None),
    ("c_double", -1, None),
    ("c_double", -2, None),
    (["c_double"], 2, "col_output"),
    (["c_double"], 2, None),
    (["c_double", "c_double_2"], 2, None),
    (["c_double", "c_int"], 1, None),
    (["c_double", "c_int"], -1, None),
]

_data = [
    ("1", 101.342),
    ("2", 3.14159),
    ("3", 0.123456782),
    ("4", -354.123456789),
    ("5", -6.0),
    ("6", float("nan")),
]


# fmt: on


@pytest.fixture(scope="module", name="df_input")
def _get_df_input():
    df = pd.DataFrame(_data, columns=["id", "c_double"])
    df["c_double_2"] = df["c_double"].copy() * 2
    df["c_int"] = np.arange(df.shape[0]).astype("int32")
    return df


def test_round_values_invalid_parameters():
    """Test case with output column and multiple input columns."""
    with pytest.raises(AssertionError):
        RoundValues(input_columns=["c1", "c2"], output_column="c3", precision=1)


def test_round_values_duplicated_input_columns():
    """Test case with duplicated input columns."""
    with pytest.raises(AssertionError):
        RoundValues(input_columns=["a", "a"], precision=1)


def test_round_values_invalid_input_column_type(df_input):
    """Test case with input column with non-numeric type."""
    with pytest.raises(TypeError):
        RoundValues(input_columns="id", precision=2).transform(df_input)


@pytest.mark.parametrize("df_type", ["pandas", "polars"])
@pytest.mark.parametrize("input_columns, precision, output_column", _params)
def test_round_values(df_input, df_type: str, input_columns, precision, output_column):
    """Unit-test RoundValues."""
    int_in_columns: bool = (input_columns == "c_int") or ("c_int" in input_columns)
    if df_type == "polars" and int_in_columns and precision >= 0:
        # i32 / i64 types do not support round with positive precision
        return

    t = RoundValues(
        input_columns=input_columns, precision=precision, output_column=output_column
    )

    df_exp = df_input.copy()

    df_input = pandas_to_polars(df_type, df_input)

    df_chk = t.transform(df_input)

    input_columns = (
        input_columns if isinstance(input_columns, list) else [input_columns]
    )

    if output_column:
        df_exp[output_column] = df_exp[input_columns[0]].round(precision)
    else:
        for c in input_columns:
            df_exp[c] = df_exp[c].round(precision)

    df_exp = pandas_to_polars(df_type, df_exp)

    assert_pandas_polars_frame_equal(df_type, df_exp, df_chk)
