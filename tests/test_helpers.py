"""Unit-test module for helpers functions."""

import numpy as np
import pandas as pd
import pytest

from nebula.helpers import *


@pytest.fixture(scope="module", name="df_input")
def _get_df_input():
    data = {
        "c_str": {0: "a", 1: "b", 2: None},
        "c_int": {0: 10, 1: 20, 2: None},
        "c_set": {0: {"a"}, 1: {"a", "b"}, 2: None},
        "c_none": {0: None, 1: None, 2: None},
        "c_str_none": {0: "a", 1: None, 2: None},
        "c_list": {0: ["a"], 1: [1, 2, 3], 2: None},
        "c_dict": {0: {"b": 10}, 1: {"a": 20}, 2: None},
        "c_np_array": {0: np.ones(1), 1: np.empty(1, dtype=np.bool_), 2: None},
        "c_np_array_obj": {0: np.array(["a"]), 1: np.empty(1, dtype="object"), 2: None},
        "c_complex": {0: {"b": {"inner_set_1"}}, 1: {"a": {"inner_set_2"}}, 2: None},
    }
    return pd.DataFrame(data)


@pytest.mark.parametrize("try_orjson", [True, False])
@pytest.mark.parametrize("deep", [True, False])
@pytest.mark.parametrize("copy", [True, False])
def test_assert_pandas_df_equal(
    df_input: pd.DataFrame, deep: bool, try_orjson: bool, copy: bool
):
    """Test 'assert_pandas_df_equal' function."""
    df1 = df_input.copy(deep=True)
    df2 = df_input.copy(deep=True)

    if deep:
        with pytest.raises(AssertionError):
            assert_pandas_df_equal(df1, df2, assert_not_deep=deep)
    else:
        df1 = df1.drop(columns=["c_complex"])
        df2 = df2.drop(columns=["c_complex"])
        assert_pandas_df_equal(
            df1, df2, assert_not_deep=deep, try_orjson=try_orjson, copy=copy
        )


@pytest.mark.parametrize("cut_rows", [True, False])
def test_assert_pandas_df_equal_wrong_shapes(df_input: pd.DataFrame, cut_rows: bool):
    """Test 'assert_pandas_df_equal' function passing different shapes."""
    df1 = df_input.copy(deep=True).drop(columns=["c_complex"])
    df2 = df_input.copy(deep=True).drop(columns=["c_complex"])

    if cut_rows:
        df1 = df1.iloc[:-1, :]
    else:
        df1 = df1.iloc[:, :-1]

    with pytest.raises(AssertionError):
        assert_pandas_df_equal(df1, df2)


def test_assert_pandas_df_equal_wrong_columns(df_input: pd.DataFrame):
    """Test 'assert_pandas_df_equal' function with different columns."""
    df1 = df_input.copy(deep=True).drop(columns=["c_complex"])
    df2 = df_input.copy(deep=True).drop(columns=["c_complex"])
    with pytest.raises(AssertionError):
        assert_pandas_df_equal(df1.rename(columns={"c_str": "c_str_2"}), df2)


@pytest.mark.parametrize("try_orjson", [True, False])
@pytest.mark.parametrize("orjson_opt_serialize_numpy", [True, False])
@pytest.mark.parametrize("deep", [True, False])
def test_hash_complex_type_pandas_dataframe(
    df_input: pd.DataFrame,
    deep: bool,
    try_orjson: bool,
    orjson_opt_serialize_numpy: bool,
):
    """Test 'hash_complex_type_pandas_dataframe' function."""
    if deep:
        with pytest.raises(AssertionError):
            hash_complex_type_pandas_dataframe(
                df_input,
                assert_not_deep=deep,
                try_orjson=try_orjson,
                orjson_opt_serialize_numpy=orjson_opt_serialize_numpy,
            )
    else:
        df = df_input.drop(columns=["c_complex"])
        ar_hash = hash_complex_type_pandas_dataframe(
            df,
            assert_not_deep=deep,
            orjson_opt_serialize_numpy=orjson_opt_serialize_numpy,
        )
        assert len(ar_hash) == df_input.shape[0]


@pytest.mark.parametrize("add_hash", [None, "hashed"])
def test_return_different_rows_in_pandas_dataframes(add_hash):
    """Test 'return_different_rows_in_pandas_dataframes' function."""
    data_1 = {
        "c_str": {0: "str_a", 1: "str_b"},
        "c_dict": {0: {"b": 10}, 1: {"a": 20}},
    }
    df1 = pd.DataFrame(data_1)

    data_2 = {
        "c_str": {0: "str_a", 1: "str_b"},
        "c_dict": {0: {"b": 10}, 1: {"a": 21}},
    }
    df2 = pd.DataFrame(data_2)

    df1_chk, df2_chk = return_different_rows_in_pandas_dataframes(
        df1, df2, add_hash=add_hash
    )

    df1_exp = df1.copy().loc[[1]]
    df2_exp = df2.copy().loc[[1]]

    if add_hash:
        h1 = df1_chk[add_hash].tolist()
        h2 = df2_chk[add_hash].tolist()
        assert all(isinstance(i, int) for i in h1)
        assert all(isinstance(i, int) for i in h2)
        df1_chk.drop(columns=[add_hash], inplace=True)
        df2_chk.drop(columns=[add_hash], inplace=True)

    pd.testing.assert_frame_equal(df1_chk, df1_exp)
    pd.testing.assert_frame_equal(df2_chk, df2_exp)
