"""Module containing helper functions."""

import json
from typing import Hashable

import numpy as np
import pandas as pd

try:  # pragma: no cover
    import orjson

    _HAS_ORJSON = True
except ImportError:  # pragma: no cover
    _HAS_ORJSON = False

__all__ = [
    "assert_pandas_df_equal",
    "hash_complex_type_pandas_dataframe",
    "return_different_rows_in_pandas_dataframes",
]


def __all_elements_are_hashable(s: pd.Series) -> None:
    if not all(isinstance(i, Hashable) for i in s.tolist()):
        name = s.name
        raise AssertionError(f"Column {name} contains not hashable elements")


def _get_cols_to_json(
        df, deep: bool
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Get 4 lists indicating which columns must be dumped to JSON."""
    ret_dict: list[str] = []  # dictionaries
    ret_list: list[str] = []  # lists
    ret_set: list[str] = []  # sets
    ret_ar: list[str] = []  # arrays

    c: str
    for c, dtype in df.dtypes.to_dict().items():
        if dtype.name != "object":
            continue

        series = df[c].dropna()

        if len(series) == 0:
            continue

        if deep:
            __all_elements_are_hashable(series)

        el = series.iloc[0]
        if isinstance(el, dict):
            ret_dict.append(c)
        elif isinstance(el, list):
            ret_list.append(c)
        elif isinstance(el, set):
            ret_set.append(c)
        elif isinstance(el, np.ndarray):
            ret_ar.append(c)
    return ret_dict, ret_list, ret_set, ret_ar


def __dict_to_json(s: pd.Series) -> pd.Series:
    return s.apply(lambda x: "null" if x is None else json.dumps(sorted(x.items())))


def __list_to_json(s: pd.Series) -> pd.Series:
    # Do not sort it
    return s.apply(json.dumps)


def __set_to_json(s: pd.Series) -> pd.Series:
    return s.apply(lambda x: "null" if x is None else json.dumps(sorted(x)))


def __ar_to_json(s: pd.Series) -> pd.Series:
    # Do not sort it
    return s.apply(lambda x: "null" if x is None else json.dumps(x.tolist()))


def __dict_to_orjson(s: pd.Series) -> pd.Series:
    return s.apply(lambda x: b"null" if x is None else orjson.dumps(sorted(x.items())))


def __list_to_orjson(s: pd.Series) -> pd.Series:
    # Do not sort it
    return s.apply(orjson.dumps)


def __set_to_orjson(s: pd.Series) -> pd.Series:
    return s.apply(lambda x: b"null" if x is None else orjson.dumps(sorted(x)))


def __ar_to_orjson(s: pd.Series) -> pd.Series:
    # Do not sort it
    return s.apply(lambda x: b"null" if x is None else orjson.dumps(x.tolist()))


def __ar_to_orjson_serialized(s: pd.Series) -> pd.Series:
    # Do not sort it
    return s.apply(
        lambda x: b"null"
        if x is None
        else orjson.dumps(x, option=orjson.OPT_SERIALIZE_NUMPY)
    )


def _hash_df(
        df: pd.DataFrame,
        dict_to_json: list[str],
        list_to_json: list[str],
        set_to_json: list[str],
        ar_to_json: list[str],
        *,
        copy: bool,
        try_orjson: bool,
        orjson_opt_serialize_numpy: bool,
) -> np.ndarray:
    """Convert a dataframe to a sorted array of hash.

    Dump columns to JSON if needed.
    """
    if _HAS_ORJSON and try_orjson:
        _dict_to_json = __dict_to_orjson
        _list_to_json = __list_to_orjson
        _set_to_json = __set_to_orjson
        if orjson_opt_serialize_numpy:
            _ar_to_json = __ar_to_orjson_serialized
        else:
            _ar_to_json = __ar_to_orjson
    else:
        _dict_to_json = __dict_to_json
        _list_to_json = __list_to_json
        _set_to_json = __set_to_json
        _ar_to_json = __ar_to_json

    if copy:
        df_hashable = df.copy()
    else:
        df_hashable = df

    for c in dict_to_json:
        df_hashable[c] = _dict_to_json(df_hashable[c])

    for c in list_to_json:
        df_hashable[c] = _list_to_json(df_hashable[c])

    for c in set_to_json:
        df_hashable[c] = _set_to_json(df_hashable[c])

    for c in ar_to_json:
        if _HAS_ORJSON and try_orjson and orjson_opt_serialize_numpy:
            try:
                df_hashable[c] = __ar_to_orjson_serialized(df_hashable[c])
            except TypeError:
                df_hashable[c] = __ar_to_orjson(df_hashable[c])
        else:
            df_hashable[c] = _ar_to_json(df_hashable[c])

    return pd.util.hash_pandas_object(df_hashable, index=False).values


def assert_pandas_df_equal(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        assert_not_deep: bool = False,
        copy: bool = True,
        try_orjson: bool = True,
        orjson_opt_serialize_numpy: bool = False,
) -> None:
    """Assert if 2 flat complex-type pandas dataframe are equal.

    !!!
    When a complex column contains mixed types (i.e., a set containing both
    integers and strings: {"a", 1}) it throws a TypeError.
    !!!

    Flat complex types like list / set / dictionaries are handled with JSON.
    Nested complex types, like dictionaries of dictionaries are not handled.

    Args:
        df1 (pd.DataFrame):
            First dataframe to compare.
        df2 (pd.DataFrame):
            Second dataframe to compare.
        assert_not_deep (bool):
            If True assert, that the values are not nested.
            If so, raise an assertion Error. Defaults to False.
        copy (bool):
            If False, modifies inplace the input dataframes to save
            memory. Default to True.
        try_orjson (bool):
            If True, try to use orjson package instead of the builtin JSON.
            Defaults to True.
        orjson_opt_serialize_numpy (bool):
            Serialize the array with orjson internal serializer.
            The array must be a contiguous C array (C_CONTIGUOUS) and one of the
            supported datatypes.
            float64 are downcast to float32, This can result in different
            rounding. Defaults to False

    Returns (None):
    """
    if df1.shape != df2.shape:
        raise AssertionError("Different shapes")

    if set(df1.columns) != set(df2.columns):
        raise AssertionError("Different columns")

    dict_to_json, list_to_json, set_to_json, ar_to_json = _get_cols_to_json(
        df1, assert_not_deep
    )

    sorted_cols = df1.columns.tolist()
    ar_1 = _hash_df(
        df1,
        dict_to_json,
        list_to_json,
        set_to_json,
        ar_to_json,
        copy=copy,
        try_orjson=try_orjson,
        orjson_opt_serialize_numpy=orjson_opt_serialize_numpy,
    )
    ar_2 = _hash_df(
        df2[sorted_cols],
        dict_to_json,
        list_to_json,
        set_to_json,
        ar_to_json,
        copy=copy,
        try_orjson=try_orjson,
        orjson_opt_serialize_numpy=orjson_opt_serialize_numpy,
    )

    ar_1_sorted = np.sort(np.sort(ar_1))
    ar_2_sorted = np.sort(np.sort(ar_2))

    np.testing.assert_array_equal(ar_1_sorted, ar_2_sorted)


def hash_complex_type_pandas_dataframe(
        df: pd.DataFrame,
        assert_not_deep: bool = False,
        copy: bool = True,
        try_orjson: bool = True,
        orjson_opt_serialize_numpy: bool = False,
) -> np.ndarray:
    """Hash a flat complex-type pandas dataframe.

    !!!
    When a complex column contains mixed types (i.e., a set containing both
    integers and strings: {"a", 1}) it throws a TypeError.
    !!!

    Flat complex types like list / set / dictionaries are handled with JSON.
    Nested complex types, like dictionaries of dictionaries are not handled.

    Args:
        df (pd.DataFrame):
            Input dataframe to hash.
        assert_not_deep (bool):
            If True assert, that the values are not nested.
            If so, raise an assertion Error. Defaults to False.
        copy (bool):
            If False, modifies inplace the input dataframe to save
            memory. Default to True.
        try_orjson (bool):
            If True, try to use orjson package instead of the builtin JSON.
            Defaults to True.
        orjson_opt_serialize_numpy (bool):
            Serialize the array with orjson internal serializer.
            The array must be a contiguous C array (C_CONTIGUOUS) and one of the
            supported datatypes.
            float64 are downcast to float32, This can result in different
            rounding. Defaults to False

    Returns (np.ndarray):
        1-d int64 array containing the hashes
    """
    dict_to_json, list_to_json, set_to_json, ar_to_json = _get_cols_to_json(
        df, assert_not_deep
    )
    return _hash_df(
        df,
        dict_to_json,
        list_to_json,
        set_to_json,
        ar_to_json,
        copy=copy,
        try_orjson=try_orjson,
        orjson_opt_serialize_numpy=orjson_opt_serialize_numpy,
    )


def return_different_rows_in_pandas_dataframes(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        add_hash: str | None = None,
        assert_not_deep: bool = False,
        copy: bool = True,
        try_orjson: bool = True,
        orjson_opt_serialize_numpy: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the rows that are different between two pandas DataFrames.

    !!!
    When a complex column contains mixed types (i.e., a set containing both
    integers and strings: {"a", 1}) it throws a TypeError.
    !!!

    Flat complex types like lists / sets / dictionaries are handled with JSON.
    Nested complex types, like dictionaries of dictionaries are not handled.

    Args:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.
        add_hash (str | None):
            Whether to add the hash column used for comparison.
            Defaults to False.
        assert_not_deep (bool):
            If True assert, that the values are not nested.
            If so, raise an assertion Error.
            Defaults to False.
        copy (bool):
            If False, modifies inplace the input dataframe to save
            memory. Default to True.
        try_orjson (bool):
            If True, try to use orjson package instead of the builtin JSON.
            Defaults to True.
        orjson_opt_serialize_numpy (bool):
            Serialize the array with orjson internal serializer.
            The array must be a contiguous C array (C_CONTIGUOUS) and one of the
            supported datatypes.
            float64 are downcast to float32, This can result in different
            rounding. Defaults to False

    Returns (pd.DataFrame, pd.DataFrame):
        A tuple containing two DataFrames, where each DataFrame
        contains the rows that are different between df1 and df2.
    """
    s_hash_1 = pd.Series(
        hash_complex_type_pandas_dataframe(
            df1,
            assert_not_deep,
            copy=copy,
            try_orjson=try_orjson,
            orjson_opt_serialize_numpy=orjson_opt_serialize_numpy,
        )
    )
    s_hash_2 = pd.Series(
        hash_complex_type_pandas_dataframe(
            df2,
            assert_not_deep,
            copy=copy,
            try_orjson=try_orjson,
            orjson_opt_serialize_numpy=orjson_opt_serialize_numpy,
        )
    )

    set_hash_1 = set(s_hash_1.tolist())
    set_hash_2 = set(s_hash_2.tolist())

    set_diff = set_hash_1.symmetric_difference(set_hash_2)
    n_diff = len(set_diff)

    print(f"Found {n_diff} different rows")

    df1_ret = df1[s_hash_1.isin(set_diff)].copy()
    df2_ret = df2[s_hash_2.isin(set_diff)].copy()

    if add_hash:
        df1_ret[add_hash] = s_hash_1
        df2_ret[add_hash] = s_hash_2

    return df1_ret, df2_ret
