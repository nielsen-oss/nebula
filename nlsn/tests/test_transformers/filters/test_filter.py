"""Unit-test for Filter."""

import math
import operator as py_operator
import re

import numpy as np
import pytest
from pyspark.sql.types import ArrayType, FloatType, StringType, StructField, StructType

from nlsn.nebula.spark_transformers import Filter

_DATA = [
    ["a", 0.0, [1.0, 2.0]],
    ["a1", 1.0, [2.0, 3.0]],
    ["a", 2.0, [3.0, 4.0]],
    ["a1", 3.0, [4.0, 5.0]],
    ["b1", 4.0, [5.0]],
    ["ab1", 5.0, []],
    ["  ", -math.inf, None],
    ["", math.inf, None],
    ["", np.nan, None],
    [None, None, None],
]

_LIST_STR = [i[0] for i in _DATA]
_LIST_FLT = [i[1] for i in _DATA]
# Keep tuple as the unit-test needs to hash it
_LIST_ARR = [None if i[2] is None else tuple(i[2]) for i in _DATA]


def test_filter_ne_operator_disallowed():
    """Test Filter transformer with a disallowed 'operator' value."""
    with pytest.raises(AssertionError):
        Filter(input_col="c1", operator="ne", value=0, perform="keep")


def _isna(x) -> bool:
    if isinstance(x, (int, float)):
        return math.isnan(x)
    return False


def __set_nan(o) -> (bool, set):
    flag = False
    ret = o
    try:
        flag = any(_isna(i) for i in ret)
        ret = {i for i in ret if not _isna(i)}
    except:  # noqa: E722
        pass
    return flag, ret


def _check_set_with_nan(x1, y1) -> bool:
    # like set_1 = set_2, but:
    # {0.0, 2.0, None, np.nan} == {0.0, 2.0, None, np.nan} -> true
    # {0.0, 2.0, None, float("nan")} == {0.0, 2.0, None, float("nan")} -> false

    flag_x, x2 = __set_nan(x1)
    flag_y, y2 = __set_nan(y1)

    return (flag_x == flag_y) and (x2 == y2)


def _cmp_flt(x, y, op) -> bool:
    if x is None:
        return False
    if _isna(x):
        return False
    cmp = getattr(py_operator, op)
    return cmp(x, y)


def _chk_full(df, exp, kwargs, c, t, op) -> None:
    transformer = Filter(input_col=c, perform=op, **kwargs)
    li_chk = transformer.transform(df).select(c).collect()

    if t == "array":
        chk = {None if i[0] is None else tuple(i[0]) for i in li_chk}
        # Don't check nan here
        ans = chk == exp
    else:
        chk = {i[0] for i in li_chk}
        ans = _check_set_with_nan(chk, exp)

    if not ans:
        msg = f"test {t} + {op}: args: {kwargs} | exp={exp} | chk={chk}"
        raise AssertionError(msg)


@pytest.fixture(scope="module", name="df_input")
def _get_input_df(spark):
    fields = [
        StructField("input_string", StringType(), True),
        StructField("input_float", FloatType(), True),
        StructField("input_array", ArrayType(FloatType()), True),
    ]
    schema = StructType(fields)
    return spark.createDataFrame(_DATA, schema=schema).persist()


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "operator": "eq",
            "value": "a",
            "exp_keep": {i for i in _LIST_STR if i == "a"},
        },
        # ne operator is disallowed since can be ambiguous for null values
        # {"operator": "ne", "value": "a", "exp_keep": {i for i in _LIST_STR if i != "a"}},
        {
            "operator": "isin",
            "value": {"a", "b1", "c1"},
            "exp_keep": {i for i in _LIST_STR if i in {"a", "b1", "c1"}},
        },
        {
            "operator": "contains",
            "value": "a",
            "exp_keep": {i for i in _LIST_STR if isinstance(i, str) and ("a" in i)},
        },
        {
            "operator": "startswith",
            "value": "ab",
            "exp_keep": {
                i for i in _LIST_STR if isinstance(i, str) and i.startswith("ab")
            },
        },
        {
            "operator": "endswith",
            "value": "b1",
            "exp_keep": {
                i for i in _LIST_STR if isinstance(i, str) and i.endswith("b1")
            },
        },
        {
            "operator": "like",
            "value": "ab%",
            "exp_keep": {
                i for i in _LIST_STR if isinstance(i, str) and re.match("^ab.*", i)
            },
        },
        {
            "operator": "rlike",
            "value": "b1$",
            "exp_keep": {
                i for i in _LIST_STR if isinstance(i, str) and re.match(".*b1", i)
            },
        },
        {"operator": "isNull", "exp_keep": {None}},
        {"operator": "isNotNull", "exp_keep": {i for i in _LIST_STR if i is not None}},
        {"operator": "isNaN", "exp_keep": set()},
        {"operator": "isNotNaN", "exp_keep": set(_LIST_STR)},
    ],
)
def test_filter_with_strings(df_input, kwargs):
    """Test 'Filter' transformer with strings."""
    exp_keep = kwargs.pop("exp_keep")
    exp_remove = set(_LIST_STR).difference(exp_keep)

    _chk_full(df_input, exp_keep, kwargs, "input_string", "string", "keep")
    _chk_full(df_input, exp_remove, kwargs, "input_string", "string", "remove")


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "operator": "eq",
            "value": 1.0,
            "exp_keep": {i for i in _LIST_FLT if i == 1.0},
        },
        {
            "operator": "le",
            "value": 2.0,
            "exp_keep": {i for i in _LIST_FLT if _cmp_flt(i, 2.0, "le")},
        },
        {
            "operator": "lt",
            "value": 2.0,
            "exp_keep": {i for i in _LIST_FLT if _cmp_flt(i, 2.0, "lt")},
        },
        {
            "operator": "ge",
            "value": 2.0,
            "exp_keep": {i for i in _LIST_FLT if _cmp_flt(i, 2.0, "ge")},
        },
        {
            "operator": "gt",
            "value": 2.0,
            "exp_keep": {i for i in _LIST_FLT if _cmp_flt(i, 2.0, "gt")},
        },
        # ne operator is disallowed since can be ambiguous for null values
        # {"operator": "ne", "value": "a", "exp_keep": {i for i in _LIST_STR if i != "a"}},
        {
            "operator": "isin",
            "value": {1.0, 2.0, -5.5},
            "exp_keep": {i for i in _LIST_FLT if i in {1.0, 2.0, -5.5}},
        },
        {
            "operator": "between",
            "value": [2, 4],
            "exp_keep": {i for i in _LIST_FLT if (i is not None) and (2 <= i <= 4)},
        },
        {"operator": "isNull", "exp_keep": {None}},
        {"operator": "isNotNull", "exp_keep": {i for i in _LIST_FLT if i is not None}},
        {"operator": "isNaN", "exp_keep": {np.nan}},
        {"operator": "isNotNaN", "exp_keep": {i for i in _LIST_FLT if not _isna(i)}},
    ],
)
def test_filter_with_floats(df_input, kwargs):
    """Test 'Filter' transformer with floats."""
    exp_keep = kwargs.pop("exp_keep")
    exp_remove = set(_LIST_FLT).difference(exp_keep)

    _chk_full(df_input, exp_keep, kwargs, "input_float", "float", "keep")
    _chk_full(df_input, exp_remove, kwargs, "input_float", "float", "remove")


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "operator": "array_contains",
            "value": 3.0,
            "exp_keep": {i for i in _LIST_ARR if i and (3.0 in i)},
        },
    ],
)
def test_filter_with_arrays(df_input, kwargs):
    """Test 'Filter' transformer with arrays."""
    exp_keep = kwargs.pop("exp_keep")
    exp_remove = set(_LIST_ARR).difference(exp_keep)

    _chk_full(df_input, exp_keep, kwargs, "input_array", "array", "keep")
    _chk_full(df_input, exp_remove, kwargs, "input_array", "array", "remove")
