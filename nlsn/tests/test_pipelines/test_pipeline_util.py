"""Unit-tests for pipeline utils."""

from typing import List

import pytest

from nlsn.nebula.base import Transformer
from nlsn.nebula.pipelines.pipelines import TransformerPipeline
from nlsn.nebula.pipelines.util import *
from nlsn.nebula.spark_transformers import Distinct, NanToNull, SelectColumns


@pytest.mark.parametrize(
    "args",
    [
        ([NanToNull(glob="*")], True),
        ([Distinct()], True),
        ([[Distinct()]], False),
        ([Distinct(), 1], False),
    ],
)
def test_is_plain_transformer_list(args):
    """Test 'is_plain_transformer_list' function."""
    lst, exp = args
    assert is_plain_transformer_list(lst) == exp


def test_get_pipeline_name():
    """Test 'get_pipeline_name' function."""
    p1 = TransformerPipeline([])
    p1_name = get_pipeline_name(p1)
    assert "TransformerPipeline" in p1_name

    p2 = TransformerPipeline([], name="p2_test")
    p2_name = get_pipeline_name(p2)
    assert "TransformerPipeline" in p2_name
    assert "p2_test" in p2_name


@pytest.mark.parametrize(
    "transformers",
    [
        Distinct(),
        [Distinct()],
        (Distinct(),),
        [Distinct(), Distinct()],
        (Distinct(), Distinct()),
        [],
        None,
    ],
)
def test_sanitize_list_transformers_valid(transformers):
    """Test 'sanitize_list_transformers' function with valid inputs."""
    chk = sanitize_list_transformers(transformers)
    if not transformers:
        assert chk == []
        return

    if isinstance(transformers, tuple):
        exp = list(transformers)
    elif isinstance(transformers, list):
        exp = transformers
    elif isinstance(transformers, Transformer):
        exp = [transformers]
    else:
        raise ValueError("Unknown input test")
    assert exp == chk


@pytest.mark.parametrize("transformers", [1, [1], [Distinct(), 1]])
def test_sanitize_list_transformers_error(transformers):
    """Test 'sanitize_list_transformers' function with wrong inputs."""
    with pytest.raises(ValueError):
        sanitize_list_transformers(transformers)


@pytest.mark.parametrize("add_params", [True, False])
@pytest.mark.parametrize("max_len", [-1, 0, 100])
@pytest.mark.parametrize("wrap_text", [True, False])
@pytest.mark.parametrize("as_list", [True, False])
def test_get_transformer_name(add_params, max_len, wrap_text, as_list):
    """Test 'get_transformer_name' function."""
    cols_select: List[str] = ["this_column_is_23_chars"] * 100
    param_len_full: int = len("".join(cols_select))
    t = SelectColumns(columns=cols_select)
    kwargs = {
        "add_params": add_params,
        "max_len": max_len,
        "wrap_text": wrap_text,
        "as_list": as_list,
    }
    if add_params and wrap_text and as_list:
        with pytest.raises(AssertionError):
            get_transformer_name(t, **kwargs)
        return

    chk = get_transformer_name(t, **kwargs)

    base_len = len(f"{t.__class__.__name__} -> PARAMS: ") * 1.1  # keep a margin

    if as_list:
        assert isinstance(chk, list)
        if not add_params:
            return
        n_chk = sum(len(i) for i in chk)
        if max_len <= 0:
            assert n_chk > param_len_full, chk
        else:
            assert n_chk <= (base_len + max_len), chk
    else:
        assert isinstance(chk, str)
        if not add_params:
            return
        n_chk = len(chk)
        if max_len <= 0:
            assert n_chk > param_len_full, chk
        else:
            assert n_chk <= (base_len + max_len), chk


def _func1():
    pass


def _func2():
    pass


def _func3():
    pass


_expected_dict_extra_func = {"_func1": _func1, "_func2": _func2, "_func3": _func3}


@pytest.mark.parametrize(
    "funcs, expected",
    [
        ({}, {}),
        ([], {}),
        (_func2, {"_func2": _func2}),
        (_expected_dict_extra_func, _expected_dict_extra_func),
        ([_func1, _func2, _func3], _expected_dict_extra_func),
    ],
)
def test_create_dict_extra_functions(funcs, expected):
    """Test 'create_dict_extra_functions' function."""
    chk = create_dict_extra_functions(funcs)
    assert chk == expected


@pytest.mark.parametrize(
    "o",
    [
        "wrong",
        [_func1, _func2, _func2],
        [_func1, _func2, 1],
        {"_func1": _func1, "_func2": "_func2"},
    ],
)
def test_create_dict_extra_functions_error(o):
    """Test 'create_dict_extra_functions' function with wrong arguments."""
    with pytest.raises(AssertionError):
        create_dict_extra_functions(o)
