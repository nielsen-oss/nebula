"""Test 'create_stages' function."""

from typing import List

import pytest

from nlsn.nebula.base import Transformer
from nlsn.nebula.pipelines.pipelines import TransformerPipeline, _create_stages
from nlsn.nebula.spark_transformers import (
    Cache,
    Distinct,
    LogDataSkew,
    NanToNull,
    SelectColumns,
)

_TRANSFORMERS: List[Transformer] = [
    SelectColumns(glob="*"),
    NanToNull(glob="*"),
    Distinct(),
]

_INTERLEAVED: List[Transformer] = [
    Cache(),
    LogDataSkew(),
]


def test_create_stages_error():
    """Test 'create_stages' function with input error."""
    with pytest.raises(TypeError):
        _create_stages(1, [], False)


def test_create_stages_with_single_transformer():
    """Test 'create_stages' function with a single transformer."""
    t = Distinct()
    chk, n_chk = _create_stages(t, [], False)
    assert n_chk == 1
    exp = [t]
    assert chk == exp


def test_create_stages_list():
    """Test 'create_stages' function with a list of transformers."""
    chk, n_chk = _create_stages(_TRANSFORMERS, [], False)
    assert n_chk == len(_TRANSFORMERS)
    assert chk == _TRANSFORMERS


@pytest.mark.parametrize("add_start", [False, True])
def test_create_stages_list_interleaved(add_start):
    """Test 'create_stages' function w/ a list(transformers) and 'interleaved'."""
    chk, n_chk = _create_stages(_TRANSFORMERS, _INTERLEAVED, add_start)
    exp = []

    if add_start:
        exp.extend(_INTERLEAVED[:])

    for t in _TRANSFORMERS:
        exp.append(t)
        exp.extend(_INTERLEAVED[:])

    assert n_chk == len(exp)
    assert chk == exp


@pytest.mark.parametrize("add_start", [False, True])
def test_create_stages_nested(add_start):
    """Test 'create_stages' function w/ transformers and pipeline objects."""
    pipe_test = TransformerPipeline(
        [Distinct()], interleaved=Cache(), prepend_interleaved=False, name="pipe_test"
    )

    nested_pipe = [_TRANSFORMERS[:], pipe_test, LogDataSkew()]
    interleaved = [Cache()]

    chk, n_chk = _create_stages(nested_pipe, interleaved, add_start)
    exp = []

    if add_start:
        exp.extend(interleaved[:])

    for obj in nested_pipe:
        if isinstance(obj, (tuple, list)):
            for el in obj:
                exp.append(el)
                exp.extend(interleaved[:])
        else:
            exp.append(obj)
            exp.extend(interleaved[:])

    assert n_chk == len(exp)
    assert chk == exp
