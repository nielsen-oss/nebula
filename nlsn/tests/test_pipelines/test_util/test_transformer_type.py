"""Test functions to verify if an object is a valid transformer for a nebula pipeline."""

import pytest

from nlsn.nebula.pipelines.transformer_type_util import (
    is_duck_typed_transformer,
    is_transformer,
)
from nlsn.nebula.transformers import AssertNotEmpty


# ---------------------------- VALID TRANSFORMERS -----------------------------


class TransformerOk:
    def __init__(self):  # noqa: D107
        ...

    def transform(self, df):  # noqa: D102
        return df


class TransformerOkStatic:
    def __init__(self):  # noqa: D107
        ...

    @staticmethod
    def transform(df):  # noqa: D102
        return df


class TransformerOkKeyWordEmpty:
    def __init__(self):  # noqa: D107
        ...

    def transform(self, df, a=0):  # noqa: D102
        return df


class TransformerOkKeyWordEmptyStatic:
    def __init__(self):  # noqa: D107
        ...

    @staticmethod
    def transform(df, a=None):  # noqa: D102
        return df


# -------------------------- NOT VALID TRANSFORMERS ---------------------------


class TransformerKoWrongMethod:
    def __init__(self):  # noqa: D107
        ...

    def transform_wrong(self, df):  # noqa: D102
        return df


class TransformerKoTwoArgs:
    def __init__(self):  # noqa: D107
        ...

    def transform(self, df, a):  # noqa: D102
        return df


class TransformerKoTwoArgsStatic:
    def __init__(self):  # noqa: D107
        ...

    @staticmethod
    def transform(df, _a):  # noqa: D102
        return df


class TransformerKoKeyWordEmpty:
    def __init__(self):  # noqa: D107
        ...

    def transform(self, df, *, a):  # noqa: D102
        return df


class TransformerKoKeyWordEmptyStatic:
    def __init__(self):  # noqa: D107
        ...

    @staticmethod
    def transform(df, *, a):  # noqa: D102
        return df


class TransformerKoNoArgs:
    def __init__(self):  # noqa: D107
        ...

    def transform(self):  # noqa: D102
        return


class TransformerKoNoArgsStatic:
    def __init__(self):  # noqa: D107
        ...

    @staticmethod
    def transform():  # noqa: D102
        return


class TransformerKoKeyWordOnlyDF:
    def __init__(self):  # noqa: D107
        ...

    def transform(self, *, df):  # noqa: D102
        return


class TransformerKoKeyWordOnlyDFStatic:
    def __init__(self):  # noqa: D107
        ...

    @staticmethod
    def transform(*, df):  # noqa: D102
        return


@pytest.mark.parametrize(
    "cls",
    [
        TransformerOk,
        TransformerOkStatic,
        TransformerOkKeyWordEmpty,
        TransformerOkKeyWordEmptyStatic,
    ],
)
def test_valid_transformer_type(cls):
    """Test valid transformers w/o known parent class."""
    # classes
    assert is_duck_typed_transformer(cls)
    # objects
    assert is_duck_typed_transformer(cls())


@pytest.mark.parametrize(
    "cls",
    [
        TransformerKoWrongMethod,
        TransformerKoTwoArgs,
        TransformerKoTwoArgsStatic,
        TransformerKoKeyWordEmpty,
        TransformerKoKeyWordEmptyStatic,
        TransformerKoNoArgs,
        TransformerKoNoArgsStatic,
        TransformerKoKeyWordOnlyDF,
        TransformerKoKeyWordOnlyDFStatic,
    ],
)
def test_not_valid_transformer_type(cls):
    """Test not valid transformers w/o known parent class."""
    # classes
    assert not is_duck_typed_transformer(cls)
    # objects
    assert not is_duck_typed_transformer(cls())


@pytest.mark.parametrize(
    "o, exp", [(TransformerOk(), True), (AssertNotEmpty(), True), ([1], False)]
)
def test_is_transformer(o, exp: bool):
    """Test 'is_transformer' function."""
    assert is_transformer(o) == exp
