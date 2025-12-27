"""Inspect and verify the correctness of the transformer classes.

Ensure that:
- each class is a subclass of `nebula.base.Transformer`
- each class has a docstring within its `__init__` constructor
- no positional arguments are used in the `__init__` constructor
- the `_transform` method accepts only one positional argument and nothing else.

Then check if all the transformers are publicly declared in `__all__` for each py file.
"""

import inspect
from functools import lru_cache

import pytest

from nebula import transformers
from nebula.base import Transformer


# ------------------------------ Invalid Transformers


class InvalidPublicMethodTransform:
    def __init__(self):  # noqa: D107
        ...  # Private method "_transform is not defined"

    def transform(self):  # noqa: D102
        ...  # Public method "transform" should not be overwritten


class InvalidInitSignature:
    def __init__(self, a):  # noqa: D107
        ...  # Positional argument in __init__


# ------------------------------ Check Functions


def _test_transformer_subclass(transformer):
    """Transformer must be a subclass of base.Transformer."""
    assert issubclass(transformer, Transformer)


def _test_transformer_docstring(transformer):
    """Transformer must have the docstring in the '__init__' constructor."""
    doc = transformer.__init__.__doc__
    assert doc
    assert len(doc) > 10


def _test_transformer_init_signature(transformer):
    """Positional arguments are not allowed in the '__init__' constructor."""
    signature = inspect.signature(transformer.__init__).parameters
    # At least 'self'
    assert len(signature) >= 1

    li_param_names = list(signature)

    if len(signature) == 1:
        return

    li_param_names = li_param_names[1:]

    trf_name = transformer.__name__

    for param_name in li_param_names:
        param = signature[param_name]
        msg = f"Parameter '{param_name}' in {trf_name}.__init__ "
        msg += "must be keyword only."
        assert param.kind.name == "KEYWORD_ONLY", msg


# ------------------------------ Public Tests


class TestInvalidTransformers:
    """Test invalid transformers defined in this module.

    Use for loops and avoid pytest parametrize.
    """

    def test_invalid_transformer_basic_syntax(self):
        """Test the basic syntax."""
        li_func = [
            _test_transformer_subclass,
            _test_transformer_docstring,
        ]

        for func in li_func:
            with pytest.raises(AssertionError):
                func(InvalidPublicMethodTransform)

    def test_invalid_transformer_init_signature(self):
        """Test the init signature."""
        with pytest.raises(AssertionError):
            _test_transformer_init_signature(InvalidInitSignature)


def _is_transformer_subclass(o) -> bool:
    if o is Transformer:  # 'o' is a Transformer, not a subclass
        return False
    try:
        if issubclass(o, Transformer):
            return True
    except TypeError:  # not a class
        pass
    return False


def _filter_transformer_class(all_attrs):
    """Given a list from dir(...) filter the Transformer classes."""
    ret = []

    for el in all_attrs:
        if _is_transformer_subclass(el):
            ret.append(el)

    return sorted(ret, key=lambda x: x.__name__)


@lru_cache(maxsize=4)
def _get_all_transformers() -> list:
    """Retrieve all the public transformers."""
    all_attrs = [getattr(transformers, i) for i in dir(transformers)]
    return _filter_transformer_class(all_attrs)


def test_transformer_correctness():
    """Test All the public transformers defined in __init__.py."""
    li_transformers = _get_all_transformers()
    n_transformers = len(li_transformers)

    # Tested in an inner for loop. Don't use pytest parametrize.
    li_func = [
        _test_transformer_subclass,
        _test_transformer_docstring,
        _test_transformer_init_signature,
    ]

    for transformer in li_transformers:
        for func in li_func:
            try:
                func(transformer)

            except Exception as exc:
                trf_name = transformer.__name__
                func_name = func.__name__
                msg = f"Error: {trf_name} -> {func_name}"
                raise AssertionError(msg) from exc

    print(f"Tested the code correctness of {n_transformers} transformers")


def _get_transformers_modules() -> list[tuple]:
    # put here files that do not contain transformers
    avoid = {"__init__", "_constants"}

    ret = []
    for name in dir(transformers):
        if name in avoid:
            continue
        obj = getattr(transformers, name)
        if inspect.ismodule(obj):
            ret.append((name, obj))
    return ret


def test_public_declarations():
    """Test if __all__ is correctly populated for each py file."""
    # get all python files in /transformers
    modules = _get_transformers_modules()

    for module_name, module in modules:
        declared_in_file = set(module.__all__)
        dict_attrs = {i: getattr(module, i) for i in dir(module)}
        name: str
        for name, obj in dict_attrs.items():
            if name.startswith("_"):  # skip private objs
                continue
            if _is_transformer_subclass(obj):
                msg = f'"{name}" not declared in {module_name}.py "__all__"'
                assert name in declared_in_file, msg
