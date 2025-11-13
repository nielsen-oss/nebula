"""Inspect and verify the correctness of the transformer classes.

Ensure that:
- the class names do not end with `Transformer`
- each class is a subclass of `nlsn.nebula.base.Transformer`
- each class has a docstring within its `__init__` constructor
- no positional arguments are used in the `__init__` constructor
- the class does not have a public method named `transform`
- each class has a private method named `_transform`
- the `_transform` method accepts only one positional argument and nothing else.

Then check if all the transformers are publicly declared in `__all__` for each py file.
"""

import inspect
from functools import lru_cache
from typing import List, Mapping, Optional, Set

import pytest

from nlsn.nebula import spark_transformers
from nlsn.nebula.base import Transformer

# ------------------------------ Wrong Transformers


class WrongInitSignature1:
    def __init__(self, a):  # noqa: D107
        ...  # Positional argument in __init__


class WrongTransformMethodSignature1:
    def _transform(self): ...  # It does not take df as an argument


class WrongTransformMethodSignature2:
    def _transform(self, a, b): ...  # Too many arguments


class WrongTransformMethodSignature3:
    def _transform(self, *, a): ...  # "df" is not positional


class WrongTransformerBackendName:  # Wrong backend name
    backends = {"invalid"}


class WrongTransformerBackendsType:  # Invalid backend Type
    backends = ["invalid"]


class WrongTransformerBackendMethodMismatch:  # Backend method mismatch
    backends = {"spark", "polars"}

    def _transform_spark(self, df): ...

    def _transform_pandas(self, df): ...


# ------------------------------ Check Functions


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


def _test_transformer_method_signature(transformer):
    """Verify the `_transform` method.

    - The method '_transform' must accept only one positional argument and
        nothing else.
    - If _transform is not static, the "self" parameter (the reference to the
        current instance) must be called "self".
    """
    meth_type = inspect.getattr_static(transformer, "_transform")
    is_static: bool = isinstance(meth_type, staticmethod)

    meth = getattr(transformer, "_transform")
    signature: Mapping[str, inspect.Parameter] = inspect.signature(meth).parameters
    li_params_name: List[str] = list(signature)

    # Since they are NOT initialized, I need to remove the 'self' parameter
    if is_static:
        assert len(signature) == 1
        param_df_name: str = li_params_name[0]
    else:
        # 'self' and 'df'
        assert len(signature) == 2

        param_df_name: str = li_params_name[1]

    param_df: inspect.Parameter = signature[param_df_name]
    kind: str = param_df.kind.name

    if kind == "POSITIONAL_ONLY":
        pass
    elif kind == "POSITIONAL_OR_KEYWORD":
        assert param_df.default == inspect.Parameter.empty
    else:
        msg = "Parameter in '_transform' method must be positional "
        msg += "only or keyword without any default."
        raise AssertionError(msg)


def _test_transformer_backend(transformer):
    """Ensure that if the backend is declared, it is valid, and it has the proper method."""
    allowed_backends = {"pandas", "spark", "polars"}

    backends: Optional[Set[str]] = getattr(transformer, "backends", set())

    # Assert the implemented backend methods are declared.
    for backend in allowed_backends:
        if hasattr(transformer, f"_transform_{backend}") and (backend not in backends):
            name = transformer.__name__
            msg = f"Found the method '_transform_{backend}' in {name} "
            msg += f"but '{backend}' is not declared in the transformer"
            raise AssertionError(msg)

    if not backends:
        return

    if not isinstance(backends, set):
        raise AssertionError("If declared, 'backends' must be a <set<str>>")

    # Assert the set of backends is valid.
    if not backends.issubset(allowed_backends):
        diff = backends.difference(allowed_backends)
        raise AssertionError(f"backend(s) '{diff}' not allowed")


# ------------------------------ Public Tests


class TestWrongTransformers:
    """Test Wrong transformers defined in this module.

    Use for loops and avoid pytest parametrize.
    """

    def test_wrong_transformer_init_signature(self):
        """Test the init signature."""
        li_wrong_init_signature = [
            WrongInitSignature1,
        ]
        for trf in li_wrong_init_signature:
            with pytest.raises(AssertionError):
                _test_transformer_init_signature(trf)

    def test_wrong_transformer_method_signature(self):
        """Test the method signature."""
        li_wrong_transform_signature = [
            WrongTransformMethodSignature1,
            WrongTransformMethodSignature2,
            WrongTransformMethodSignature3,
        ]
        for trf in li_wrong_transform_signature:
            with pytest.raises(AssertionError):
                _test_transformer_method_signature(trf)

    @pytest.mark.parametrize(
        "trf",
        [
            WrongTransformerBackendName,
            WrongTransformerBackendsType,
            WrongTransformerBackendMethodMismatch,
        ],
    )
    def test_wrong_transformer_backend(self, trf):
        """Test the backend declarations and methods."""
        with pytest.raises(AssertionError):
            _test_transformer_backend(trf)


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
    all_attrs = [getattr(spark_transformers, i) for i in dir(spark_transformers)]
    return _filter_transformer_class(all_attrs)


def test_transformer_correctness():
    """Test All the public transformers defined in __init__.py."""
    li_transformers = _get_all_transformers()
    n_transformers = len(li_transformers)

    # Tested in an inner for loop. Don't use pytest parametrize.
    li_func = [
        _test_transformer_init_signature,
        _test_transformer_method_signature,
        _test_transformer_backend,
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


def _get_transformers_modules() -> List[tuple]:
    # put here files that do not contain transformers
    avoid = {"__init__", "_constants"}

    ret = []
    for name in dir(spark_transformers):
        if name in avoid:
            continue
        obj = getattr(spark_transformers, name)
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
