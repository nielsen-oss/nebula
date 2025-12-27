"""Utils to verify if an object is a valid transformer for a nebula pipeline."""

import inspect
from inspect import Parameter
from typing import Mapping

from nebula.base import Transformer

__all__ = ["is_transformer", "is_duck_typed_transformer"]


def is_transformer(o) -> bool:
    """Check if an object is a transformer."""
    return isinstance(o, Transformer) or is_duck_typed_transformer(o)


def _check_multiple_args(
    param_names: list[str], params: Mapping[str, Parameter]
) -> bool:
    """Check if additional parameters (after 'df') have default values.

    There are other parameters apart from 'df'.
    They are allowed, but they must have a default value.
    The first parameter has already been checked

    Args:
        param_names (list(str)):
            List of parameter names.
        params (mapping(str, inspect.Parameter)):
            Mapping of parameter names to Parameter objects.

    Returns (bool):
        True if all additional parameters have default values, False otherwise.
    """
    return all(params[name].default != Parameter.empty for name in param_names[1:])


def is_duck_typed_transformer(o) -> bool:
    """Check if object is a valid transformer via duck-typing.

    Allows custom transformers that don't inherit from Transformer.
    Must have a transform method with signature: transform(df, *args)
    where additional parameters have default values.

    Design notes:
    - First parameter name is flexible (df, data, etc.)
    - First parameter must be positional (not keyword-only)
    - Static methods are supported
    - Works with both class definitions and instances

    Args:
        o: Object to check (can be class or instance)

    Returns:
        True if object has valid transform method signature

    Examples:
        >>> class MyTransformer:
        ...     def transform(self, df, option=True):
        ...         return df
        >>> is_duck_typed_transformer(MyTransformer)
        True
        >>> is_duck_typed_transformer(MyTransformer())
        True
    """
    if not hasattr(o, "transform"):
        return False

    meth = getattr(o, "transform")
    params: Mapping[str, Parameter] = inspect.signature(meth).parameters
    param_names: list[str] = list(params)

    # If 'o' is a class and not an object yet, remove the 'self' parameter
    # if 'transform' is not a static method.
    # meth_type = inspect.getattr_static(o, "transform")
    is_static: bool = isinstance(inspect.getattr_static(o, "transform"), staticmethod)
    if isinstance(o, type) and (not is_static):
        param_names = param_names[1:]

    # Number of arguments w/o 'self'
    n_params: int = len(param_names)
    if not n_params:  # The method takes no arguments. Not valid.
        return False
    first_param_name: str = param_names[0]

    # "The first parameter of 'Transformer.transform' cannot be a 'keyword_only'"
    first_param: Parameter = params[first_param_name]
    if first_param.kind not in {
        Parameter.POSITIONAL_ONLY,
        Parameter.POSITIONAL_OR_KEYWORD,
    }:
        return False
    return (n_params == 1) or _check_multiple_args(param_names, params)
