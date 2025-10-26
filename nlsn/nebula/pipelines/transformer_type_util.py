"""Utils to verify if an object is a valid transformer for a nebula pipeline."""

import inspect
from inspect import Parameter
from typing import List, Mapping

from nlsn.nebula.backend_util import HAS_SPARK
from nlsn.nebula.base import Transformer

__all__ = ["is_transformer", "is_generic_transformer"]

if HAS_SPARK:
    from pyspark.ml import Transformer as PysparkTransformer

    TransformerTypes = (Transformer, PysparkTransformer)
else:  # pragma: no cover
    TransformerTypes = (Transformer,)


def is_transformer(o) -> bool:
    """Check if an object is a transformer."""
    return isinstance(o, TransformerTypes) or is_generic_transformer(o)


def _check_multiple_args(
    param_names: List[str], params: Mapping[str, Parameter]
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


def is_generic_transformer(o) -> bool:
    """Check if the input is a generic transformer object w/o any known parent class."""
    if not hasattr(o, "transform"):
        return False

    meth = getattr(o, "transform")
    params: Mapping[str, Parameter] = inspect.signature(meth).parameters
    param_names: List[str] = list(params)

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
