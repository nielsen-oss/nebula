"""Handle deprecations."""

import functools
import warnings
from copy import deepcopy

__all__ = [
    "deprecate_transformer",
]


def _deprecate_func(func, msg: str):
    """Decorator for deprecated functions.

    This is a decorator that can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn(msg, DeprecationWarning, 2)
        return func(*args, **kwargs)

    return new_func


def _deprecate_class_attribute(attrs: dict, name: str, msg: str) -> None:
    """Raise a warning whenever an attribute / method / descriptor is called.

    Args:
        attrs (dict):
            Class attributes.
        name (str):
            Name of the attribute (method, descriptor, property) that
            will be decorated with the warning.
        msg (msg):
            Warning message.

    Returns: None
        Inplace modification.
    """
    old_init = deepcopy(attrs[name])
    new_init = _deprecate_func(old_init, msg)
    attrs[name] = new_init


def deprecate_transformer(new_cls: type, name: str = "", msg: str = ""):
    """Deprecate a transformer.

    Create a new class with the old name and keep the docstring.
    Add the 'DEPRECATED' attribute to the class and set to True.

    Args:
        new_cls (type):
            <Transformer> class to duplicate.
        name (str):
            Name of the deprecated transformer to deprecate.
            If not provided, use the transformer.__name__ attribute.
        msg (str):
            Custom warning message.

    Returns (type):
        Some input class but with:
        - New name (cls.__name__)
        - class attribute 'DEPRECATED' set to True
    """
    attrs: dict = new_cls.__dict__.copy()
    if msg.strip() == "":
        current_name: str = new_cls.__name__
        msg = f'"{name}" is deprecated'
        if name and name.strip():
            msg += f' and has been renamed to "{current_name}"'
        msg += "."

    attrs["DEPRECATED"] = True

    # Add the same warning message in the init and in the _transform
    _deprecate_class_attribute(attrs, "__init__", msg)
    _deprecate_class_attribute(attrs, "_transform", msg)

    if (name is None) or (not name.strip()):
        name = new_cls.__name__

    return type(name, (new_cls,), attrs)
