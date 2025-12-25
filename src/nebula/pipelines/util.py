"""Pipeline utils."""

from types import FunctionType
from typing import Callable

import narwhals as nw

from nebula.auxiliaries import truncate_long_string
from nebula.base import LazyWrapper, Transformer, is_ns_lazy_request
from nebula.df_types import GenericDataFrame, NwDataFrame, get_dataframe_type
from nebula.nw_util import get_condition, null_cond_to_false, to_native_dataframes
from nebula.pipelines.transformer_type_util import is_transformer

__all__ = [
    "create_dict_extra_functions",
    "get_native_schema",
    "get_pipeline_name",
    "get_transformer_name",
    "is_lazy_transformer",
    "is_plain_transformer_list",
    "sanitize_list_transformers",
    "split_df",
    "to_schema",
]


def create_dict_extra_functions(
        o: Callable | list[Callable] | dict[str, Callable]
) -> dict[str, Callable]:
    """Create a dictionary of extra functions from a list or dictionary.

    Args:
        o (callable, list(callable) | dict(str, callable)):
            Either a list or a dictionary of callable functions.

    Returns (dict(str, callable)):
        A dictionary where keys are function names and values are
        callable functions.

    Raises:
        AssertionError: If there are duplicated function names in the list.
        ValueError: If the input is not a list, tuple, or dictionary.

    Example:
        >>> def func1(df): ...
        >>> def func2(df): ...
        >>> def func3(df): ...
        >>> functions_list = [func1, func2, func3]
        >>> extra_functions_dict = create_dict_extra_functions(functions_list)
        >>> print(extra_functions_dict)
        {
            'func1': <function func1 at 0x...>,
            'func2': <function func2 at 0x...>,
            'func3': <function func3 at 0x...>
        }

        >>> functions_dict = {'foo': func1, 'bar': func2, 'baz': func3}
        >>> extra_functions_dict = create_dict_extra_functions(functions_dict)
        >>> print(extra_functions_dict)
        {
            'foo': <function func1 at 0x...>,
            'bar': <function func2 at 0x...>,
            'baz': <function func3 at 0x...>
        }
    """
    # Check if the input is empty
    if not o:
        return {}

    # Check if the input is a dictionary
    if isinstance(o, dict):
        # Check if all values are callable
        if not all(callable(i) for i in o.values()):
            msg = 'If the "extra_function" is provided as <dict>, '
            msg += "all its values must be <callable>"
            raise AssertionError(msg)
        return o

    if callable(o):
        # It is a plain function / callable class
        return {o.__name__: o}

    # Check if the input is a list or tuple
    if isinstance(o, (list, tuple)):
        # Check if all values are callable
        if not all(callable(i) for i in o):
            msg = 'If the "extra_function" is provided as <list> | <tuple>, '
            msg += "all its values must be <callable>"
            raise AssertionError(msg)

        # Extract function names and check for duplicates
        names = [i.__name__ for i in o]
        if len(names) != len(set(names)):
            raise AssertionError(f"Duplicated function names: {names}")

        # Create a dictionary using function names as keys and functions as values
        return dict(zip(names, o))

    # Raise an error if the input is not a list, tuple, or dictionary
    raise AssertionError(
        f'If "functions" is provided it must be <list> | <tuple> | <dict>. Got {type(o)}'
    )


def is_lazy_transformer(t) -> bool:
    """Check if a transformer instance is lazy."""
    return isinstance(t, LazyWrapper)


def is_plain_transformer_list(lst: list | tuple) -> bool:
    """True if the iterable contains only Transformers (or is empty)."""
    return all(is_transformer(i) for i in lst)


def get_native_schema(df):
    if isinstance(df, (nw.DataFrame, nw.LazyFrame)):
        df = nw.to_native(df)

    if get_dataframe_type(df) == "spark":
        schema = df.schema
    elif get_dataframe_type(df) == "pandas":
        schema = df.dtypes.to_dict()
    elif get_dataframe_type(df) == "polars":
        schema = df.schema
    else:  # pragma: no cover
        raise ValueError("Unsupported dataframe type")

    return schema


def get_pipeline_name(o) -> str:
    """Extract the pipeline name.

    E.g., if the pipeline has <name> return:
    **SplitPipeline**: <name>

    Otherwise, return only the class name
    **SplitPipeline**
    """
    cls_name: str = o.__class__.__name__
    name = f"*** {cls_name} ***"
    if hasattr(o, "name"):
        inner_name = getattr(o, "name")
        if inner_name:
            name += f': "{inner_name}"'

    if hasattr(o, "get_number_transformers"):
        name += f" ({o.get_number_transformers()} transformers)"
    return name


# def _get_transformer_params_formatted(
#         tf_name: str, *, li_attrs: list[str], as_list: bool, max_len: int, wrap_text: bool
# ) -> str | list[str]:
#     if not li_attrs:  # pragma: no cover
#         return [tf_name] if as_list else tf_name
#
#     if wrap_text:
#         ret: str = tf_name + "\nPARAMETERS:\n"
#         if max_len > 0:
#             li_attrs_short = [truncate_long_string(i, max_len) for i in li_attrs]
#             ret += "\n".join(li_attrs_short)
#         else:
#             ret += "\n".join(li_attrs)
#     elif as_list:
#         if max_len > 0:
#             li_attrs = [truncate_long_string(i, max_len) for i in li_attrs]
#         ret: list[str] = [tf_name, *li_attrs]
#     else:
#         str_params: str = truncate_long_string(", ".join(li_attrs), max_len)
#         ret: str = tf_name + " -> PARAMS: " + str_params
#
#     return ret


def _get_transformer_params_formatted(
        *, li_attrs: list[str], as_list: bool, max_len: int, wrap_text: bool
) -> str | list[str]:
    if not li_attrs:  # pragma: no cover
        return [] if as_list else ""

    if wrap_text:
        ret: str = ""
        if max_len > 0:
            li_attrs_short = [truncate_long_string(i, max_len) for i in li_attrs]
            ret += "\n".join(li_attrs_short)
        else:
            ret += "\n".join(li_attrs)
    elif as_list:
        if max_len > 0:
            li_attrs = [truncate_long_string(i, max_len) for i in li_attrs]
        ret: list[str] = li_attrs
    else:
        ret: str = truncate_long_string(", ".join(li_attrs), max_len)

    return ret


def replace_params_references(obj):
    """Recursively parse an iterable and replace nebula storage references.

    Traverses dict, list, tuple structures and replaces any 2-element
    list/tuple where the first element is nebula_storage with a string
    like 'ns.get("key")'.

    Args:
        obj: Any object - will recursively process dicts, lists, tuples.

    Returns:
        The object with all ns references replaced by their string representation.

    Example:
        >>> data = {'data': [{'alias': 'c5', 'value': (ns, 'my_key')}]}
        >>> replace_params_references(data)
        {'data': [{'alias': 'c5', 'value': 'ns.get("my_key")'}]}
    """
    # Check for ns lazy request FIRST (before generic list/tuple handling)
    if is_ns_lazy_request(obj):
        return f'ns.get("{obj[1]}")'

    if isinstance(obj, FunctionType):
        return obj.__name__

    # Recurse into dictionaries
    if isinstance(obj, dict):
        return {k: replace_params_references(v) for k, v in obj.items()}

    # Recurse into lists
    if isinstance(obj, list):
        return [replace_params_references(item) for item in obj]

    # Recurse into tuples (preserve tuple type)
    if isinstance(obj, tuple):
        return tuple(replace_params_references(item) for item in obj)

    # Base case: return as-is
    return obj


#
# def get_transformer_name(
#         obj: Transformer | LazyWrapper,
#         *,
#         add_params: bool = False,
#         max_len: int = 80,
#         wrap_text: bool = False,
#         as_list: bool = False,
# ) -> str | list[str]:
#     """Get the name of a transformer object.
#
#     Args:
#         obj (Transformer):
#             The transformer object.
#         add_params (bool):
#             If True, include transformer initialization
#             parameters in the name.
#         max_len (int):
#             When 'as_list' is set to False:
#                 After converting the transformer input parameters to a single
#                 string, truncate the characters in the middle in order not to
#                 exceede 'max_len'.
#             When 'as_list' is set to True:
#                 Each string-parameter in the list is truncated at 'max_len'.
#             If max_len <= 0, the parameter is ignored. Defaults to 80.
#         wrap_text (bool):
#             If True, the string of parameters will be returned as wrapped text,
#             creating a new line for each parameter. In this case, the
#             'max_len' parameter is ignored.
#             This behavior is only applicable if the 'add_params' parameter
#             is set to True.
#             Defaults to False.
#         as_list (bool):
#             If True, the name of the transformer and the parameters are
#             returned as strings in a list. In this case, the 'max_len'
#             parameter refers to each string in the output list.
#             This behavior is only applicable if the 'add_params' parameter
#             is set to True.
#             The trasformer name will always be the first element of the list.
#             Defaults to False.
#
#     Returns (str | list(str)):
#         (str): The transformer name, possibly including initialization
#             parameters.
#         list(str): list containing the transformer name (first element)
#             and the parameters as strings.
#
#     Raises:
#         AssertionError: If both wrap_text and as_list are True.
#
#     Example:
#         >>> class CustomTransformer(Transformer):
#         >>>     def _transform(self, df): ...
#         >>> my_transformer = CustomTransformer(param1=42, param2="example")
#         >>> name = get_transformer_name(my_transformer, add_params=True)
#         >>> print(name)
#         'CustomTransformer -> PARAMS: param1=42, param2="example"'
#     """
#     if add_params and wrap_text and as_list:
#         raise ValueError('"wrap_text" and "as_list" cannot be both True.')
#
#     is_lazy = is_lazy_transformer(obj)
#
#     tf_name: str = f"(Lazy) {obj.trf.__name__}" if is_lazy else obj.__class__.__name__
#
#     if not add_params:
#         return [tf_name] if as_list else tf_name
#
#     li_attrs: list[str] = []
#     v_show: str
#
#     if is_lazy:
#         tf_attrs = obj.kwargs
#         if not tf_attrs:
#             return [tf_name] if as_list else tf_name
#
#         try:
#             # Extract and format call parameters
#             for k, v in sorted(tf_attrs.items()):
#                 if is_lazy_function(v):
#                     v_show = f"{v.__name__}()"
#                 else:
#                     v_processed = replace_ns_references(v)
#                     v_show = f'"{v_processed}"' if isinstance(v_processed, str) else v_processed
#
#                 li_attrs.append(f"{k}={v_show}")
#         except:  # noqa PyBroadException  pragma: no cover
#             return [tf_name] if as_list else tf_name
#
#     else:
#         tf_attrs = getattr(obj, "transformer_init_parameters", {})
#         if not tf_attrs:
#             return [tf_name] if as_list else tf_name
#
#         try:
#             # Extract and format initialization parameters
#             for k, v in sorted(tf_attrs.items()):
#                 v_show = f'"{v}"' if isinstance(v, str) else v
#                 li_attrs.append(f"{k}={v_show}")
#         except:  # noqa PyBroadException  pragma: no cover
#             return [tf_name] if as_list else tf_name
#
#     return _get_transformer_params_formatted(
#         tf_name,
#         li_attrs=li_attrs,
#         as_list=as_list,
#         max_len=max_len,
#         wrap_text=wrap_text,
#     )


def get_transformer_name(
        obj: Transformer | LazyWrapper,
        *,
        add_params: bool = False,
        max_len: int = 80,
        wrap_text: bool = False,
        as_list: bool = False,
) -> str | list[str]:
    """Get the name of a transformer object.

    Args:
        obj (Transformer):
            The transformer object.
        add_params (bool):
            If True, include transformer initialization
            parameters in the name.
        max_len (int):
            When 'as_list' is set to False:
                After converting the transformer input parameters to a single
                string, truncate the characters in the middle in order not to
                exceede 'max_len'.
            When 'as_list' is set to True:
                Each string-parameter in the list is truncated at 'max_len'.
            If max_len <= 0, the parameter is ignored. Defaults to 80.
        wrap_text (bool):
            If True, the string of parameters will be returned as wrapped text,
            creating a new line for each parameter. In this case, the
            'max_len' parameter is ignored.
            This behavior is only applicable if the 'add_params' parameter
            is set to True.
            Defaults to False.
        as_list (bool):
            If True, the name of the transformer and the parameters are
            returned as strings in a list. In this case, the 'max_len'
            parameter refers to each string in the output list.
            This behavior is only applicable if the 'add_params' parameter
            is set to True.
            The trasformer name will always be the first element of the list.
            Defaults to False.

    Returns (str | list(str)):
        (str): The transformer name, possibly including initialization
            parameters.
        list(str): list containing the transformer name (first element)
            and the parameters as strings.

    Raises:
        AssertionError: If both wrap_text and as_list are True.

    Example:
        >>> class CustomTransformer(Transformer):
        >>>     def _transform(self, df): ...
        >>> my_transformer = CustomTransformer(param1=42, param2="example")
        >>> name = get_transformer_name(my_transformer, add_params=True)
        >>> print(name)
        'CustomTransformer -> PARAMS: param1=42, param2="example"'
    """
    if add_params and wrap_text and as_list:
        raise ValueError('"wrap_text" and "as_list" cannot be both True.')

    is_lazy = is_lazy_transformer(obj)

    if not add_params:
        return [] if as_list else ""

    li_attrs: list[str] = []
    v_show: str

    if is_lazy:
        tf_attrs = obj.kwargs
        if not tf_attrs:
            return [] if as_list else ""

        try:
            # Extract and format call parameters
            for k, v in sorted(tf_attrs.items()):
                v_processed = replace_params_references(v)
                v_show = f'"{v_processed}"' if isinstance(v_processed, str) else v_processed

                li_attrs.append(f"{k}={v_show}")
        except Exception as e:  # noqa PyBroadException  pragma: no cover
            return [] if as_list else ""

    else:
        tf_attrs = getattr(obj, "transformer_init_parameters", {})
        if not tf_attrs:
            return [] if as_list else ""

        try:
            # Extract and format initialization parameters
            for k, v in sorted(tf_attrs.items()):
                v_show = f'"{v}"' if isinstance(v, str) else v
                li_attrs.append(f"{k}={v_show}")
        except:  # noqa PyBroadException  pragma: no cover
            return [] if as_list else ""

    return _get_transformer_params_formatted(
        li_attrs=li_attrs,
        as_list=as_list,
        max_len=max_len,
        wrap_text=wrap_text,
    )


def sanitize_list_transformers(transformers) -> list[Transformer]:
    """Ensure that the input object is a flat list of transformers.

    Args:
        transformers (Transform | list(Transformer) | tuple(transformer)):
            Input list.

    Returns (list(Transformer)):
        Flat list of transformers.

    Raises: ValueError if a not allowed type is passed.
    """
    ret: list[Transformer]
    if transformers:
        if is_transformer(transformers):
            ret = [transformers]
        else:
            msg = 'Argument  must be a <Transformer> or a <list<Transformer>>'
            if not isinstance(transformers, (list, tuple)):
                raise ValueError(msg)
            if not all(is_transformer(i) for i in transformers):
                raise ValueError(msg)
            if isinstance(transformers, tuple):
                ret = list(transformers)
            else:
                ret = transformers
    else:
        ret = []
    return ret


def split_df(df, cfg: dict) -> tuple[NwDataFrame, NwDataFrame]:
    """Split a dataframe according to the given configuration."""
    if not isinstance(df, (nw.DataFrame, nw.LazyFrame)):
        df = nw.from_native(df)

    cond = get_condition(
        col_name=cfg.get("input_col"),
        operator=cfg.get("operator"),
        value=cfg.get("value"),
        compare_col=cfg.get("comparison_column"),
    )
    otherwise_cond = ~null_cond_to_false(cond)

    return df.filter(cond), df.filter(otherwise_cond)


def to_schema(li_df: list, schema) -> list[GenericDataFrame]:
    """Cast a list of dataframes to a schema."""
    native_dataframes, native_backend, nw_found = to_native_dataframes(li_df)

    if native_backend == "pandas":
        ret = [_df.astype(schema) for _df in native_dataframes]

    elif native_backend == "polars":
        ret = [_df.cast(schema) for _df in native_dataframes]

    elif native_backend == "spark":
        from nebula.spark_util import cast_to_schema

        ret = [cast_to_schema(i, schema) for i in native_dataframes]

    else:  # pragma: no cover
        raise ValueError(f"Unsupported dataframe type: {native_backend}")

    return [nw.from_native(i) for i in ret] if nw_found else ret
