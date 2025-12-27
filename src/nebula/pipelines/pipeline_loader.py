"""Text pipeline loader."""

from dataclasses import dataclass
from types import ModuleType
from typing import Callable

from nebula.auxiliaries import extract_kwarg_names
from nebula.base import LazyWrapper, Transformer
from nebula.pipelines.loop_expansion import expand_loops
from nebula.pipelines.pipe_aux import create_dict_extra_functions, is_keyword_request
from nebula.pipelines.pipeline import TransformerPipeline
from nebula.storage import nebula_storage as ns

__all__ = ["load_pipeline"]

# def _split_function_example(df):
#     """Example of split-function."""
#     df_1 = ...
#     df_2 = ...
#     return {"split_1": df_1, "split_2": df_2}

# Set the forbidden split names in YAML file ...
_NOT_ALLOWED_SPLIT_NAMES: set[str] = {
    "loop",
    "pipeline",
    "store",
    "store_debug",
    "storage_debug_mode",
    "replace_with_stored_df",
    "transformer",
}

# ... extract the kwarg name from "TransformerPipeline" ...
_kws_pipeline: set[str] = set(extract_kwarg_names(TransformerPipeline))
if not _kws_pipeline:  # pragma: no cover
    raise RuntimeError("Unable to detect kwarg name in the 'TransformerPipeline'")

# and update the forbidden split names in the YAML file.
_NOT_ALLOWED_SPLIT_NAMES.update(_kws_pipeline)
_allowed_kws_pipeline = {"pipeline", "data"}.union(_kws_pipeline)

_MSG_ERR_EXTRA_TRANSFORMER = """
If "extra_transformers" is provided it must be a <dict<str, Transformer>>
 or a <list/tuple<ModuleType | DataClass>> where each
attribute match the corresponding transformer name.

E.g. with ModuleType and DataClass:
********************************************************************************
from dataclasses import dataclass
from my_libray import my_transformer_module


class MyTransformer(Transformer): ...


@dataclass
class ExtraTransformers:
    MyTransformer = MyTransformer
    AnotherTransformer = AnotherTransformer

load_pipeline(..., extra_transformers=[my_transformer_module, ExtraTransformers])
********************************************************************************
"""

_native_transformers: dict[str, type] = {}
_full_transformers: dict[str, type] = {}


def _cache_transformers(container) -> dict[str, type]:
    from nebula.pipelines.transformer_type_util import is_transformer
    ret = {}
    for name in dir(container):
        obj = getattr(container, name)
        if is_transformer(obj):
            ret[name] = obj
    return ret


def __load_native():
    if not _native_transformers:
        from nebula import transformers as nebula_transformers
        _native_transformers.update(_cache_transformers(nebula_transformers))


def _cache_transformer_packages(ext_transformers: list | dict | None):
    """Load the transformers with the right priority."""
    __load_native()
    _full_transformers.clear()
    _full_transformers.update(_native_transformers)

    if not ext_transformers:
        return

    if isinstance(ext_transformers, dict):
        _full_transformers.update(ext_transformers)
        return

    for container in ext_transformers[::-1]:  # reverse for priority
        _full_transformers.update(_cache_transformers(container))


def _resolve_lazy_string_marker(obj):
    """Recursively resolve lazy string markers in nested structures.

    This function traverses nested dicts, lists, and tuples, converting:
    - "__ns__<key>" strings: to (ns, "<key>") tuples for lazy storage access

    These markers are the YAML/JSON-serializable equivalents of the Python API's
    lazy references (decorated functions and (ns, "key") tuples).

    Args:
        obj: Any value that may contain lazy string markers at any nesting level.

    Returns:
        The processed value with all string markers converted to lazy references.

    Example (YAML input):
        data:
          - alias: "col1"
            value: "__ns__stored_value"

        Becomes:
        data:
          - alias: "col1"
            value: (ns, "stored_value")  # Will be resolved at transform time
    """
    # Check for lazy string markers
    if isinstance(obj, str):
        # [6:] because len("__ns__") == 6
        if obj.startswith("__ns__"):
            # Return as tuple for lazy resolution at transform time
            return ns, obj[6:]
        return obj

    # Recurse into dictionaries
    if isinstance(obj, dict):
        return {k: _resolve_lazy_string_marker(v) for k, v in obj.items()}

    # Recurse into lists
    if isinstance(obj, list):
        return [_resolve_lazy_string_marker(item) for item in obj]

    # Recurse into tuples
    if isinstance(obj, tuple):
        return tuple(_resolve_lazy_string_marker(item) for item in obj)

    # Base case: return value as-is
    return obj


def extract_lazy_params(input_params: dict) -> dict:
    """Extract and convert lazy string markers from YAML/JSON parameters.

    This is the entry point for processing lazy parameters loaded from
    YAML or JSON configuration files. It recursively processes all values,
    converting string markers to their lazy equivalents.

    Args:
        input_params: Dictionary of parameters that may contain lazy string
                      markers at any nesting depth.

    Returns:
        New dictionary with all lazy string markers converted.

    Example:
        >>> extra_funcs = {"get_threshold": lambda: 0.5}
        >>> params = {
        ...     "simple": "static_value",
        ...     "from_storage": "__ns__my_key",
        ...     "nested": {
        ...         "deep": [{"value": "__ns__nested_key"}]
        ...     }
        ... }
        >>> extract_lazy_params(params)
        {
            "simple": "static_value",
            "from_storage": (ns, "my_key"),
            "nested": {
                "deep": [{"value": (ns, "nested_key")}]
            }
        }
    """
    return _resolve_lazy_string_marker(input_params)


def _load_transformer(d: dict) -> Transformer | None:
    if d.get("skip") or (d.get("perform") is False):
        return None
    name: str = d["transformer"]
    params: dict = d.get("params")
    is_lazy: bool = bool(d.get("lazy", False))
    if params is None:
        # don't rely on .get("params", {}) because if "params" is None
        # it throws an error when **unpacking.
        params = {}

    # Iterate through the packages and stop as soon as the transformer is found.
    t = _full_transformers.get(name)
    if t is None:
        searched = sorted(_full_transformers)
        raise NameError(f'Unknown transformer "{name}". Searched: {searched}')

    try:
        if is_lazy:
            lazy_params: dict = extract_lazy_params(params)
            t_loaded = LazyWrapper(t, **lazy_params)
        else:
            t_loaded = t(**params)
        desc = d.get("description")
        # If the description is provided and the base class is Transformer:
        if desc and isinstance(t_loaded, Transformer):
            t_loaded.set_description(desc)
        return t_loaded
    except Exception as e:
        msg = f'{e}\nError while loading "{t.__name__}" with params: {params}'
        raise type(e)(msg)


def _load_generic(o, **kwargs) -> TransformerPipeline | Transformer | dict:
    if "transformer" in o:
        return _load_transformer(o)
    elif "pipeline" in o:
        return _load_pipeline(o, **kwargs)
    elif is_keyword_request(o):
        return o
    else:  # pragma: no cover
        msg = "Not understood. At this stage the loader is looking "
        msg += f"for 'transformer' or 'branch' or 'pipeline'. Passed {o}."
        raise TypeError(msg)


def _load_objects(o, **kwargs):
    if o is None or o is False:
        return None
    elif isinstance(o, dict):
        return _load_generic(o, **kwargs)
    elif isinstance(o, (tuple, list)):
        ret = []
        for i in o:
            t = _load_generic(i, **kwargs)
            # Do not append transformer when skip = True
            if t:
                ret.append(t)
        return ret
    else:  # pragma: no cover
        x = type(o)
        msg = "Not understood. At this stage the loader is looking for a "
        msg += f"<False | None | list | tuple | dict>. Found: {x} Passed {o}."
        raise TypeError(msg)


def _extract_function(li: list[Callable], name: str) -> Callable:
    names = [i.__name__ for i in li]
    if len(names) != len(set(names)):
        raise AssertionError(f"Duplicated function names: {names}")
    d = dict(zip(names, li))
    return d[name]


def _load_pipeline(o, *, extra_funcs) -> TransformerPipeline:
    provided_keys = set(o.keys())
    if not provided_keys.issubset(_allowed_kws_pipeline):
        diff = provided_keys.difference(_allowed_kws_pipeline)
        msg = (
            f"Unknown kwargs {diff} (provided keys={provided_keys}). "
            f"Allowed arguments: {_allowed_kws_pipeline}"
        )
        raise AssertionError(msg)

    pipeline_name: str = o.get("name")
    skip_pipeline: bool = bool(o.get("skip")) or (o.get("perform") is False)
    if skip_pipeline:
        ret = TransformerPipeline(
            [],
            name=pipeline_name,
            df_input_name=o.get("df_input_name"),
            df_output_name=o.get("df_output_name"),
            skip=True,
        )
        return ret

    # Get the pipeline dictionary
    o_pipe = o["pipeline"]

    # Check if it contains a split_function. If so, it is considered
    # a split-pipeline.
    split_func_name: str = o.get("split_function")
    if split_func_name:
        if not isinstance(o_pipe, dict):
            msg = f'A split function has been provided to "{pipeline_name}", '
            msg += 'so it is supposed to be a "split-pipeline", but in this '
            msg += 'case the value of "pipeline" must be a <dict>. '
            msg += f"Found {type(o_pipe)}"
            raise AssertionError(msg)

        keys = set(o_pipe)

        diff = _NOT_ALLOWED_SPLIT_NAMES.difference(keys)
        if diff != _NOT_ALLOWED_SPLIT_NAMES:
            found = _NOT_ALLOWED_SPLIT_NAMES.intersection(keys)
            msg = "Not allowed split names when loading from text source: "
            msg += f"{_NOT_ALLOWED_SPLIT_NAMES}.\nGot: {found}"
            raise AssertionError(msg)

        split_func = extra_funcs[split_func_name]
    else:
        split_func = None

    if isinstance(o_pipe, dict):
        if "transformer" in o_pipe:  # split w/ a single transformer
            input_pipe_data = _load_objects([o_pipe])
        elif is_keyword_request(o_pipe):
            input_pipe_data = o_pipe
        else:  # split pipeline
            input_pipe_data = {}
            for k, v in o_pipe.items():
                if not v:
                    input_pipe_data[k] = []
                    continue
                t = _load_objects(v, extra_funcs=extra_funcs)
                # do not append transformer when skip = True
                if not (t is None or t is False):
                    # Keep if it is an empty list
                    input_pipe_data[k] = t
    elif isinstance(o_pipe, (list, tuple)):
        # Linear pipeline
        input_pipe_data = _load_objects(o_pipe, extra_funcs=extra_funcs)

    else:  # pragma: no cover
        x = type(o)
        msg = "Not understood. At this stage the loader is looking for a "
        msg += f"<list | tuple | dict>. Found: {x} Passed {o}."
        raise TypeError(msg)

    interleaved = _load_objects(o.get("interleaved"))

    # These are considered only in a split-pipeline.
    split_apply_after_splitting = _load_objects(o.get("split_apply_after_splitting"))
    split_apply_before_appending = _load_objects(o.get("split_apply_before_appending"))

    otherwise = o.get("otherwise")
    if otherwise:
        otherwise = _load_objects(otherwise, extra_funcs=extra_funcs)

    ret = TransformerPipeline(
        input_pipe_data,
        name=pipeline_name,
        split_function=split_func,
        interleaved=interleaved,
        prepend_interleaved=o.get("prepend_interleaved", False),
        append_interleaved=o.get("append_interleaved", False),
        split_apply_after_splitting=split_apply_after_splitting,
        split_apply_before_appending=split_apply_before_appending,
        splits_no_merge=o.get("splits_no_merge"),
        splits_skip_if_empty=o.get("splits_skip_if_empty"),
        cast_subsets_to_input_schema=o.get("cast_subsets_to_input_schema", False),
        repartition_output_to_original=o.get("repartition_output_to_original", False),
        coalesce_output_to_original=o.get("coalesce_output_to_original", False),
        branch=o.get("branch"),
        apply_to_rows=o.get("apply_to_rows"),
        allow_missing_columns=o.get("allow_missing_columns"),
        df_input_name=o.get("df_input_name"),
        df_output_name=o.get("df_output_name"),
        otherwise=otherwise,
    )
    return ret


def load_pipeline(
        o: dict | list | tuple,
        *,
        extra_functions: Callable | list[Callable] | dict[str, Callable] | None = None,
        extra_transformers: list[ModuleType] | list[dataclass] | dict | None = None,
        evaluate_loops: bool = True,
) -> TransformerPipeline:
    """Load a Nebula pipeline object starting from a dictionary.

    Args:
        o (dict(str, any), list(amy)):
            The input dictionary / list defining the pipeline.
        extra_functions (callable, list(callable), dict(str, callable) | None):
            List of user-defined split functions.
            Provide as a list of callables (e.g.,
            [split_func_1, split_func_2]), ensuring they have the same
            names as the split pipelines.
            Alternatively, use a dictionary format (e.g.,
            {"split_1": func_1, "split_2": func_2}), where keys must match
            the split pipeline names.
        extra_transformers (list(python module) | list(dataclass) | dict(str, T) | None):
            Custom transformers, if passed al list of module | dataclasses,
            they must be ordered from highest to lowest priority in case
            duplicated name.
        evaluate_loops (bool):
            If `True`, the parser will search for and evaluate for-loops within
            the pipelines. This is the safest option, and if no loops are present,
            the final outcome remains correct.
            For a potential speed increase, you can set this to `False` to skip
            the parsing step entirely.
            Warning: Only set to `False` if you are certain the pipelines contain
            no for-loops. If loops are present, they will be silently ignored,
            which may lead to unexpected results.
            Defaults to `True`.

    Example:
    ```
    from my_lib_1 import transformers as lib_1_transformers
    import my_lib_2.transformers

    # The first element has higher priority over the last one.
    load_pipeline(
        ...,
        functions=...,
        extra_transformers=[lib_1_transformers, my_lib_2.transformers]
    )
    ```

    Returns (TransformerPipeline):
        The resulting TransformerPipeline object.
    """
    # Check the input type
    if not isinstance(o, (dict, list, tuple)):
        msg = f"Input type must be <dict> | <list> | <tuple>. Found <{type(o)}>"
        raise TypeError(msg)

    # Transform the input type from <list> / <tuple> to <dict>.
    if isinstance(o, dict):
        pass
    elif isinstance(o, (list, tuple)):
        o = {"pipeline": o}
    else:  # pragma: no cover
        raise TypeError("Not understood. The pipeline must be a list or a dict.")

    # Check & load the external transformer packages if needed.
    if extra_transformers is not None:
        if not isinstance(extra_transformers, (list, tuple, dict)):
            raise TypeError(_MSG_ERR_EXTRA_TRANSFORMER)

    _cache_transformer_packages(extra_transformers)

    # Load the user-defined functions.
    extra_funcs = create_dict_extra_functions(extra_functions)

    # Assert that the pipeline is present in the dictionary
    if "pipeline" not in o:
        raise AssertionError("Unable to find the key 'pipeline'.")

    if evaluate_loops:
        o = expand_loops(o)

    return _load_pipeline(o, extra_funcs=extra_funcs)


if __name__ == "__main__":  # pragma: no cover
    pipe_cfg = {
        "df_input_name": "START",
        "name": "main-pipeline",
        "pipeline": [
            {"transformer": "SelectColumns", "params": {"glob": "cx_*"}},
            {"transformer": "AssertNotEmpty"},
            {
                "branch": {"storage": "df_ads_metadata_input", "end": "append"},
                "pipeline": [
                    {"transformer": "LogDataSkew"},
                    {"transformer": "DropColumns", "params": {"columns": "cx_0"}},
                ],
            },
        ],
    }
    pipe = load_pipeline(
        pipe_cfg,
        #     extra_functions=extra_functions,
        #     extra_transformers=my_n1_transformer,
    )
    pipe.show(add_params=True)
