"""Pipeline yaml loader."""

from types import ModuleType
from typing import Callable, Optional, Union

from nlsn.nebula.auxiliaries import extract_kwarg_names
from nlsn.nebula.base import LazyWrapper, Transformer
from nlsn.nebula.pipelines.loop_exploder import explode_loops_in_pipeline
from nlsn.nebula.pipelines.pipelines import TransformerPipeline, is_storage_request
from nlsn.nebula.pipelines.util import create_dict_extra_functions
from nlsn.nebula.storage import nebula_storage as ns

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
If "extra_transformers" is provided it must be a <list> | <tuple> of python
modules or dataclasses where each attribute match the corresponding
transformer name.

E.g.:
********************************************************************************
from dataclasses import dataclass
from my_libray import my_transformer_module


class MyTransformer(Transformer): ...


@dataclass
class ExtraTransformers
    MyTransformer = MyTransformer
    AnotherTransformer = AnotherTransformer

load_pipeline(..., extra_transformers=[my_transformer_module, ExtraTransformers])
********************************************************************************
"""

_cache = {}


def _cache_transformer_packages(ext_transformers: Optional[list], t: Optional[str]):
    """Create a list of transformer packages w/ the right priority."""
    if t is not None:
        if t == "spark":  # pragma: no cover
            from nlsn.nebula import spark_transformers as nebula_transformers
        elif t == "pandas":  # pragma: no cover
            from nlsn.nebula import pandas_polars_transformers as nebula_transformers
        elif t == "polars":  # pragma: no cover
            from nlsn.nebula import pandas_polars_transformers as nebula_transformers
        else:  # pragma: no cover
            raise ValueError
    else:
        from nlsn.nebula import spark_transformers as nebula_transformers

    _cache["transformer_packages"] = (ext_transformers or []) + [nebula_transformers]


def extract_lazy_params(input_params: dict, extra_funcs: dict[str, Callable]) -> dict:
    ret: dict = {}
    for k, v in input_params.items():
        # [6:] 6 is the len of "__fn__" / "__ns__"
        if isinstance(v, str):
            if v.startswith("__fn__"):
                func_name: str = v[6:]
                ret[k] = extra_funcs[func_name]
                continue
            if v.startswith("__ns__"):
                ret[k] = (ns, v[6:])
                continue
        ret[k] = v
    return ret


def _load_transformer(d: dict, **kwargs) -> Optional[Transformer]:
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
    for pkg in _cache["transformer_packages"]:
        if hasattr(pkg, name):
            t = getattr(pkg, name)
            break
    else:
        msg = f'Unknown transformer "{name}". Maybe it is a pandas/polars '
        msg += 'one and you have to set backend="pandas/polars"'
        msg += " in the outermost dictionary configuration?"
        raise NameError(msg)

    try:
        if is_lazy:
            lazy_params: dict = extract_lazy_params(params, kwargs["extra_funcs"])
            t_loaded = LazyWrapper(t, **lazy_params)
        else:
            t_loaded = t(**params)
        desc = d.get("msg")
        # If the description is provided and the base class is Transformer:
        if desc and isinstance(t_loaded, Transformer):
            t_loaded.set_description(desc)
        return t_loaded
    except Exception as e:
        msg = f'{e}\nError while loading "{t.__name__}" with params: {params}'
        raise type(e)(msg)


def _load_generic(o, **kwargs) -> Union[TransformerPipeline, Optional[Transformer]]:
    if "transformer" in o:
        return _load_transformer(o, **kwargs)
    elif "pipeline" in o:
        return _load_pipeline(o, **kwargs)
    # elif ("pipeline" in o) and ("branch" not in o):
    #     return _load_pipeline(o, **kwargs)
    elif is_storage_request(o):
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
            backend=o.get("backend"),
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
        elif is_storage_request(o_pipe).value > 0:
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
        cast_subset_to_input_schema=o.get("cast_subset_to_input_schema", False),
        repartition_output_to_original=o.get("repartition_output_to_original", False),
        coalesce_output_to_original=o.get("coalesce_output_to_original", False),
        branch=o.get("branch"),
        apply_to_rows=o.get("apply_to_rows"),
        allow_missing_columns=o.get("allow_missing_columns"),
        df_input_name=o.get("df_input_name"),
        df_output_name=o.get("df_output_name"),
        backend=o.get("backend"),
        otherwise=otherwise,
    )
    return ret


def load_pipeline(
    o: Union[dict, list, tuple],
    *,
    extra_functions: Optional[
        Union[Callable, list[Callable], dict[str, Callable]]
    ] = None,
    extra_transformers: Optional[ModuleType] = None,
    backend: Optional[str] = None,
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
        extra_transformers (list(python module) | None):
            User modules containing transformers, ordered from highest to
            lowest priority.
        backend (str | None):
            "pandas". "polars" or "spark".
            If not passed, the backend will be set to "spark".
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
        if ("backend" in o) and (backend is None):
            backend = o["backend"]
        elif ("pipeline" in o) and (backend is None):
            backend = o.get("backend")
    elif isinstance(o, (list, tuple)):
        o = {"pipeline": o}
    else:  # pragma: no cover
        raise TypeError("Not understood. The pipeline must be a list or a dict.")

    # Check & load the external transformer packages if needed.
    if extra_transformers is not None:
        if not isinstance(extra_transformers, (list, tuple)):
            raise TypeError(_MSG_ERR_EXTRA_TRANSFORMER)

    _cache_transformer_packages(extra_transformers, backend)

    # Load the user-defined functions.
    extra_funcs = create_dict_extra_functions(extra_functions)

    # Assert that the pipeline is present in the dictionary
    if "pipeline" not in o:
        raise AssertionError("Unable to find the key 'pipeline'.")

    if evaluate_loops:
        o = explode_loops_in_pipeline(o)

    return _load_pipeline(o, extra_funcs=extra_funcs)


if __name__ == "__main__":  # pragma: no cover
    # Just for testing and debugging.
    pipe_cfg = {
        "df_input_name": "HELLO",
        "name": "main-pipeline",
        "pipeline": [
            {"transformer": "EmptyArrayToNull", "params": {"glob": "*"}},
            {"transformer": "Distinct"},
            {
                "branch": {"storage": "df_ads_metadata_input", "end": "append"},
                "pipeline": [
                    {"transformer": "LogDataSkew"},
                    {
                        "transformer": "RoundValues",
                        "params": {"input_columns": "c1", "precision": 1},
                    },
                ],
            },
            {
                "transformer": "RoundValues",
                "params": {"input_columns": "c1", "precision": 1},
            },
        ],
    }

    pipe = load_pipeline(
        pipe_cfg,
        #     extra_functions=extra_functions,
        #     extra_transformers=my_n1_transformer,  # <module>
    )

    pipe.show_pipeline(add_transformer_params=True)
    pipe._print_dag()

    # pipe_example_dict = {
    #     "name": "outer pipe",
    #     "split_function": "split_func_outer",
    #     "pipeline": {
    #         "outer_split_1": {
    #             "name": "nested pipe",
    #             "split_function": "split_func_nested",
    #             "pipeline": {
    #                 # "nested_split_1": None,
    #                 "nested_split_1": [],
    #                 # 'nested_split_1': [{"transformer": "Count"}],
    #                 "nested_split_2": [
    #                     {
    #                         "transformer": "EmptyArrayToNull",
    #                         "params": {"columns": "col_2"},
    #                     }
    #                 ],
    #             },
    #         },
    #         "outer_split_2": [
    #             {
    #                 "transformer": "Cast",
    #                 "params": {"cast": {"col_2": "float"}},
    #             },
    #             {"store": "outer_split_2"},
    #             {"store_debug": "outer_split_2_debug"},
    #         ],
    #     },
    # }
    #
    # dict_split_funcs = {
    #     "split_func_outer": lambda x: x,
    #     "split_func_nested": lambda x: x,
    # }
    #
    # pipe_example = load_pipeline(pipe_example_dict, extra_functions=dict_split_funcs)
    #
    # pipe_example.show_pipeline(add_transformer_params=True)
