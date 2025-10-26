"""Pipelines module.

This file was written in a few days and needs refactoring.
"""

from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from time import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

from nlsn.nebula.auxiliaries import (
    assert_allowed,
    assert_at_most_one_args,
    ensure_flat_list,
    get_class_name,
)
from nlsn.nebula.base import LazyWrapper, Transformer
from nlsn.nebula.df_types import GenericDataFrame, get_dataframe_type
from nlsn.nebula.logger import logger
from nlsn.nebula.pipelines._checks import *
from nlsn.nebula.pipelines._dag import create_dag, print_dag
from nlsn.nebula.pipelines.auxiliaries import *
from nlsn.nebula.pipelines.exceptions import *
from nlsn.nebula.pipelines.transformer_type_util import is_transformer
from nlsn.nebula.pipelines.util import *
from nlsn.nebula.storage import nebula_storage as ns

from ._df_funcs import *

__all__ = [
    "PipeType",
    "set_max_len_string_param",
    "TransformerPipeline",
    "pipeline_config",
]

pipeline_config: dict = {
    "max_len_string_param": 80,
    "activate_failure_cache": True,
}

_PREFIX_FAIL_CACHE: str = "FAIL_DF_"


def set_max_len_string_param(n: int) -> None:  # pragma: no cover
    """Set the maximum length displayed for transformer initialization parameters.

    Default 80.
    """
    if n < 10:
        raise AssertionError('"n" must be >= 10')
    pipeline_config["max_len_string_param"] = n


PipeType = Union[
    Transformer,
    List[Transformer],
    LazyWrapper,
    List[LazyWrapper],
    List[Dict[str, str]],  # Storage requests & hooks
    "TransformerPipeline",
    List["TransformerPipeline"],
]

TransformerOrTransformerList = Union[None, Transformer, List[Transformer]]

_FAIL_CACHE: Dict[Tuple[str, str], "GenericDataFrame"] = {}


def _update_fail_name(name: str) -> str:
    if not ns.isin(name):
        return name

    for i in range(100):
        if ns.isin(f"{name}_{i}"):
            continue
        name = f"{name}_{i}"
        break
    else:
        raise PipelineError("Unable to properly store the fail dataframe(s).")
    return name


def _cache_to_nebula_storage() -> Tuple[str, str, List[str]]:
    li_names: List[str] = []
    li_keys: List[str] = []
    set_types: Set[str] = set()
    for (t, name), df in sorted(_FAIL_CACHE.items()):
        li_names.append(name)
        set_types.add(t)
        key = _PREFIX_FAIL_CACHE + name
        key = _update_fail_name(key)
        ns.set(key, df)
        li_keys.append(key)

    if len(set_types) != 1:
        raise AssertionError(f"Unexpected number of types {set_types}")

    t_out: str = list(set_types)[0]

    if t_out == "split":
        names = ", ".join([f'"{i}"' for i in li_names])
        keys_str = ", ".join([f'"{i}"' for i in li_keys])
        msg = "Unable to union by field names the dataframes. "
        msg += f"Find the output dataframes of each single split {names} in "
        msg += f"the nebula storage by using the keys {keys_str} respectively."
    elif t_out == "transformer":
        n = len(li_names)
        if n != 1:
            raise AssertionError(f"Found {n} keys, expected 1")
        name = li_names[0]
        key = li_keys[0]
        msg = "Find the input dataframe of the failed transformer "
        msg += f'"{name}" in the nebula storage by using the key "{key}".'
    elif t_out in {"branch", "apply_to_rows"}:
        keys_str = ", ".join([f'"{i}"' for i in li_keys])
        msg = f"Unable to complete the '{t_out}'. "
        msg += "Find the input / output dataframes in "
        msg += f"the nebula storage by using the keys {keys_str} respectively."

    else:  # pragma: no cover
        raise AssertionError(f"Unexpected type: {t_out}")
    return t_out, msg, li_keys


def _make_error_msg(o) -> str:
    t_err = type(o)
    add_msg = f" with keys: {set(o)}" if isinstance(o, dict) else ""
    msg = MSG_NOT_UNDERSTOOD
    msg += "\n\n"
    msg += f"Found a {t_err}{add_msg}"
    return msg


def _get_n_partitions(df: "GenericDataFrame", obj: PipeType) -> int:
    """Get the number of partitions if the DF is a spark one and if requested."""
    n = 0
    if obj.repartition_output_to_original or obj.coalesce_output_to_original:
        if get_dataframe_type(df) == "spark":
            n = df.rdd.getNumPartitions()
    return n


def _repartition_coalesce(df, obj: PipeType, n: int) -> "GenericDataFrame":
    """Repartition / coalesce if "df" is a spark DF and if requested."""
    if obj.repartition_output_to_original:
        if get_dataframe_type(df) == "spark":
            df = df.repartition(n)
    elif obj.coalesce_output_to_original:
        if get_dataframe_type(df) == "spark":
            df = df.coalesce(n)
    return df


def _remove_last_transformers(li: List[Transformer], n: int) -> None:
    """Remove the last 'n' Transformers from the list but keep the storage requests.

    It is used when the 'interleaved' input is set, but the
    'append_interleaved' is set to False.

    Inplace modification of 'li'.

    Args:
        li (list(Transformer)):
            The list of Transformer objects.
        n (int):
            The number of Transformer objects to remove.

    Raises:
        AssertionError: If 'n' is greater than or equal to the length of the list.
        AssertionError: If an unexpected element is encountered in the list.
        AssertionError: If everything in the list has been removed.

    Example:
        >>> t1 = ...
        >>> t2 = ...
        >>> t3 = ...
        >>> storage_request = {"store": "x"}
        >>> transformers_list = [t1, t2, storage_request, t3]
        >>> _remove_last_transformers(transformers_list, 2)
        >>> print(transformers_list)
        [t1, t2, t3]
    """
    full_len = len(li)
    if not full_len:  # It was an empty list (allowed)
        return

    if full_len <= n:
        raise AssertionError('"n" >= list length')

    count = 0

    # Iterate through the list in reverse order
    for i in range(len(li) - 1, -1, -1):
        obj = li[i]
        if is_transformer(obj):
            # Remove the integer and increase the count
            li.pop(i)
            count += 1

            # Stop when the desired number of integers is removed
            if count == n:
                break
        elif is_storage_request(obj).value > 0:
            # It is related to nebula storage, do nothing.
            continue
        else:
            raise AssertionError(f"Not expected: {obj}")

    if full_len == count:  # pragma: no cover
        raise AssertionError("Everything has been removed")


def _handle_storage(
    _storage_request, d: Dict[str, Union[str, bool]], df
) -> "GenericDataFrame":
    if _storage_request == StoreRequest.STORE_DF:
        key, msg = get_store_key_msg(d)
        logger.info(f"   --> {msg}")
        ns.set(key, df)
    elif _storage_request == StoreRequest.STORE_DF_DEBUG:
        key, msg = get_store_debug_key_msg(d)
        logger.info(f"   --> {msg}")
        ns.set(key, df, debug=True)
    elif _storage_request == StoreRequest.ACTIVATE_DEBUG:
        logger.info(f"   --> {MSG_ACTIVATE_DEBUG_MODE}")
        ns.allow_debug(True)
    elif _storage_request == StoreRequest.DEACTIVATE_DEBUG:
        logger.info(f"   --> {MSG_DEACTIVATE_DEBUG_MODE}")
        ns.allow_debug(False)
    elif _storage_request == StoreRequest.REPLACE_WITH_STORED_DF:
        key, msg = get_replace_with_stored_df_msg(d)
        logger.info(f"   --> {msg}")
        df = ns.get(key)
    else:  # pragma: no cover
        raise ValueError("Unknown Enum in _StoreRequest")
    return df


def __transform(
    df: "GenericDataFrame", trf: Transformer, backend: Optional[str]
) -> "GenericDataFrame":
    _FAIL_CACHE.clear()

    name_full: str = get_transformer_name(
        trf, add_params=True, max_len=pipeline_config["max_len_string_param"]
    )
    name: str = get_transformer_name(trf, add_params=False)
    logger.info(f"Running {name_full} ...")

    # Store into the cache before transforming
    _FAIL_CACHE[("transformer", name)] = df

    t_start: float = time()
    available_backends: Set[str] = getattr(trf, "backends", set())

    try:
        if backend == "spark" and backend in available_backends:
            df = trf.transform_spark(df)
        elif backend == "pandas" and backend in available_backends:
            df = trf.transform_pandas(df)
        elif backend == "polars" and backend in available_backends:
            df = trf.transform_polars(df)
        else:
            df = trf.transform(df)

    except Exception as e:
        if not pipeline_config["activate_failure_cache"]:
            raise e
        _, msg, _ = _cache_to_nebula_storage()
        # raise PipelineError(msg) from e
        raise_pipeline_error(e, msg)
    t_run: float = time() - t_start

    logger.info(f"Execution time for {name}: {t_run:.1f}s")
    _FAIL_CACHE.clear()
    return df


def _transform(
    stages: List[Transformer],
    df: "GenericDataFrame",
    backend: Optional[str],
    forced_trf: Optional[Transformer],
) -> "GenericDataFrame":
    """Apply the actual transformation and log the time."""
    t: Transformer
    for t in stages:
        df = __transform(df, t, backend)

        if forced_trf is not None:
            df = __transform(df, forced_trf, backend)

    return df


def __names_in_iterable(obj) -> str:  # pragma: no cover
    """Extracts and formats names from iterable objects.

    Just for logger.debug.

    Args:
        obj (Iterable):
            The iterable containing objects, possibly including
            TransformerPipeline instances.
    Returns (str):
        A comma-separated string of formatted names extracted from the input.
    """
    names = []
    for el in obj:
        if isinstance(el, TransformerPipeline):
            name = getattr(el, "name", None)
            cls_name: str = el.__class__.__name__
            if name:
                name = f"{name} <{cls_name}>"
            else:
                name = cls_name
        else:
            name = get_class_name(el)[0]
        names.append(str(name))
    return ", ".join(names)


def _get_fork_header(name: str, d) -> Tuple[str, str]:
    show_dict = {k: v for k, v in d.items() if k != "storage"}
    sep = "\n  "
    new_lines = sep.join([f"- {k}: {v}" for k, v in show_dict.items()])
    header = f"------ {name.upper()} ------{sep}{new_lines}"

    input_storage = d.get("storage")
    msg_storage = ""
    if input_storage:
        msg_storage = f"Input storage key: {input_storage}"

    return header, msg_storage


def _create_stages(
    obj: PipeType,
    interleaved: Optional[List[Transformer]],
    prepend_interleaved: bool,
    stages=None,
    count_transformers: int = 0,
) -> Tuple[List[PipeType], int]:
    """Create pipeline stages.

    Given an object of type 'PipeType', create the stages for the pipeline
    adding the 'interleaved' transformers and prepending them if requested.

    Args:
        obj: (PipeType)
            Pipeline object.
        interleaved: (list(Transformer) | None)
            Transformers to be inserted in between.
        prepend_interleaved: (bool)
            If True, prepend the 'interleaved' to the output list.
        stages: /
            Initialization parameter for the recursion, don't provide.

    Returns: (list(PipeType))
        List of stages.

    Raises: TypeError if 'obj' is not recognized as a valid type.
    """
    n_interleaved: int = len(interleaved) if interleaved else 0

    if stages is None:
        # Init stages if empty
        stages = []
        if prepend_interleaved and interleaved:
            stages.extend(interleaved)
            count_transformers += n_interleaved

    # Parse the input
    if is_transformer(obj):  # The input is a transformer
        stages.append(obj)
        # in between is a list(Transformer), cannot be None or a Transformer
        stages.extend(interleaved)
        count_transformers += n_interleaved + 1
        return stages, count_transformers

    elif isinstance(obj, TransformerPipeline):  # The input is a pipeline
        stages.append(obj)
        stages.extend(interleaved)
        count_transformers += n_interleaved
        count_transformers += obj.get_number_transformers()
        return stages, count_transformers

    elif (
        is_storage_request(obj).value > 0
    ):  # Check whether the input is a storage request
        # It is related to the nebula storage, do nothing, just append.
        stages.append(obj)
        return stages, count_transformers

    # If it is a list, iterate over it
    elif isinstance(obj, (tuple, list)):
        for el in obj:
            stages, n_trf = _create_stages(
                el, interleaved, prepend_interleaved, stages, 0
            )
            count_transformers += n_trf
        return stages, count_transformers
    else:
        msg = _make_error_msg(obj)
        raise TypeError(msg)


def _run_pipeline(
    obj: Union[PipeType, Dict[str, str]],
    df: "GenericDataFrame",
    backend: Optional[str],
    forced_trf: Optional[Transformer],
) -> "GenericDataFrame":
    """Run the pipeline(s) on the input DataFrame.

    Allowed obj types:
    - list / tuple
    - TransformerPipeline

    Args:
        obj (PipeType | dict(str, str)):
            The pipeline object to be executed. It can be either a list of
            transformers, a TransformerPipeline.
        df (DataFrame):
            The input DataFrame to be processed by the pipeline.
        backend (str | None):
            The input DataFrame to be processed by the pipeline.
        forced_trf (Transformer | None):
            Forced interleaved transformer.
    """
    _storage_request: Enum = is_storage_request(obj)

    if isinstance(obj, (list, tuple)):
        # Get formatted names for logging
        # names: str = __names_in_iterable(obj)
        if is_plain_transformer_list(obj):  # plain list of transformers
            # logger.debug(f"Input is a plain list/tuple of Transformers: {names}")
            df = _transform(obj, df, backend, forced_trf)

        else:
            # Handle mixed list/tuple of Transformers/Pipelines
            # logger.debug(f"Input is a mixed list of Transformers/Pipelines: {names}")
            # logger.debug("Coalescing Transformers in lists and executes separately...")

            # Separate transformers and pipelines into sub-lists.
            # The aim is to create a plain list of transformers and
            # separate them from other objects.
            # Once it is done, recall this function recursively
            # to process the sub-lists in the proper way.
            # I.e., assuming t* are transformers:
            # input -> [t1, t2, storage, t3, pipe, t4, t5]
            # output -> [[t1, t2], storage, [t3], pipe, [t4, t5]]
            outer_list = []
            inner_list = []
            for el in obj:
                if is_transformer(el):
                    inner_list.append(el)
                else:
                    # If there are transformers in the inner list, append
                    # it to the outer list
                    if inner_list:
                        outer_list.append(inner_list)
                    outer_list.append(el)
                    inner_list = []
            # If there are remaining transformers in the inner list, add it
            # to the outer list
            if inner_list:
                outer_list.append(inner_list)

            # logger.debug(f"Created {len(outer_list)} sub-lists.")

            # Recursively run the pipeline on each sub-list
            for el in outer_list:
                df = _run_pipeline(el, df, backend, forced_trf)

    elif _storage_request.value > 0:
        df = _handle_storage(_storage_request, obj, df)

    # From now on is a <TransformerPipeline>
    # the obj is a flat pipeline
    elif obj.get_pipe_type() == NodeType.LINEAR_PIPELINE:
        if obj.branch:
            sec_header, msg_storage = _get_fork_header("branch", obj.branch)
            logger.info(sec_header)
            if msg_storage:
                logger.info(msg_storage)
            logger.info(f"Running {get_pipeline_name(obj)}")
            n_part_orig: int = _get_n_partitions(df, obj)

            # logger.debug(f"Running the stages: {__names_in_iterable(obj.stages)}")
            stored_df_key = obj.branch.get("storage")
            if stored_df_key is None:
                df_input = df
            else:
                df_input = ns.get(stored_df_key)

            df_out = _run_pipeline(obj.stages, df_input, backend, forced_trf)

            if obj.otherwise:
                logger.info(
                    f"Running otherwise-pipeline {get_pipeline_name(obj.otherwise)}"
                )
                # df_input is the branch input that could be from the storage
                df = _run_pipeline(obj.otherwise.stages, df, backend, forced_trf)

            _FAIL_CACHE.clear()
            _FAIL_CACHE[("branch", "input")] = df_input
            _FAIL_CACHE[("branch", "output")] = df_out

            type_value: str = obj.branch["end"]
            try:
                if type_value == "append":
                    logger.info("Appending the dataframes ...")
                    df = append_df([df, df_out], obj.allow_missing_cols)
                elif type_value == "join":
                    logger.info("Joining the dataframes ...")
                    df = join_dfs(
                        df,
                        df_out,
                        on=obj.branch["on"],
                        how=obj.branch["how"],
                        broadcast=obj.branch.get("broadcast"),
                    )
                else:
                    pass  # It is a dead-end pipeline, the input df will return as is
                df = _repartition_coalesce(df, obj, n_part_orig)

                if forced_trf is not None:
                    df = __transform(df, forced_trf, backend)

            except Exception as e:  # pragma: no cover
                if not pipeline_config["activate_failure_cache"]:
                    raise e
                _, msg, _ = _cache_to_nebula_storage()
                raise_pipeline_error(e, msg)

            _FAIL_CACHE.clear()

        elif obj.apply_to_rows:
            sec_header, _ = _get_fork_header("apply to rows", obj.apply_to_rows)
            logger.info(sec_header)
            logger.info(f"Running {get_pipeline_name(obj)}")

            n_part_orig: int = _get_n_partitions(df, obj)

            df_apply, df_otherwise = split_df(df, obj.apply_to_rows)

            skip_if_empty: bool = obj.apply_to_rows["skip_if_empty"]
            is_empty = False
            if skip_if_empty:
                logger.info("Checking whether the branched data frame is empty ...")
                if df_is_empty(df_apply):
                    logger.info(
                        "The branched dataframe is empty, skip 'apply_to_rows' sub-pipeline"
                    )
                    is_empty = True
                else:
                    logger.info("The branched dataframe is not empty")
            if not is_empty:
                df_out = _run_pipeline(obj.stages, df_apply, backend, forced_trf)

            if obj.otherwise:
                logger.info(
                    f"Running otherwise-pipeline {get_pipeline_name(obj.otherwise)}"
                )
                df_otherwise = _run_pipeline(
                    obj.otherwise.stages, df_otherwise, backend, forced_trf
                )

            if obj.apply_to_rows.get("dead-end") or is_empty:
                df = df_otherwise  # Just return the untouched subset
            else:
                logger.info("Appending the dataframes ...")
                _FAIL_CACHE.clear()
                _FAIL_CACHE[("apply_to_rows", "input")] = df_otherwise
                _FAIL_CACHE[("apply_to_rows", "output")] = df_out
                try:
                    df = append_df([df_otherwise, df_out], obj.allow_missing_cols)

                    if forced_trf is not None:
                        df = __transform(df, forced_trf, backend)

                except Exception as e:  # pragma: no cover
                    if not pipeline_config["activate_failure_cache"]:
                        raise e
                    _, msg, _ = _cache_to_nebula_storage()
                    raise_pipeline_error(e, msg)
                _FAIL_CACHE.clear()
                df = _repartition_coalesce(df, obj, n_part_orig)

        else:
            logger.info(f"Running {get_pipeline_name(obj)}")
            # logger.debug(f"Running the stages: {__names_in_iterable(obj.stages)}")
            df = _run_pipeline(obj.stages, df, backend, forced_trf)

    # obj is a split pipeline
    elif obj.get_pipe_type() == NodeType.SPLIT_PIPELINE:
        pipe_name = get_pipeline_name(obj)
        logger.info(f"Running {pipe_name}")

        if get_dataframe_type(df) == "spark":
            input_schema = df.schema
        elif get_dataframe_type(df) == "pandas":
            input_schema = df.dtypes.to_dict()
        elif get_dataframe_type(df) == "polars":
            input_schema = df.schema
        else:  # pragma: no cover
            raise ValueError("Unsupported dataframe type")

        dict_df_split_input: Dict[str, "GenericDataFrame"]
        dict_df_split_input = obj.split_function(df)

        keys_split_function = set(dict_df_split_input.keys())
        keys_sub_pipelines = set(obj.splits.keys())

        if keys_split_function != keys_sub_pipelines:
            diff = keys_split_function.symmetric_difference(keys_sub_pipelines)
            msg = "Different keys in <split_function> and <dict_transformers> "
            msg += f"--> {sorted(keys_split_function)} "
            msg += f"<dict_transformers> --> {sorted(keys_sub_pipelines)}. "
            msg += f"Difference --> {sorted(diff)}"
            raise KeyError(msg)

        n_part_orig: int = _get_n_partitions(df, obj)

        li_df_split: List["GenericDataFrame"] = []  # splits to merge
        split_to_merge_names: List[str] = []
        dead_end_splits: Set[str] = obj.splits_no_merge

        _df_split_output: "GenericDataFrame"

        for split_name, el in obj.splits.items():
            logger.info(f"Running SPLIT <<< {split_name} >>>")

            if split_name not in dead_end_splits:
                split_to_merge_names.append(split_name)

            # logger.debug(f"Running the split sublist: {__names_in_iterable(el)}")

            _df_split_input = dict_df_split_input[split_name]
            if split_name in obj.splits_skip_if_empty:
                if df_is_empty(_df_split_input):
                    logger.info(f"Split '{split_name}' is empty, skip its pipeline")
                    continue

            _df_split_output = _run_pipeline(el, _df_split_input, backend, forced_trf)
            if split_name not in dead_end_splits:
                li_df_split.append(_df_split_output)

        split_names_joined = ", ".join(split_to_merge_names)
        logger.info(f"Merging splits: {split_names_joined} ...")
        # The 'split_apply_before_appending' transformers are already
        # included in the stages.

        # Store into cache before re-merging
        _FAIL_CACHE.clear()
        for split_name, _df in zip(split_to_merge_names, li_df_split):
            _FAIL_CACHE[("split", split_name)] = _df

        # Don't run it if li_df_split is empty.
        # It is useless (nothing to merge) and throws an error in 'reduce'.
        if li_df_split:
            try:
                if obj.cast_subset_to_input_schema:
                    li_df_split = to_schema(li_df_split, input_schema)
                df = append_df(li_df_split, obj.allow_missing_cols)
            except Exception as e:
                if not pipeline_config["activate_failure_cache"]:
                    raise e
                _, msg, _ = _cache_to_nebula_storage()
                raise_pipeline_error(e, msg)

            df = _repartition_coalesce(df, obj, n_part_orig)

        _FAIL_CACHE.clear()

    else:  # pragma: no cover
        msg = _make_error_msg(obj)
        raise TypeError(msg)

    return df


def _show_pipeline(
    obj: Union[PipeType, Dict[str, str]],
    level: int,
    ret: List[Tuple[int, str]],
    add_trf_params: bool,
) -> None:
    """Recursively traverse and display information about the pipeline.

    Args:
        obj (PipeType):
           The transformer pipeline object.
        level (int):
           The current level of recursion in the pipeline.
        ret (list(tuple(int, str))):
           A list to store the formatted information about the pipeline.
        add_trf_params (bool):
        If True, include transformer initialization parameters in the display.

    Returns (None):
        Inplace modification of `ret` input argument.

    Raises:
        TypeError: If the type of the input object is not understood.
    """
    _storage_request: Enum = is_storage_request(obj)

    # obj is a transformer
    if is_transformer(obj):
        # raise
        # if is_lazy_transformer(obj):
        #     msg = f"(Lazy) {obj.cls.__class__.__name__}"
        #     raise
        tf_name: str = get_transformer_name(
            obj,
            add_params=add_trf_params,
            max_len=pipeline_config["max_len_string_param"],
        )
        ret.append((level, " - " + tf_name))
        if hasattr(obj, "get_description"):
            trf_desc = obj.get_description()
            if trf_desc:
                ret.append((level, f"     Description: {trf_desc}"))

    # obj is a storage request
    elif _storage_request.value > 0:
        if _storage_request == StoreRequest.STORE_DF:
            _, msg = get_store_key_msg(obj)
        elif _storage_request == StoreRequest.STORE_DF_DEBUG:
            _, msg = get_store_debug_key_msg(obj)
        elif _storage_request == StoreRequest.ACTIVATE_DEBUG:
            msg = MSG_ACTIVATE_DEBUG_MODE
        elif _storage_request == StoreRequest.DEACTIVATE_DEBUG:
            msg = MSG_DEACTIVATE_DEBUG_MODE
        elif _storage_request == StoreRequest.REPLACE_WITH_STORED_DF:
            _, msg = get_replace_with_stored_df_msg(obj)
        else:  # pragma: no cover
            raise ValueError("Unknown Enum in _StoreRequest")
        msg_store = f"   --> {msg}"
        ret.append((level, msg_store))

    # obj is an iterable
    elif isinstance(obj, (list, tuple)):
        for el in obj:
            _show_pipeline(el, level, ret, add_trf_params)

    # From now on is a <TransformerPipeline>
    # the obj is a flat pipeline
    elif obj.get_pipe_type() == NodeType.LINEAR_PIPELINE:
        add_level = 0
        if obj.branch:
            sec_header, msg_storage = _get_fork_header("branch", obj.branch)
            ret.append((level, sec_header))
            add_level += 1
            if msg_storage:
                ret.append((level + add_level, msg_storage))
        elif obj.apply_to_rows:
            sec_header, _ = _get_fork_header("apply to rows", obj.apply_to_rows)
            ret.append((level, sec_header))
            add_level += 1

        ret.append((level + add_level, get_pipeline_name(obj)))
        for el in obj.stages:
            # I don't add indentation for linear pipelines
            _show_pipeline(el, level + add_level, ret, add_trf_params)

        if obj.otherwise:
            ret.append((level + add_level, "+-+-+-+-+ OTHERWISE +-+-+-+-+"))
            ret.append((level + add_level, get_pipeline_name(obj.otherwise)))
            for el in obj.otherwise.stages:
                _show_pipeline(el, level + add_level, ret, add_trf_params)

        if obj.branch:
            if obj.branch["end"] == "append":
                ret.append((level, "<<< Append DFs >>>"))
            elif obj.branch["end"] == "join":
                ret.append((level, "<<< Join DFs >>>"))
        elif obj.apply_to_rows:
            ret.append((level, "<<< Append Rows >>>"))

    # obj is a split pipeline
    elif obj.get_pipe_type() == NodeType.SPLIT_PIPELINE:
        ret.append((level, get_pipeline_name(obj)))
        dead_end_splits: Set[str] = obj.splits_no_merge
        split_names = []

        for split_name, el in obj.splits.items():
            ret.append((level, f"SPLIT <<< {split_name} >>>:"))
            split_names.append(split_name)
            _show_pipeline(el, level + 1, ret, add_trf_params)

        # Splits to merge
        split_title = ""
        if obj.cast_subset_to_input_schema:
            split_title = "CAST EACH SPLIT TO MERGE TO THE INPUT SCHEMA AND "
        ret.append((level, split_title + "MERGE SPLITS:"))
        flag_no_split_to_merge = True
        for split_name in split_names:
            if split_name in dead_end_splits:
                continue
            flag_no_split_to_merge = False
            ret.append((level, f"   - <<< {split_name} >>>"))

        if flag_no_split_to_merge:
            ret.append((level, "   No splits to merge"))

        # Dead-end splits
        if dead_end_splits:
            ret.append((level, "DEAD-END SPLITS (NOT MERGED):"))
            for dead_end_split in sorted(dead_end_splits):
                ret.append((level, f"   - <<< {dead_end_split} >>>"))

    else:  # pragma: no cover
        raise TypeError(f"Not understood: {type(obj)}")


class TransformerPipeline:
    def __init__(
        self,
        data: Union[PipeType, Dict[str, PipeType]],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        split_function: Optional[Callable] = None,
        split_order: Optional[List[str]] = None,
        interleaved: TransformerOrTransformerList = None,
        prepend_interleaved: bool = False,
        append_interleaved: bool = False,
        split_apply_after_splitting: TransformerOrTransformerList = None,
        split_apply_before_appending: TransformerOrTransformerList = None,
        splits_no_merge: Union[None, str, Iterable[str]] = None,
        splits_skip_if_empty: Union[None, str, Iterable[str]] = None,
        cast_subset_to_input_schema: bool = False,
        repartition_output_to_original: bool = False,
        coalesce_output_to_original: bool = False,
        allow_missing_columns: bool = False,
        branch: Optional[Dict[str, str]] = None,
        apply_to_rows: Optional[Dict[str, Any]] = None,
        otherwise: Optional[Union[PipeType, Dict[str, PipeType]]] = None,
        df_input_name: Optional[str] = None,
        df_output_name: Optional[str] = None,
        backend: Optional[str] = None,
        skip: Optional[bool] = None,
        perform: Optional[bool] = None,
    ):
        """Create a transformer pipeline.

        A pipeline can consist of transformers or, recursively, other
        <TransformerPipeline>.

        Public functionalities:
        - Display succinctly the pipeline flow as a log
        - Render graphically the pipeline with 'graphviz' & 'pyyaml'
        - Run the pipeline

        The type <PipeType> can be one of the following:
        - Transformer
        - list(Transformer)
        - TransformerPipeline
        - list(TransformerPipeline)

        Args:
            data (PipeType | dict(str, PipeType)):
                Can be one of the following:
                - Transformer:
                - List of Transformer
                - TransformerPipeline
                - List of TransformerPipeline
                - A dictionary <str> -> all the previous types.
                    If it is a dictionary with len > 1, the split function must
                    be provided to create sub-pipelines.
                All the transformers must be initialized.
            name (str | None):
                Name of the pipeline that will appear in the log / dag.
                Defaults to None.
            description (str | None):
                Description of the pipeline that will appear in the log / dag.
                Defaults to None.
            split_function (callable | None):
                Function to create a split pipeline (used if 'data' is a
                dictionary). It must return a dictionary with the same keys as
                'data' where each value is a subset of the original dataframe.
                Ignored if 'data' is a dictionary of length == 1.
                Defaults to None.
            split_order (list(str) | None):
                When 'split_function' is provided and 'data' contains a
                dictionary of split pipelines, the 'split_order' allows the
                user to choose the split execution order.
                If not provided, they run in alphabetical order.
                If provided, 'split_order' must contain exactly the same keys
                listed in 'data', otherwise it throws a 'KeyError'.
                Defaults to None.
            interleaved (Transformer | list(Transformer) | None)
                If specified, you can provide a Transformer or a list of
                pre-initialized Transformers. These transformers will be
                inserted between each of the primary transformers.
                It's important to note that this feature is intended for
                development and debugging purposes only, as it can
                potentially introduce ambiguities in complex pipelines.
                Some common transformers used for debugging include
                'Count()' and 'LogDataSkew()'.
                Defaults to None.
            prepend_interleaved (bool):
                If True, prepend the 'interleaved' transformers at the
                beginning of the pipeline. Ignored if no 'interleaved'
                transformers are provided. Defaults to False.
            append_interleaved (bool):
                If True, append the 'interleaved' transformers at the end
                of the pipeline. Ignored if no 'interleaved' transformers
                are provided. Defaults to False.
            split_apply_after_splitting (Transformer | list(Transformer) | None):
                A pipeline to be applied after the split function, before
                executing the single split pipeline.
                Ignored when it is a linear pipeline.
                Defaults to None.
            split_apply_before_appending (Transformer | list(Transformer) | None)
                A pipeline to be applied after each split, before re-merging
                them back. Ignored when it is a linear pipeline.
                Defaults to None.
            splits_no_merge (str | list(str) | None):
                Dead-end splits that will not be merged.
                Defaults to None.
            splits_skip_if_empty (str | Iterable(str) | None):
                Specify whether to skip a split sub-pipeline if the input
                subset DataFrame for the indicated splits is empty. This
                requires an eager operation due to the use of the 'isEmpty'
                method. Defaults to None.
            cast_subset_to_input_schema (bool):
                Cast each split dataframe to the input schema before the
                splitting occurs.
                This parameter is used only for split pipelines or when the
                'apply_to_rows' dictionary is provided.
                Defaults to False.
            repartition_output_to_original (bool):
                When the pipeline generates sub-pipelines during splitting,
                the resulting number of partitions will be equal to the
                initial number of partitions multiplied by the number of
                splits, potentially leading to data skew. If this parameter
                is set to True, the final dataframe will be repartitioned
                to match the initial number of partitions.
                This parameter is used only for split pipelines or when the
                'apply_to_rows' dictionary is provided.
                Defaults to False.
            coalesce_output_to_original (bool):
                Similar to 'repartition_output_to_original,' this function
                performs a 'coalesce' operation instead of a repartition.
                While it is faster, it may be less effective because it does
                not guarantee the elimination of empty or skewed partitions.
                This parameter is used only for split pipelines or when the
                'apply_to_rows' dictionary is provided.
                Defaults to False.
            allow_missing_columns (bool):
                The set of column names in the dataframes to append can differ;
                missing columns will be filled with null and cast to the
                proper types.
                This parameter is used only for split pipelines or when the
                'apply_to_rows' / 'branch' dictionary is provided.
                Defaults to False.
            branch (dict(str, str) | None):
                Used exclusively for flat pipelines.
                If provided, it initiates a secondary pipeline, originating
                either from the primary dataframe, if "storage" is not
                specified, or from a dataframe fetched from the Nebula storage
                using the specified "storage" key.
                At the end of the pipeline, there are three possible
                scenarios based on the 'end' value:
                - branch={"storage": ..., "end": "dead-end"}:
                    the dataframe will not be merged back.
                - branch={"storage": ..., "end": "append"}:
                    the dataframe will be appended to the main one.
                    To allow the union if the set of column names differ, the
                    user can set the "allow_missing_column" parameter to True.
                - branch={
                        "storage": ...,
                        "end": "join",
                        "on": ...,
                        "how": ...,
                        "skip" Optional(bool): If provided, it cannot be
                            contradictory with 'skip'.
                        "perform" Optional(bool): If provided, it cannot be
                            contradictory with 'perform'.
                        "broadcast" Optional(bool): for end="join" only
                    }:
                    the dataframe will be joined to the primary one using the
                    provided 'on' and 'how' parameters.
                    If the boolean parameter "skip" is set to True, the
                    branch will be skipped.
                    The "broadcast" parameter is intended for spark optional.
                    With other backends it is simply ignored. In Spark, if set to
                    True, the right dataframe (the branched one) will be
                    broadcast before joining.
                    If the branch is skipped (whether because skip=True, or
                    perform=False), only the "otherwise" pipeline (if
                    provided) will be executed.
                Defaults to None.
            apply_to_rows (dict(str, any) | None):
                Used exclusively for flat pipelines.
                If provided, the input dataframe is split in two subsets
                according to the provided condition. The provided
                transformers are then applied only to the row that matches
                the condition. The other subset remains untouched.
                At the end of the pipeline the subsets are appended by
                field name.
                This <dict> parameter takes the following keys:
                - "input_col" (necessary - str):
                    Specifies the input column to be utilized to match
                    the condition.
                - operator (necessary - str):
                    - "eq":             equal
                    - "le":             less equal
                    - "lt":             less than
                    - "ge":             greater equal
                    - "gt":             greater than
                    - "isin":           iterable of valid values
                    - "array_contains": has at least one instance of <value> in an array column
                    - "contains":       has at least one instance of <value> in a string column
                    - "startswith":     The row value starts with <value> in a string column
                    - "endswith":       The row value ends with <value> in a string column
                    - "between":        is between 2 values, lower and upper bound inclusive
                    - "like":           matches a SQL LIKE pattern
                    - "rlike":          matches a regex pattern
                    - "isNull"          *
                    - "isNotNull"       *
                    - "isNaN"           *
                    - "isNotNaN"        *
                    * Does not require the optional "value" argument
                    "ne" (not equal) is not allowed
                - value (optional - any):
                    Value used for the comparison.
                - comparison_column (optional - str):
                    Name of column to be compared with `input_col`.
                    It must be different from 'input_col'.
                    If 'value' is provided, this parameter must not be set.
                - dead-end (bool | None):
                    If True, the rows that matched the condition are not
                    merged back. Defaults to None.
                - skip_if_empty (bool):
                    If True and the subset of rows that match the condition is
                    empty, skip this branched pipeline and its final append.
                    Default to False.
            otherwise (PipeType | dict(str, PipeType) | None):
                A pipeline operates on the dataframe or a subset of rows that
                are unaffected by sub-pipelines originating from either
                ‘branch’ or ‘apply_to_rows’. This functionality applies under
                specific conditions:
                - For ‘branch’, it must originate from the primary dataframe
                    (without the ‘storage’ key) and have an ‘end’
                    different from ‘dead-end’.
                - For ‘apply_to_rows’, the ‘dead-end’ key should not be
                    provided or set to ‘None’ / ‘False’.
                In all other cases or when ‘branch’ / ‘apply_to_rows’ are not
                specified, it throws an error. Default to None.
            df_input_name (str):
                Name of the dataframe displayed in the visualization.
            df_output_name (str):
                Name of the dataframe displayed in the visualization.
            backend (str | None):
                If provided, force the pipeline to use either "pandas" or "spark"
                backend, otherwise, the dataframe type is inspected
                automatically. Currently allowed backends are:
                "spark" | "pandas" | None.
            skip (bool | None):
                If True, skip the pipeline and return an empty one.
                If provided, it must not contradict 'perform'.
                Defaults to None.
            perform (bool):
                If False, skip the pipeline and return an empty one.
                If provided, it must not contradict 'skip'.
                Defaults to None.
        """
        if backend is not None:
            assert_allowed(backend, {"pandas", "polars", "spark"}, "backend")

        skip = validate_skip_perform(skip, perform)
        ensure_no_branch_or_apply_to_rows_otherwise(branch, apply_to_rows, otherwise)

        self.branch: Optional[Dict[str, str]] = None
        self.apply_to_rows: Optional[Dict[str, Any]] = None
        self.backend: Optional[str] = backend

        self.description = description if description else ""
        self._pipe_type: NodeType

        self.name: Optional[str] = name
        self.df_input_name: str = df_input_name if df_input_name else "DF input"
        self.df_output_name: str = df_output_name if df_output_name else "DF output"

        self._interleaved: List[Transformer] = []

        self._n_transformers: int = 0

        self.otherwise: Optional[TransformerPipeline] = None

        self.repartition_output_to_original: bool = False
        self.coalesce_output_to_original: bool = False
        self.allow_missing_cols: bool = False
        self.cast_subset_to_input_schema: bool = False

        self.split_function: Optional[Callable] = None

        self.split_after_splitting: List[Transformer] = []
        self.split_before_appending: List[Transformer] = []

        self.splits_no_merge: Set[str] = set()
        self.splits_skip_if_empty: Set[str] = set()

        self.stages: List[Transformer] = []
        self.splits = OrderedDict()

        if skip:
            # Force it to a linear pipeline to allow the user not to
            # define the split_function for this case.
            self._pipe_type = NodeType.LINEAR_PIPELINE
        else:
            data = self._define_pipe_type(data, split_function)

            if self._pipe_type == NodeType.SPLIT_PIPELINE:
                # They cannot be both True
                assert_at_most_one_args(
                    cast_subset_to_input_schema, allow_missing_columns
                )
                ensure_no_branch_or_apply_to_rows_in_split_pipeline(
                    branch, apply_to_rows
                )

            if branch:
                assert_branch_inputs(branch)
                if branch.get("skip") or (not branch.get("perform", True)):
                    if otherwise:
                        data = otherwise
                    else:
                        data = []
                    branch = None
                    otherwise = None
                    cast_subset_to_input_schema = False
                    repartition_output_to_original = False
                    coalesce_output_to_original = False
                    allow_missing_columns = False
                else:
                    self.branch: Dict[str, Any] = deepcopy(branch)
                    self.branch.update(
                        {"allow_missing_columns": bool(allow_missing_columns)}
                    )

            if apply_to_rows:
                assert_apply_to_rows_inputs(apply_to_rows)
                self.apply_to_rows = deepcopy(apply_to_rows)
                self.apply_to_rows["skip_if_empty"] = self.apply_to_rows.get(
                    "skip_if_empty", False
                )

            self._interleaved = sanitize_list_transformers(interleaved)
            self.repartition_output_to_original = bool(repartition_output_to_original)
            self.coalesce_output_to_original = bool(coalesce_output_to_original)
            self.allow_missing_cols = bool(allow_missing_columns)

            if otherwise:
                if isinstance(otherwise, TransformerPipeline):
                    self.otherwise = otherwise
                else:
                    self.otherwise = TransformerPipeline(otherwise)
                self._n_transformers += self.otherwise.get_number_transformers()

            self.split_function = split_function

            # Set to False some arguments if not needed
            if apply_to_rows or branch:
                is_dead_end: bool
                if apply_to_rows:
                    is_dead_end = bool(apply_to_rows.get("dead-end"))
                else:
                    is_dead_end = bool(branch.get("dead-end"))
                if is_dead_end:
                    self.repartition_output_to_original = False
                    self.coalesce_output_to_original = False
                    self.allow_missing_cols = False

            self.cast_subset_to_input_schema = bool(cast_subset_to_input_schema)

            self.splits_no_merge = self.__set_aux_splits(splits_no_merge)
            self.splits_skip_if_empty = self.__set_aux_splits(splits_skip_if_empty)

            # Checks and first initializations are completed.
            # From now on, create the stages

            # Create stages for a linear pipeline
            if self._pipe_type == NodeType.LINEAR_PIPELINE:
                self._make_stages_in_linear_pipe(
                    data, prepend_interleaved, append_interleaved
                )

            # Create stages for a split pipeline
            elif self._pipe_type == NodeType.SPLIT_PIPELINE:
                assert_at_most_one_args(
                    repartition_output_to_original, coalesce_output_to_original
                )
                self._make_stages_in_split_pipe(
                    data,
                    split_order,
                    split_apply_after_splitting,
                    split_apply_before_appending,
                    prepend_interleaved,
                    append_interleaved,
                )

            # Neither a linear pipeline nor a split one.
            else:  # pragma: no cover
                raise TypeError("Pipeline type not understood.")

        self._dag = create_dag(self, self.df_input_name, self.df_output_name)

        # For HTML, experimental
        if self._pipe_type == NodeType.LINEAR_PIPELINE:
            self._sub_pipe_type = "linear"
        if branch:
            self._sub_pipe_type = "branch"
        elif apply_to_rows:
            self._sub_pipe_type = "a2r"
        if self._pipe_type == NodeType.SPLIT_PIPELINE:
            self._sub_pipe_type = "split"

    def _make_stages_in_linear_pipe(
        self, data, prepend_interleaved: bool, append_interleaved: bool
    ):
        stages, n_trf = _create_stages(data, self._interleaved, prepend_interleaved)
        self.stages = stages
        self._n_transformers = n_trf

        n_interleaved = len(self._interleaved)
        if n_interleaved and (not append_interleaved):
            self._n_transformers -= n_interleaved
            _remove_last_transformers(stages, n_interleaved)

    @staticmethod
    def __set_aux_splits(aux_split) -> Set[str]:
        if not aux_split:
            return set()

        if isinstance(aux_split, set):  # Just to ensure a flat iterable
            aux_split = list(aux_split)
        return set(ensure_flat_list(aux_split))

    @staticmethod
    def __assert_aux_splits(main_splits, splits_to_check: Set[str], name: str):
        diff: Set[str] = splits_to_check.difference(main_splits)
        if diff:
            diff_str = ", ".join(sorted(diff))
            raise KeyError(f'"{name}" has unmatched splits: {diff_str}')

    def _make_stages_in_split_pipe(
        self,
        data,
        split_order,
        split_apply_after_splitting,
        split_apply_before_appending,
        prepend_interleaved,
        append_interleaved,
    ):
        if split_order:
            if not all(isinstance(i, str) for i in split_order):
                raise TypeError(f'"split_order" must be <list<str>>: {split_order}')
            diff = set(split_order).symmetric_difference(data)
            if diff:
                msg = "'split_order' and 'data', must contain the same "
                msg += f"keys are not in common: {diff}"
                raise KeyError(msg)
        else:
            split_order = sorted(data.keys())

        self.split_after_splitting = self._set_splits(split_apply_after_splitting)
        self.split_before_appending = self._set_splits(split_apply_before_appending)

        n_interleaved = len(self._interleaved)

        self.splits = OrderedDict()

        for split_name in split_order:
            obj = data[split_name]
            stages, n_trf = _create_stages(obj, self._interleaved, prepend_interleaved)
            if n_interleaved and (not append_interleaved):
                _remove_last_transformers(stages, n_interleaved)
                n_trf -= n_interleaved

            # apply after splitting
            stages = self.split_after_splitting + stages
            n_trf += len(self.split_after_splitting)

            # apply before appending
            stages.extend(self.split_before_appending)
            n_trf += len(self.split_before_appending)

            self.splits[split_name] = stages
            self._n_transformers += n_trf

        self.__assert_aux_splits(self.splits, self.splits_no_merge, "splits_no_merge")
        self.__assert_aux_splits(
            self.splits, self.splits_skip_if_empty, "splits_skip_if_empty"
        )

    def _define_pipe_type(self, data, split_function):
        """Define the pipe type.

        If 'data' is a dictionary of length 1, it will be returned as a flat
        list and the pipeline will be considered a linear one.
        """
        # If the main input is:
        # - single transformer
        # - list of transformers
        # - TransformerPipeline
        # then the final object will be a linear pipeline.
        if is_transformer(data) or isinstance(data, (list, TransformerPipeline)):
            # It is a linear pipeline
            self._pipe_type = NodeType.LINEAR_PIPELINE

        elif isinstance(data, dict):
            if is_storage_request(data).value > 0:
                self._pipe_type = NodeType.LINEAR_PIPELINE
                return data

            self._pipe_type = NodeType.SPLIT_PIPELINE
            _n_data = len(data)
            if _n_data == 1:
                self._pipe_type = NodeType.LINEAR_PIPELINE
                # Extract the single-split from the dictionary
                data = list(data.values())[0]
                if split_function is not None:
                    msg = "An expected split-pipeline has been passed with "
                    msg += "a dictionary of length 1, "
                    msg += "the split function will be ignored."
                    logger.warning(msg)

            elif (len(data) > 1) and (not callable(split_function)):
                if split_function is None:
                    raise AssertionError("'split_function' not declared (got 'None')")
                t = type(split_function)
                raise TypeError(f"'split_function' must be <callable>. Found {t}")

        else:
            raise TypeError("'data' type not understood")

        return data

    @staticmethod
    def _set_splits(o) -> List[Transformer]:
        """Used to set the 'before_appending' and 'after_splitting' transformers."""
        if o:
            if isinstance(o, tuple):
                return list(o)
            elif isinstance(o, list):
                return o
            elif is_transformer(o):
                return [o]
            else:
                msg = '"split_apply_before_appending" and "split_apply_after_splitting"'
                msg += "  must be a <Transformer> or a <list<Transformer>>"
                raise ValueError(msg)
        else:
            return []

    def get_pipe_type(self) -> NodeType:
        """Get the pipe type as <str>: "linear" | "split"."""
        return self._pipe_type

    def get_number_transformers(self) -> int:
        """Return the overall number of transformers."""
        return self._n_transformers

    def show_pipeline(
        self, split_indentation: int = 4, add_transformer_params: bool = False
    ) -> None:
        """Iterate through the pipeline recursively and print the transformers.

        Args:
            split_indentation (int):
                Indentations for splits.
            add_transformer_params:
                If True, add the transformer initialization parameters.
                Default False.
        """
        ret = []
        _show_pipeline(self, 0, ret, add_transformer_params)

        prepend = " " * split_indentation
        for lev, name in ret:
            print(prepend * lev + name)

    def run(
        self, df_input, force_interleaved_transformer: Optional[Transformer] = None
    ):
        """Run the transformer pipeline.

        Args:
            df_input (pandas.DataFrame | pyspark.sql.DataFrame):
                Input dataframe.
            force_interleaved_transformer (Transformer | None):
                If provided, execute this transformer after each transformation step.

        Returns (pandas.DataFrame | pyspark.sql.DataFrame):
            Transformed dataframe.
        """
        start_time = time()
        df_ret = _run_pipeline(
            self, df_input, self.backend, force_interleaved_transformer
        )
        end_time = time()
        run_time = f" in {int(end_time - start_time)} seconds"
        logger.info(f"Pipeline run completed{run_time}.")
        return df_ret

    def plot_dag(
        self, add_transformer_params=False, add_transformer_description=False
    ):  # pragma: no cover
        """Plot the dag using GraphViz & pyyaml."""
        from ._graphviz import create_graph

        return create_graph(
            self._dag.dag, add_transformer_params, add_transformer_description
        )

    def _print_dag(self):
        return print_dag(self._dag.dag)


if __name__ == "__main__":  # pragma: no cover
    from nlsn.nebula.spark_transformers import (
        Count,
        EmptyArrayToNull,
        LogDataSkew,
        NanToNull,
    )

    def _f_split(x):
        return x

    p1a = TransformerPipeline({"s1": [], "s2": []}, split_function=_f_split, name="p1a")

    p1 = TransformerPipeline([p1a], name="FULL PIPELINE")

    # p1.show_pipeline(add_transformer_params=True)
    p1._print_dag()  # noqa

    _li_trf = [
        EmptyArrayToNull(columns="c3"),
        Count(),
        {"store": "df_x_processed"},
        {"store_debug": "df_x_processed_debug"},
        NanToNull(columns="*"),
    ]

    pipe_example = TransformerPipeline(
        _li_trf,
        interleaved=[LogDataSkew()],
        prepend_interleaved=True,
        append_interleaved=True,
        name="cleaning",
    )
