"""TransformerPipeline - the main user-facing pipeline class.

This module provides the public API for creating and running pipelines.
It wraps the IR building, execution, and visualization components.

The TransformerPipeline class maintains backward compatibility with
the existing API while using the new IR-based architecture internally.

Example:
    from nebula import TransformerPipeline

    pipeline = TransformerPipeline(
        [
            SelectColumns(columns=['a', 'b']),
            Filter(input_col='a', operator='gt', value=0),
            my_custom_function,
        ],
        name="my_pipeline",
    )

    result = pipeline.run(df)
    pipeline.show(add_params=True)
    pipeline.plot().render('pipeline.png')
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Union

from nebula.auxiliaries import assert_at_most_one_args
from nebula.base import Transformer, LazyWrapper
from nebula.pipelines._checks import *
from nebula.pipelines.execution import (
    PipelineExecutor,
    PipelineHooks,
    NoOpHooks,
    LoggingHooks,
)
from nebula.pipelines.ir import IRBuilder, SequenceNode
from nebula.pipelines.pipe_aux import (
    is_keyword_request,
    is_split_pipeline,
    PIPELINE_KEYWORDS,
)
from nebula.pipelines.pipe_cfg import PIPE_CFG
from nebula.pipelines.visualization import PipelinePrinter

__all__ = ["TransformerPipeline", "PipelineHooks", "NoOpHooks", "LoggingHooks"]

# Type alias for pipeline data
PipeType = Union[
    Transformer,
    LazyWrapper,
    Callable,
    list,
    tuple,
    dict,
    "TransformerPipeline",
]


class TransformerPipeline:
    """A declarative data transformation pipeline.

    TransformerPipeline allows you to compose transformers, functions,
    and nested pipelines into a reusable, inspectable pipeline.

    Features:
    - Linear pipelines: Sequential transformers
    - Split pipelines: Parallel branches that merge back
    - Branch pipelines: Fork from main or stored DataFrame
    - Apply-to-rows: Filter, transform subset, merge back
    - Storage operations: Store/load intermediate results
    - Bare functions: Use regular functions as transformers
    - Interleaved debugging: Insert debug transformers between steps
    - Checkpointing: Resume from failed step (future)

    Example:
        # Simple linear pipeline
        pipe = TransformerPipeline([
            SelectColumns(columns=['a', 'b']),
            Filter(input_col='a', operator='gt', value=0),
        ])
        result = pipe.run(df)

        # With functions and storage
        pipe = TransformerPipeline([
            SelectColumns(columns=['a', 'b']),
            {'store': 'intermediate'},
            my_transform_function,
            {'store': 'final'},
        ])

        # Split pipeline
        def split_by_type(df):
            return {
                'high': df.filter(col('value') > 100),
                'low': df.filter(col('value') <= 100),
            }

        pipe = TransformerPipeline(
            {
                'high': [HighValueTransformer()],
                'low': [LowValueTransformer()],
            },
            split_function=split_by_type,
        )
    """

    def __init__(
        self,
        data: PipeType | dict[str, PipeType],
        *,
        name: str | None = None,
        df_input_name: str | None = None,
        df_output_name: str | None = None,
        # Interleaved (debugging)
        interleaved: list | None = None,
        prepend_interleaved: bool = False,
        append_interleaved: bool = False,
        # Split pipeline options
        split_function: Callable | None = None,
        split_order: list[str] | None = None,
        split_apply_after_splitting: list | None = None,
        split_apply_before_appending: list | None = None,
        splits_no_merge: str | Iterable[str] | None = None,
        splits_skip_if_empty: str | Iterable[str] | None = None,
        # Branch/Fork options
        branch: dict[str, Any] | None = None,
        apply_to_rows: dict[str, Any] | None = None,
        otherwise: PipeType | None = None,
        # Merge options
        allow_missing_columns: bool = False,
        cast_subsets_to_input_schema: bool = False,
        repartition_output_to_original: bool = False,
        coalesce_output_to_original: bool = False,
        # Skip options
        skip: bool | None = None,
        perform: bool | None = None,
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
            cast_subsets_to_input_schema (bool):
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
                        "right_on": ...,
                        "left_on": ...,
                        "how": ...,
                        "suffix": ...,  (for polars / narwhals)
                        "broadcast" (bool | None): for end="join" only with spark backend
                    }:
                    the dataframe will be joined to the primary one using the
                    provided 'on' and 'how' parameters.
                    The "broadcast" parameter is intended for spark optional,
                    with other backends it is simply ignored. In Spark, if set to
                    True, the right dataframe (the branched one) will be
                    broadcast before joining.
                    Other possible key / boolean values are "skip" and "perform".
                    branch = {
                        ...
                        "skip" (bool | None): If provided, it cannot be
                            contradictory with 'perform'.
                        "perform" (bool | None): If provided, it cannot be
                            contradictory with 'skip'.
                    }
                    If the boolean parameter "skip" is set to True, the
                    branch will be skipped.
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
                - value (any | None):
                    Value used for the comparison.
                - comparison_column (str | None):
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
                - skip / perform (bool | None): like in 'branch'.
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
            skip (bool | None):
                If True, skip the pipeline and return an empty one.
                If provided, it must not contradict 'perform'.
                Defaults to None.
            perform (bool):
                If False, skip the pipeline and return an empty one.
                If provided, it must not contradict 'skip'.
                Defaults to None.
        """
        validate_skip_perform(skip, perform)
        ensure_no_branch_or_apply_to_rows_otherwise(branch, apply_to_rows, otherwise)
        assert_at_most_one_args(
            cast_subset_to_input_schema=cast_subsets_to_input_schema,
            allow_missing_columns=allow_missing_columns,
        )

        if split_function is not None:
            if not callable(split_function):
                raise TypeError(
                    "If provided, the 'split_function' must be "
                    f"a callable, found {type(split_function)}"
                )

        if is_split_pipeline(data, split_function):
            if not callable(split_function):  # pragma: no cover
                raise TypeError(
                    "For split-pipelines, the 'split_function' "
                    f"must be callable, found {type(split_function)}"
                )

            ensure_no_branch_or_apply_to_rows_in_split_pipeline(branch, apply_to_rows)

            if split_order:
                assert_split_order(data, split_order)

            # Normalize splits_no_merge and splits_skip_if_empty to sets
            splits_no_merge = set_split_options(
                data, splits_no_merge, "splits_no_merge"
            )
            splits_skip_if_empty = set_split_options(
                data, splits_skip_if_empty, "splits_skip_if_empty"
            )

        else:
            if isinstance(data, dict) and not is_keyword_request(data):
                raise ValueError(
                    "Unknown input, For split-pipelines the "
                    "number of splits must be > 1, found "
                    f"{len(data)}, key-word operations the dictionary "
                    "must be single key-value pair with a valid key"
                    f"{PIPELINE_KEYWORDS}"
                )

        self.name = name

        # Handle skip/perform
        if skip or (perform is False):
            data = []
            branch = None
            apply_to_rows = None
            otherwise = None

        interleaved = to_list_of_transformations(interleaved, "interleaved")
        split_apply_after_splitting = to_list_of_transformations(
            split_apply_after_splitting, "split_apply_after_splitting"
        )
        split_apply_before_appending = to_list_of_transformations(
            split_apply_before_appending, "split_apply_before_appending"
        )

        otherwise_only: bool = False
        if apply_to_rows:
            assert_apply_to_rows_inputs(apply_to_rows)
            if apply_to_rows.get("skip") or (apply_to_rows.get("perform") is False):
                otherwise_only = True
        if branch:
            assert_branch_inputs(branch)
            if branch.get("skip") or (branch.get("perform") is False):
                otherwise_only = True

        if otherwise_only:  # branch / apply_to_rows is skipped
            if otherwise:
                data = otherwise
            else:
                data = []
            branch = None
            apply_to_rows = None
            otherwise = None

        # Build the IR
        self._ir: SequenceNode = IRBuilder(
            data,
            name=name,
            df_input_name=df_input_name or "DF input",
            df_output_name=df_output_name or "DF output",
            interleaved=interleaved,
            prepend_interleaved=prepend_interleaved,
            append_interleaved=append_interleaved,
            split_function=split_function,
            split_order=split_order,
            split_apply_after_splitting=split_apply_after_splitting,
            split_apply_before_appending=split_apply_before_appending,
            splits_no_merge=splits_no_merge,
            splits_skip_if_empty=splits_skip_if_empty,
            branch=branch,
            apply_to_rows=apply_to_rows,
            otherwise=otherwise,
            allow_missing_columns=allow_missing_columns,
            cast_subsets_to_input_schema=cast_subsets_to_input_schema,
            repartition_output_to_original=repartition_output_to_original,
            coalesce_output_to_original=coalesce_output_to_original,
        ).build()

        # Store config for inspection
        self._config = {
            "name": name,
            "branch": branch,
            "apply_to_rows": apply_to_rows,
            "allow_missing_columns": allow_missing_columns,
        }

    def run(
        self,
        df: Any,
        *,
        hooks: PipelineHooks | None = None,
        resume_from: str | None = None,
        show_params: bool = False,
        force_interleaved_transformer: Any = None,
    ) -> Any:
        """Execute the pipeline on input DataFrame.

        Args:
            df: Input DataFrame (pandas, Polars, Spark, or Narwhals).
            hooks: Optional hooks for monitoring/extensibility.
            resume_from: Node ID to resume from (skip prior nodes).
            show_params: Whether to show the parameter of transformers, function, etc.
            force_interleaved_transformer: Transformer to run after each step.

        Returns:
            Transformed DataFrame (same backend as input).

        Example:
            result = pipeline.run(df)

            # With custom hooks
            result = pipeline.run(df, hooks=LoggingHooks())

            # Resume from checkpoint
            result = pipeline.run(df, resume_from="Filter_abc123@3")
        """
        # Default to logging hooks for backward compatibility
        if hooks is None:
            hooks = LoggingHooks(
                max_param_length=PIPE_CFG["max_param_length"], show_params=show_params
            )

        executor = PipelineExecutor(
            ir=self._ir,
            hooks=hooks,
            resume_from=resume_from,
            force_interleaved=force_interleaved_transformer,
        )

        return executor.run(df)

    def show(
        self,
        *,
        add_params: bool = False,
        add_ids: bool = False,
        indent_size: int = 4,
    ) -> None:
        """Print pipeline structure to terminal.

        Args:
            add_params: Include transformer parameters.
            add_ids: Include node IDs (useful for resume_from).
            indent_size: Spaces per indentation level.

        Example:
            pipeline.show()
            pipeline.show(add_params=True, add_ids=True)
        """
        printer = PipelinePrinter(
            self._ir,
            max_param_length=PIPE_CFG["max_param_length"],
            indent_size=indent_size,
        )
        printer.print(add_params=add_params, add_ids=add_ids)

    def to_string(self, *, add_params: bool = False) -> str:
        """Get pipeline structure as string.

        Args:
            add_params: Include transformer parameters.

        Returns:
            Multi-line string representation.
        """
        printer = PipelinePrinter(
            self._ir, max_param_length=PIPE_CFG["max_param_length"]
        )
        return printer.to_string(add_params=add_params)

    def plot(
        self,
        *,
        add_params: bool = False,
        add_description: bool = False,
    ):  # pragma: no cover
        """Create Graphviz visualization.

        Args:
            add_params: Include transformer parameters.
            add_description: Include transformer descriptions.

        Returns:
            Graphviz Digraph object.

        Example:
            dot = pipeline.plot(add_params=True)
            dot.render('pipeline', format='png')
            dot  # Display in Jupyter
        """
        from .visualization import GraphvizRenderer, HAS_GRAPHVIZ, HAS_PYYAML

        if (not HAS_GRAPHVIZ) or (not HAS_PYYAML):
            raise ImportError(
                "graphviz and pyyaml package not installed."
                " Install with: 'pip install graphviz pyyaml'"
            )

        renderer = GraphvizRenderer(self._ir)

        ret = renderer.render(add_params=add_params, add_description=add_description)
        # print(ret)
        return ret

    def get_node_ids(self) -> list[str]:
        """Get all node IDs in the pipeline.

        Useful for debugging and setting up checkpoints.

        Returns:
            List of node IDs in execution order.
        """
        return [node.id for node in self._ir.walk()]

    def find_node(self, node_id: str):
        """Find a node by its ID.

        Args:
            node_id: The node ID to find.

        Returns:
            The node if found, None otherwise.
        """
        return self._ir.find_by_id(node_id)


def _example_flat_pipeline():  # pragma: no cover
    from nebula.transformers import SelectColumns
    import polars as pl

    def my_function(_df):
        print(f"my function print -> this is a: {type(_df)}")
        return _df

    example_df_input = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    pipeline = TransformerPipeline(
        [SelectColumns(columns=["a"]), my_function],
        # name="my_pipeline",
    )
    # show the pipeline without running it ...
    pipeline.show(add_params=True)
    # ... now run it
    example_df_output = pipeline.run(example_df_input)
    print(example_df_output)


if __name__ == "__main__":  # pragma: no cover
    _example_flat_pipeline()
