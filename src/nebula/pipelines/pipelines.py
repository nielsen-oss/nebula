"""TransformerPipeline - the main user-facing pipeline class.

This module provides the public API for creating and running pipelines.
It wraps the IR building, execution, and visualization components.

The TransformerPipeline class maintains backward compatibility with
the existing API while using the new IR-based architecture internally.

Example:
    from nebula.pipelines import TransformerPipeline
    
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
from nebula.pipelines.execution import PipelineExecutor, PipelineHooks, NoOpHooks, LoggingHooks
from nebula.pipelines.ir import IRBuilder, SequenceNode
from nebula.pipelines.ir.nodes import TransformerNode, FunctionNode
from nebula.pipelines.pipe_aux import is_keyword_request, is_split_pipeline, PIPELINE_KEYWORDS
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
        
        Args:
            data: Pipeline content. Can be:
                - A single Transformer
                - A list of Transformers and/or functions
                - A dict mapping split names to transformer lists
                - A nested TransformerPipeline
            name: Optional pipeline name for display/logging.
            df_input_name: Name for input DataFrame in visualizations.
            df_output_name: Name for output DataFrame in visualizations.
            interleaved: Debug transformers to insert between steps.
            prepend_interleaved: Insert interleaved at start.
            append_interleaved: Insert interleaved at end.
            split_function: Function to split DataFrame for split pipelines.
            split_order: Execution order for splits (default: alphabetical).
            split_apply_after_splitting: Transformers to run after split.
            split_apply_before_appending: Transformers to run before merge.
            splits_no_merge: Split names to exclude from merge (dead-ends).
            splits_skip_if_empty: Skip split if input is empty.
            branch: Branch configuration dict.
            apply_to_rows: Apply-to-rows configuration dict.
            otherwise: Pipeline for non-matched rows.
            allow_missing_columns: Allow column mismatches in merge.
            cast_subsets_to_input_schema: Cast splits to input schema.
            repartition_output_to_original: Spark: repartition after merge.
            coalesce_output_to_original: Spark: coalesce after merge.
            skip: If True, create empty pipeline.
            perform: If False, create empty pipeline.
        """
        validate_skip_perform(skip, perform)
        ensure_no_branch_or_apply_to_rows_otherwise(branch, apply_to_rows, otherwise)
        assert_at_most_one_args(
            cast_subset_to_input_schema=cast_subsets_to_input_schema,
            allow_missing_columns=allow_missing_columns,
        )

        if split_function is not None:
            if not callable(split_function):
                raise TypeError("If provided, the 'split_function' must be "
                                f"a callable, found {type(split_function)}")

        if is_split_pipeline(data, split_function):
            if not callable(split_function):
                raise TypeError("For split-pipelines, the 'split_function' "
                                f"must be callable, found {type(split_function)}")

            ensure_no_branch_or_apply_to_rows_in_split_pipeline(branch, apply_to_rows)

            if split_order:
                assert_split_order(data, split_order)

            # Normalize splits_no_merge and splits_skip_if_empty to sets
            splits_no_merge = set_split_options(data, splits_no_merge, "splits_no_merge")
            splits_skip_if_empty = set_split_options(data, splits_skip_if_empty, "splits_skip_if_empty")

        else:
            if isinstance(data, dict) and not is_keyword_request(data):
                raise ValueError("Unknown input, For split-pipelines the "
                                 "number of splits must be > 1, found "
                                 f"{len(data)}, key-word operations the dictionary "
                                 "must be single key-value pair with a valid key"
                                 f"{PIPELINE_KEYWORDS}")

        self.name = name

        # Handle skip/perform
        if skip or (perform is False):
            data = []
            branch = None
            apply_to_rows = None
            otherwise = None

        interleaved = to_list_of_transformations(interleaved, "interleaved")
        split_apply_after_splitting = to_list_of_transformations(
            split_apply_after_splitting, "split_apply_after_splitting")
        split_apply_before_appending = to_list_of_transformations(
            split_apply_before_appending, "split_apply_before_appending")

        otherwise_only: bool = False
        if apply_to_rows:
            assert_apply_to_rows_inputs(apply_to_rows)
            if apply_to_rows.get('skip') or (apply_to_rows.get("perform") is False):
                otherwise_only = True
        if branch:
            assert_branch_inputs(branch)
            if branch.get('skip') or (branch.get("perform") is False):
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
            'name': name,
            'branch': branch,
            'apply_to_rows': apply_to_rows,
            'allow_missing_columns': allow_missing_columns,
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
            hooks = LoggingHooks(max_param_length=PIPE_CFG["max_param_length"],
                                 show_params=show_params)

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
        printer = PipelinePrinter(self._ir, max_param_length=PIPE_CFG["max_param_length"],
                                  indent_size=indent_size)
        printer.print(add_params=add_params, add_ids=add_ids)

    def to_string(self, *, add_params: bool = False) -> str:
        """Get pipeline structure as string.
        
        Args:
            add_params: Include transformer parameters.
        
        Returns:
            Multi-line string representation.
        """
        printer = PipelinePrinter(self._ir, max_param_length=PIPE_CFG["max_param_length"])
        return printer.to_string(add_params=add_params)

    def plot(
            self,
            *,
            add_params: bool = False,
            add_description: bool = False,
    ):
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
        from .visualization import GraphvizRenderer, HAS_GRAPHVIZ  # FIXME: here?

        if not HAS_GRAPHVIZ:
            raise ImportError(
                "graphviz package not installed. "
                "Install with: pip install graphviz"
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

    def get_number_transformers(self) -> int:
        """Get the number of transformers in the pipeline.
        
        Returns:
            Count of transformer and function nodes.
        """
        # from .ir.nodes import TransformerNode, FunctionNode  # FIXME: delete

        count = 0
        for node in self._ir.walk():
            if isinstance(node, (TransformerNode, FunctionNode)):
                count += 1
        return count


def _example_flat_pipeline():
    from nebula.transformers import SelectColumns
    import polars as pl

    def my_function(_df):
        print(f"my function print -> this is a: {type(_df)}")
        return _df

    example_df_input = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    pipeline = TransformerPipeline(
        [SelectColumns(columns=['a']), my_function],
        # name="my_pipeline",
    )
    # show the pipeline without running it
    pipeline.show(add_params=True)
    example_df_output = pipeline.run(example_df_input)
    print(example_df_output)


if __name__ == "__main__":
    _example_flat_pipeline()
