"""Pipeline executor - runs the pipeline IR.

The executor traverses the IR tree and executes each node,
handling:
- Transformers and functions
- Storage operations
- Forks (split, branch, apply_to_rows)
- Merges (append, join, dead_end)
- Checkpointing and resume
- Error recovery with fail cache

Design:
- Single-pass traversal of IR
- Context carries DataFrame and state
- Hooks called at key points for extensibility
- Fail cache stores DataFrames for error recovery
"""

from __future__ import annotations

from typing import Callable

import narwhals as nw

from nebula.base import Transformer
from nebula.df_types import GenericDataFrame, is_natively_spark
from nebula.nw_util import (
    append_dataframes,
    df_is_empty,
    join_dataframes,
    safe_from_native,
    safe_to_native,
)
from nebula.pipelines.pipe_aux import get_native_schema, split_df, to_schema
from nebula.storage import nebula_storage as ns

from ..exceptions import raise_pipeline_error
from ..ir.nodes import (
    ConversionNode,
    ForkNode,
    FunctionNode,
    InputNode,
    MergeNode,
    OutputNode,
    PipelineNode,
    SequenceNode,
    StorageNode,
    TransformerNode,
)
from .context import ExecutionContext
from .hooks import NoOpHooks, PipelineHooks

__all__ = ["PipelineExecutor", "execute_pipeline"]


def _get_n_partitions(df: "GenericDataFrame") -> int:
    """Get the number of partitions if the DF is a spark one and if requested."""
    is_spark, df = is_natively_spark(df)
    if is_spark:
        return df.rdd.getNumPartitions()
    return 0


def _repartition_coalesce(df, cfg: dict, n: int) -> "GenericDataFrame":
    """Repartition / coalesce if "df" is a spark DF and if requested."""
    if isinstance(df, (nw.DataFrame, nw.LazyFrame)):
        df = nw.to_native(df)
    if cfg.get("repartition_output_to_original"):
        df = df.repartition(n)
    elif cfg.get("coalesce_output_to_original"):
        df = df.coalesce(n)
    return df


class PipelineExecutor:
    """Executes a pipeline IR tree.

    The executor walks the IR tree, executing each node and managing
    the DataFrame state through the ExecutionContext.

    Example:
        executor = PipelineExecutor(
            ir=pipeline._ir,
            hooks=LoggingHooks(),
        )
        result_df = executor.run(input_df)
    """

    def __init__(
        self,
        ir: "SequenceNode",
        *,
        hooks: PipelineHooks | None = None,
        resume_from: str | None = None,
        checkpoint_storage: str | None = None,
        force_interleaved: Transformer | Callable | None = None,
    ):
        """Initialize the executor.

        Args:
            ir: The root IR node (SequenceNode).
            hooks: Optional hooks for monitoring/extensibility.
            resume_from: Node ID to resume from (skip prior nodes).
            checkpoint_storage: Storage key prefix for checkpoints.
            force_interleaved: Optional transformer / function to run after each step.
        """
        self.ir: "SequenceNode" = ir
        self.hooks = hooks or NoOpHooks()
        self.resume_from = resume_from
        self.checkpoint_storage = checkpoint_storage
        self.force_interleaved = force_interleaved

    def run(self, df: "GenericDataFrame") -> "GenericDataFrame":
        """Execute the pipeline on input DataFrame.

        Args:
            df: Input DataFrame (any supported backend).

        Returns:
            Transformed DataFrame.
        """
        # Track if input was native (not narwhals)
        input_was_native = not isinstance(df, (nw.DataFrame, nw.LazyFrame))

        # Create execution context
        ctx = ExecutionContext(
            df=df,
            resume_from=self.resume_from,
            should_skip=bool(self.resume_from),
            checkpoint_storage=self.checkpoint_storage,
        )

        # Notify hooks
        self.hooks.on_pipeline_start(self.ir, {"df": df})

        try:
            # Execute the IR tree
            ctx = self._execute_node(self.ir, ctx)
        except Exception as e:
            # Handle failure - store cached DFs if any
            if ctx.fail_cache:
                keys = [f"'{k}'" for k in sorted(ctx.fail_cache.keys())]
                keys = ", ".join(keys)
                msg = (
                    "Get the dataframe(s) before the failure in the nebula "
                    f"storage with the key(s): [{keys}]\nOriginal Error:"
                )
                self._store_fail_cache(ctx)
                raise_pipeline_error(e, msg)

            raise e

        # Notify hooks
        self.hooks.on_pipeline_end(self.ir, ctx.elapsed_ms(), {"df": ctx.df})

        # Convert back to native if input was native
        result_df = ctx.df
        if input_was_native and isinstance(result_df, (nw.DataFrame, nw.LazyFrame)):
            result_df = nw.to_native(result_df)

        return result_df

    def _execute_node(self, node: "PipelineNode", ctx: ExecutionContext) -> ExecutionContext:
        """Execute a single node, dispatching to type-specific handler."""
        # Check if we should skip (resume logic)
        should_skip, reason = ctx.should_skip_node(node.id)
        if should_skip:
            self.hooks.on_skip(node, reason, {"df": ctx.df})
            return ctx

        # Dispatch to type-specific handler
        if isinstance(node, SequenceNode):
            return self._execute_sequence(node, ctx)
        elif isinstance(node, TransformerNode):
            return self._execute_transformer(node, ctx)
        elif isinstance(node, FunctionNode):
            return self._execute_function(node, ctx)
        elif isinstance(node, StorageNode):
            return self._execute_storage(node, ctx)
        elif isinstance(node, ConversionNode):
            return self._execute_conversion(node, ctx)
        elif isinstance(node, ForkNode):
            return self._execute_fork(node, ctx)
        elif isinstance(node, MergeNode):
            return self._execute_merge(node, ctx)
        elif isinstance(node, InputNode):
            return self._execute_input(node, ctx)
        elif isinstance(node, OutputNode):
            return self._execute_output(node, ctx)
        else:  # pragma: no cover
            raise TypeError(f"Unknown node type: {type(node)}")

    def _execute_sequence(self, node: "SequenceNode", ctx: ExecutionContext) -> ExecutionContext:
        """Execute a sequence of nodes."""
        for step in node.steps:
            ctx = self._execute_node(step, ctx)
        return ctx

    def _execute_transformer(self, node: "TransformerNode", ctx: ExecutionContext) -> ExecutionContext:
        """Execute a transformer node."""
        ctx.start_node(node.id)
        self.hooks.on_node_start(node, {"df": ctx.df})

        # Cache for failure recovery
        ctx.cache_for_failure(f"transformer:{node.transformer_name}", ctx.df)

        try:
            ctx.df = node.transformer.transform(ctx.df)

            # Run forced interleaved if set
            if self.force_interleaved is not None:
                ctx.df = self.force_interleaved.transform(ctx.df)

        except Exception as e:
            self.hooks.on_error(node, e, {"df": ctx.df})
            raise

        ctx.clear_fail_cache()
        duration = ctx.end_node(node.id)
        self.hooks.on_node_end(node, duration, {"df": ctx.df})

        return ctx

    def _execute_function(self, node: "FunctionNode", ctx: ExecutionContext) -> ExecutionContext:
        """Execute a function node."""
        ctx.start_node(node.id)
        self.hooks.on_node_start(node, {"df": ctx.df})

        # Cache for failure recovery
        ctx.cache_for_failure(f"function:{node.func_name}", ctx.df)

        try:
            ctx.df = node.func(ctx.df, *node.args, **node.kwargs)

            # Run forced interleaved if set
            if self.force_interleaved is not None:
                ctx.df = self.force_interleaved.transform(ctx.df)

        except Exception as e:
            self.hooks.on_error(node, e, {"df": ctx.df})
            raise

        ctx.clear_fail_cache()
        duration = ctx.end_node(node.id)
        self.hooks.on_node_end(node, duration, {"df": ctx.df})

        return ctx

    def _execute_storage(self, node: "StorageNode", ctx: ExecutionContext) -> ExecutionContext:
        """Execute a storage operation."""
        ctx.start_node(node.id)
        self.hooks.on_node_start(node, {"df": ctx.df})

        if node.operation == "store":
            ns.set(node.key, ctx.df)

        elif node.operation == "store_debug":
            ns.set(node.key, ctx.df, debug=True)

        elif node.operation == "load":
            ctx.df = ns.get(node.key)

        elif node.operation == "toggle_debug":
            ns.allow_debug(node.debug_value)

        duration = ctx.end_node(node.id)
        self.hooks.on_node_end(node, duration, {"df": ctx.df})

        return ctx

    def _execute_conversion(self, node: "ConversionNode", ctx: ExecutionContext) -> ExecutionContext:
        """Execute a conversion operation."""
        ctx.start_node(node.id)
        self.hooks.on_node_start(node, {"df": ctx.df})

        if node.operation == "to_native":
            ctx.df = safe_to_native(ctx.df)
        elif node.operation == "from_native":
            ctx.df = safe_from_native(ctx.df)

        duration = ctx.end_node(node.id)
        self.hooks.on_node_end(node, duration, {"df": ctx.df})

        return ctx

    def _execute_fork(self, node: "ForkNode", ctx: ExecutionContext) -> ExecutionContext:
        """Execute a fork node (split, branch, apply_to_rows)."""
        ctx.start_node(node.id)
        self.hooks.on_node_start(node, {"df": ctx.df})

        # Cache for failure recovery (e.g., if split_function fails)
        ctx.cache_for_failure(f"fork:{node.fork_type}", ctx.df)

        if node.config.get("cast_subsets_to_input_schema"):
            node.config["input_schema"] = get_native_schema(ctx.df)

        if node.config.get("repartition_output_to_original") or node.config.get("coalesce_output_to_original"):
            n_part_orig: int = _get_n_partitions(ctx.df)
            if n_part_orig:
                node.config["spark_input_partitions"] = n_part_orig

        if node.fork_type == "split":
            ctx = self._execute_split_fork(node, ctx)
        elif node.fork_type == "branch":
            ctx = self._execute_branch_fork(node, ctx)
        elif node.fork_type == "apply_to_rows":
            ctx = self._execute_apply_to_rows_fork(node, ctx)
        else:  # pragma: no cover
            raise ValueError(f"Unknown fork type: {node.fork_type}")

        ctx.clear_fail_cache()
        duration = ctx.end_node(node.id)
        self.hooks.on_node_end(node, duration, {"df": ctx.df})

        return ctx

    def _execute_split_fork(self, node: "ForkNode", ctx: ExecutionContext) -> ExecutionContext:
        """Execute a split fork - DataFrame split by function into branches."""
        # Call split function
        split_dfs = node.split_function(ctx.df)

        # Verify keys match
        expected_keys = set(node.branches.keys())
        actual_keys = set(split_dfs.keys())
        if expected_keys != actual_keys:
            diff = expected_keys.symmetric_difference(actual_keys)
            raise KeyError(f"Split function keys don't match branches: {diff}")

        # Execute each branch and collect results
        branch_results = {}
        dead_end_splits = node.config.get("splits_no_merge", set())
        skip_if_empty = node.config.get("splits_skip_if_empty", set())

        for branch_name, branch_steps in node.branches.items():
            branch_df = split_dfs[branch_name]

            # Skip if empty (if configured)
            if branch_name in skip_if_empty:
                if df_is_empty(branch_df):
                    continue

            # Execute branch
            branch_ctx = ctx.clone_with_df(branch_df)
            for step in branch_steps:
                branch_ctx = self._execute_node(step, branch_ctx)

            # Store result (unless dead-end)
            if branch_name not in dead_end_splits:
                branch_results[branch_name] = branch_ctx.df

        # Store results in metadata for merge node
        ctx.metadata["branch_results"] = branch_results
        ctx.metadata["fork_config"] = node.config

        return ctx

    def _execute_branch_fork(self, node: "ForkNode", ctx: ExecutionContext) -> ExecutionContext:
        """Execute a branch fork - secondary pipeline from main or stored df."""
        # Get input for branch
        storage_key = node.config.get("storage")
        if storage_key:
            branch_df = ns.get(storage_key)
        else:
            branch_df = ctx.df

        # Execute main branch
        main_steps = node.branches.get("main", [])
        branch_ctx = ctx.clone_with_df(branch_df)
        for step in main_steps:
            branch_ctx = self._execute_node(step, branch_ctx)

        # Execute otherwise if present
        if node.otherwise:
            otherwise_ctx = ctx.clone_with_df(ctx.df)
            for step in node.otherwise:
                otherwise_ctx = self._execute_node(step, otherwise_ctx)
            otherwise_df = otherwise_ctx.df
        else:
            # Branch from primary df, no otherwise pipeline
            # Preserve original df for merge
            otherwise_df = ctx.df

        # Store results for merge
        ctx.metadata["branch_results"] = {
            "main": branch_ctx.df,
            "otherwise": otherwise_df,
        }
        ctx.metadata["fork_config"] = node.config
        ctx.metadata["original_df"] = ctx.df

        return ctx

    def _execute_apply_to_rows_fork(self, node: "ForkNode", ctx: ExecutionContext) -> ExecutionContext:
        """Execute apply_to_rows fork - filter, transform, merge back."""
        # Get condition from config
        df = ctx.df
        if not isinstance(df, (nw.DataFrame, nw.LazyFrame)):
            df = nw.from_native(df)

        # Split DataFrame
        matched_df, otherwise_df = split_df(df, node.config)

        # Check skip_if_empty
        matched_result = None
        if node.config.get("skip_if_empty") and df_is_empty(matched_df):
            pass  # Skip matched branch
        else:
            # Execute matched branch
            matched_steps = node.branches.get("matched", [])
            matched_ctx = ctx.clone_with_df(matched_df)
            for step in matched_steps:
                matched_ctx = self._execute_node(step, matched_ctx)
            matched_result = matched_ctx.df

        # Execute otherwise if present
        otherwise_result = otherwise_df
        if node.otherwise:
            otherwise_ctx = ctx.clone_with_df(otherwise_df)
            for step in node.otherwise:
                otherwise_ctx = self._execute_node(step, otherwise_ctx)
            otherwise_result = otherwise_ctx.df

        # Store for merge
        ctx.metadata["branch_results"] = {
            "matched": matched_result,
            "otherwise": otherwise_result,
        }
        ctx.metadata["fork_config"] = node.config

        return ctx

    def _execute_merge(  # noqa: PLR0915
        self, node: "MergeNode", ctx: ExecutionContext
    ) -> ExecutionContext:
        """Execute a merge node."""
        ctx.start_node(node.id)
        self.hooks.on_node_start(node, {"df": ctx.df})

        branch_results = ctx.metadata.get("branch_results", {})
        fork_config = ctx.metadata.get("fork_config", {})

        if node.merge_type == "dead-end":
            # For dead-end, use the otherwise result or original df
            otherwise_df = branch_results.get("otherwise")
            original_df = ctx.metadata.get("original_df", ctx.df)
            ctx.df = otherwise_df if otherwise_df is not None else original_df

        elif node.merge_type == "append":
            names = []
            dfs_to_append = []
            # Collect all DataFrames to append
            splits_no_merge = fork_config.get("splits_no_merge", set())
            for name, df in branch_results.items():
                if name in splits_no_merge:
                    continue
                if df is not None:
                    names.append(name)
                    dfs_to_append.append(df)

            if dfs_to_append:
                if node.config.get("cast_subsets_to_input_schema"):
                    for name, df in zip(names, dfs_to_append):
                        ctx.cache_for_failure(f"{name}-df-before-casting:{node.merge_type}", df)
                    input_schema = fork_config["input_schema"]
                    dfs_to_append = to_schema(dfs_to_append, input_schema)
                    # cast, clear cache
                    ctx.clear_fail_cache()

                for name, df in zip(names, dfs_to_append):
                    ctx.cache_for_failure(f"{name}-df-before-appending:{node.merge_type}", df)
                allow_missing = node.config.get("allow_missing_columns", False)
                ctx.df = append_dataframes(dfs_to_append, allow_missing_cols=allow_missing)
                # appended, clear cache
                ctx.clear_fail_cache()

        elif node.merge_type == "join":
            # Join main result with original (or otherwise)
            main_df = branch_results.get("main")
            otherwise_df = branch_results.get("otherwise")
            base_df = otherwise_df if otherwise_df is not None else ctx.metadata.get("original_df", ctx.df)

            ctx.cache_for_failure(f"join-left-df:{node.merge_type}", base_df)
            ctx.cache_for_failure(f"join-right-df:{node.merge_type}", main_df)
            ctx.df = join_dataframes(
                base_df,
                main_df,
                how=node.config.get("how", "inner"),
                on=node.config.get("on"),
                left_on=node.config.get("left_on"),
                right_on=node.config.get("right_on"),
                suffix=node.config.get("suffix"),
                broadcast=node.config.get("broadcast", False),
            )
            # joined, clear the fail_cache
            ctx.clear_fail_cache()

        # Run forced interleaved if set
        if self.force_interleaved is not None:
            ctx.cache_for_failure(f"last-interleaved:{node.merge_type}", ctx.df)
            ctx.df = self.force_interleaved.transform(ctx.df)
            # executed, clear the fail_cache
            ctx.clear_fail_cache()

        if node.merge_type != "dead-end":
            n_part_orig = fork_config.get("spark_input_partitions")
            if n_part_orig:  # 0 if backed != spark
                ctx.cache_for_failure(f"spark-partitions:{node.merge_type}", ctx.df)
                ctx.df = _repartition_coalesce(ctx.df, fork_config, n_part_orig)

        # Clear branch metadata
        ctx.metadata.pop("branch_results", None)
        ctx.metadata.pop("fork_config", None)
        ctx.metadata.pop("original_df", None)

        ctx.clear_fail_cache()
        duration = ctx.end_node(node.id)
        self.hooks.on_node_end(node, duration, {"df": ctx.df})

        return ctx

    @staticmethod
    def _execute_input(node: "InputNode", ctx: ExecutionContext) -> ExecutionContext:
        """Execute input node (no-op, just marks start)."""
        ctx.start_node(node.id)
        return ctx

    @staticmethod
    def _execute_output(node: "OutputNode", ctx: ExecutionContext) -> ExecutionContext:
        """Execute output node (no-op, just marks end)."""
        ctx.start_node(node.id)
        return ctx

    @staticmethod
    def _store_fail_cache(ctx: ExecutionContext) -> None:
        """Store cached DataFrames to nebula_storage on failure."""
        for key, df in ctx.fail_cache.items():
            storage_key = f"FAIL_DF_{key}"
            ns.set(storage_key, df)


def execute_pipeline(
    ir: "SequenceNode",
    df: "GenericDataFrame",
    *,
    hooks: PipelineHooks | None = None,
    resume_from: str | None = None,
    **kwargs,
) -> "GenericDataFrame":  # pragma: no cover
    """Convenience function to execute a pipeline IR.

    Args:
        ir: The pipeline IR root node.
        df: Input DataFrame.
        hooks: Optional hooks for monitoring.
        resume_from: Node ID to resume from.
        **kwargs: Additional executor options.

    Returns:
        Transformed DataFrame.
    """
    executor = PipelineExecutor(
        ir=ir,
        hooks=hooks,
        resume_from=resume_from,
        **kwargs,
    )
    return executor.run(df)
