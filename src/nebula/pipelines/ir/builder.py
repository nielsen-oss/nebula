"""IR Builder - constructs pipeline IR from user input.

This module takes the user's pipeline definition (transformers, dicts,
nested pipelines) and builds a clean IR tree that can be traversed
by executors, printers, and renderers.

Design decisions:
1. Interleaved transformers are expanded during build (explicit > implicit)
2. Storage requests are parsed and converted to StorageNodes
3. Nested TransformerPipelines are flattened into the IR
4. Bare functions are wrapped in FunctionNodes
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from nebula.pipelines.pipe_aux import is_split_pipeline, sanitize_steps

from .. import transformer_type_util
from .node_id import assign_ids_to_tree
from .nodes import (
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

__all__ = ["IRBuilder", "build_ir"]


class IRBuilder:
    """Builds pipeline IR from user input.

    The builder handles:
    - Linear pipelines (list of transformers)
    - Split pipelines (dict with split function)
    - Branch pipelines (fork from main or stored df)
    - Apply-to-rows pipelines (filter, transform, merge back)
    - Interleaved transformers (debug tools inserted between steps)
    - Storage operations (store, load, debug toggle)
    - Bare functions as transformers

    Example:
        builder = IRBuilder(
            data=[transformer1, transformer2],
            name="my_pipeline",
            interleaved=[CountRows()],
        )
        ir = builder.build()
    """

    def __init__(
        self,
        data,
        *,
        name: str | None = None,
        df_input_name: str = "DF input",
        df_output_name: str = "DF output",
        interleaved: list | None = None,
        prepend_interleaved: bool = False,
        append_interleaved: bool = False,
        split_function: Callable | None = None,
        split_order: list[str] | None = None,
        split_apply_after_splitting: list | None = None,
        split_apply_before_appending: list | None = None,
        splits_no_merge: set[str] | None = None,
        splits_skip_if_empty: set[str] | None = None,
        branch: dict | None = None,
        apply_to_rows: dict | None = None,
        otherwise: Any | None = None,
        allow_missing_columns: bool = False,
        cast_subsets_to_input_schema: bool = False,
        repartition_output_to_original: bool = False,
        coalesce_output_to_original: bool = False,
    ):
        """Initialize the intermediate-representation."""
        self.data = data
        self.name = name
        self.df_input_name = df_input_name
        self.df_output_name = df_output_name

        self.interleaved = interleaved or []
        self.prepend_interleaved = prepend_interleaved
        self.append_interleaved = append_interleaved

        self.split_function = split_function
        self.split_order = split_order
        self.split_apply_after_splitting = split_apply_after_splitting or []
        self.split_apply_before_appending = split_apply_before_appending or []
        self.splits_no_merge = splits_no_merge or set()
        self.splits_skip_if_empty = splits_skip_if_empty or set()

        self.branch = deepcopy(branch) if branch else None
        self.apply_to_rows = deepcopy(apply_to_rows) if apply_to_rows else None
        self.otherwise = otherwise

        self.allow_missing_columns = allow_missing_columns
        self.cast_subsets_to_input_schema = cast_subsets_to_input_schema
        self.repartition_output_to_original = repartition_output_to_original
        self.coalesce_output_to_original = coalesce_output_to_original

    def build(self) -> SequenceNode:
        """Build the IR tree from the configuration.

        Returns:
            Root SequenceNode containing the full pipeline IR.
        """
        # Create root sequence
        root = SequenceNode(name=self.name)

        # Add input node
        input_node = InputNode(name=self.df_input_name)
        root.add_step(input_node)

        # Determine pipeline type and build accordingly
        if is_split_pipeline(self.data, self.split_function):
            self._build_split_pipeline(root)
        elif self.branch:
            self._build_branch_pipeline(root)
        elif self.apply_to_rows:
            self._build_apply_to_rows_pipeline(root)
        else:
            self._build_linear_pipeline(root)

        # Add output node
        output_node = OutputNode(name=self.df_output_name)
        root.add_step(output_node)

        # Assign IDs to all nodes
        assign_ids_to_tree(root)

        return root

    def _build_linear_pipeline(self, root: SequenceNode) -> None:
        """Build a linear (sequential) pipeline."""
        steps = self._create_steps_with_interleaved(self.data)
        for step in steps:
            root.add_step(step)

    def _build_split_pipeline(self, root: SequenceNode) -> None:
        """Build a split pipeline with multiple branches."""
        # Determine split order
        if self.split_order:
            order = self.split_order
        else:
            order = sorted(self.data.keys())

        # Create fork node
        fork = ForkNode(
            fork_type="split",
            config={
                "splits_no_merge": self.splits_no_merge,
                "splits_skip_if_empty": self.splits_skip_if_empty,
                "cast_subsets_to_input_schema": self.cast_subsets_to_input_schema,
                "repartition_output_to_original": self.repartition_output_to_original,
                "coalesce_output_to_original": self.coalesce_output_to_original,
            },
            split_function=self.split_function,
        )

        # Build each branch
        for split_name in order:
            split_data = self.data.get(split_name, [])
            branch_steps = []

            # Add after-splitting transformers
            for trf in self.split_apply_after_splitting:
                branch_steps.append(self._create_node(trf))

            # Add the split's own transformers with interleaved
            split_steps = self._create_steps_with_interleaved(split_data)
            branch_steps.extend(split_steps)

            # Add before-appending transformers
            for trf in self.split_apply_before_appending:
                branch_steps.append(self._create_node(trf))

            fork.branches[split_name] = branch_steps
            # Set parent references
            for step in branch_steps:
                step.parent = fork

        root.add_step(fork)

        # Create merge node
        merge = MergeNode(
            merge_type="append",
            config={
                "allow_missing_columns": self.allow_missing_columns,
                "cast_subsets_to_input_schema": self.cast_subsets_to_input_schema,
                "repartition_output_to_original": self.repartition_output_to_original,
                "coalesce_output_to_original": self.coalesce_output_to_original,
                "splits_no_merge": self.splits_no_merge,
            },
        )
        root.add_step(merge)

    def _build_branch_pipeline(self, root: SequenceNode) -> None:
        """Build a branch pipeline (fork from main or stored df)."""
        end_type = self.branch["end"]
        storage_key = self.branch.get("storage")

        # Create fork node
        fork = ForkNode(
            fork_type="branch",
            config={
                "storage": storage_key,
                "end": end_type,
                "how": self.branch.get("how"),
                "on": self.branch.get("on"),
                "left_on": self.branch.get("left_on"),
                "right_on": self.branch.get("right_on"),
                "suffix": self.branch.get("suffix"),
                "broadcast": self.branch.get("broadcast", False),
                "allow_missing_columns": self.allow_missing_columns,
                "cast_subsets_to_input_schema": self.cast_subsets_to_input_schema,
                "repartition_output_to_original": self.repartition_output_to_original,
                "coalesce_output_to_original": self.coalesce_output_to_original,
            },
        )

        # Build main branch
        main_steps = self._create_steps_with_interleaved(self.data)
        fork.branches["main"] = main_steps
        for step in main_steps:
            step.parent = fork

        # Build otherwise branch if provided
        if self.otherwise:
            otherwise_steps = self._create_steps_with_interleaved(self.otherwise)
            fork.otherwise = otherwise_steps
            for step in otherwise_steps:
                step.parent = fork

        root.add_step(fork)

        # Create merge node (unless dead-end)
        if end_type == "dead-end":
            merge_type = "dead-end"
        elif end_type == "join":
            merge_type = "join"
        else:
            merge_type = "append"

        merge = MergeNode(
            merge_type=merge_type,
            config={
                "how": self.branch.get("how"),
                "on": self.branch.get("on"),
                "left_on": self.branch.get("left_on"),
                "right_on": self.branch.get("right_on"),
                "suffix": self.branch.get("suffix"),
                "broadcast": self.branch.get("broadcast", False),
                "allow_missing_columns": self.allow_missing_columns,
                "cast_subsets_to_input_schema": self.cast_subsets_to_input_schema,
                "repartition_output_to_original": self.repartition_output_to_original,
                "coalesce_output_to_original": self.coalesce_output_to_original,
            },
        )
        root.add_step(merge)

    def _build_apply_to_rows_pipeline(self, root: SequenceNode) -> None:
        """Build an apply-to-rows pipeline (filter, transform, merge)."""
        is_dead_end = self.apply_to_rows.get("dead-end", False)

        # Create fork node
        fork = ForkNode(
            fork_type="apply_to_rows",
            config={
                "input_col": self.apply_to_rows.get("input_col"),
                "operator": self.apply_to_rows.get("operator"),
                "value": self.apply_to_rows.get("value"),
                "comparison_column": self.apply_to_rows.get("comparison_column"),
                "dead-end": is_dead_end,
                "skip_if_empty": self.apply_to_rows.get("skip_if_empty", False),
                "cast_subsets_to_input_schema": self.cast_subsets_to_input_schema,
                "repartition_output_to_original": self.repartition_output_to_original,
                "coalesce_output_to_original": self.coalesce_output_to_original,
            },
        )

        # Build matched rows branch
        matched_steps = self._create_steps_with_interleaved(self.data)
        fork.branches["matched"] = matched_steps
        for step in matched_steps:
            step.parent = fork

        # Build otherwise branch if provided
        if self.otherwise:
            otherwise_steps = self._create_steps_with_interleaved(self.otherwise)
            fork.otherwise = otherwise_steps
            for step in otherwise_steps:
                step.parent = fork
        else:
            fork.otherwise = []

        root.add_step(fork)

        # Create merge node
        merge = MergeNode(
            merge_type="dead-end" if is_dead_end else "append",
            config={
                "allow_missing_columns": self.allow_missing_columns,
                "repartition_output_to_original": self.repartition_output_to_original,
                "coalesce_output_to_original": self.coalesce_output_to_original,
            },
        )
        root.add_step(merge)

    def _create_steps_with_interleaved(self, data) -> list[PipelineNode]:
        """Create nodes with interleaved transformers inserted.

        Interleaved transformers are debugging tools inserted between
        each "real" transformer for observability.
        """
        if not data:
            return []

        # Sanitize input to list
        if isinstance(data, (list, tuple)):
            data = sanitize_steps(data)
        else:
            data = [data]

        # If no interleaved, just create nodes
        if not self.interleaved:
            return [self._create_node(item) for item in data if item is not None]

        steps = []

        # Optionally prepend interleaved
        if self.prepend_interleaved:
            for trf in self.interleaved:
                steps.append(self._create_node(trf))

        # Process items with interleaved between
        for i, item in enumerate(data):
            if item is None:
                continue

            node = self._create_node(item)
            steps.append(node)

            # Add interleaved after each transformer (not after storage ops)
            is_last = i == len(data) - 1
            should_add_interleaved = isinstance(node, (TransformerNode, FunctionNode)) and (
                not is_last or self.append_interleaved
            )

            if should_add_interleaved:
                for trf in self.interleaved:
                    steps.append(self._create_node(trf))

        return steps

    def _create_node(self, item) -> PipelineNode | None:
        """Create appropriate node type from an item.

        Handles:
        - Transformer instances
        - LazyWrapper instances
        - Bare functions
        - (func, args, kwargs) tuples
        - Storage request dicts
        - Nested TransformerPipeline instances
        """
        # Handle None
        if item is None:
            return None

        # Check for storage request first (it's a dict)
        storage_op = self._parse_storage_request(item)
        if storage_op:
            return storage_op

        # Check for conversion request (string keywords)
        conversion_op = self._parse_conversion_request(item)
        if conversion_op:
            return conversion_op

        # Handle Transformer or LazyWrapper
        if transformer_type_util.is_transformer(item):
            description = None
            if hasattr(item, "get_description"):
                description = item.get_description()
            return TransformerNode(transformer=item, description=description)

        if (
            isinstance(item, (tuple, list))
            and (len(item) == 2)
            and transformer_type_util.is_transformer(item[0])
            and isinstance(item[1], str)
        ):
            return TransformerNode(transformer=item[0], description=item[1])

        # Handle bare function
        if callable(item):
            return FunctionNode(func=item)

        # Handle (func, args, kwargs, description) tuple
        if isinstance(item, (tuple, list)) and len(item) >= 1:
            if callable(item[0]):
                n = len(item)
                if n > 4:  # pragma: no cover
                    raise ValueError(
                        "A function passed in a tuple must be "
                        "not longer than 4 elements: "
                        "(function, args, kwargs, description"
                    )
                keys = ["func", "args", "kwargs", "description"]
                inputs = {}
                for i in range(n):
                    inputs[keys[i]] = item[i]
                return FunctionNode(**inputs)

        # Handle nested TransformerPipeline
        if hasattr(item, "_ir"):
            # It's already a TransformerPipeline with built IR
            # Extract and embed its steps (skip input/output nodes)
            nested_steps = []
            for step in item._ir.steps:
                if not isinstance(step, (InputNode, OutputNode)):
                    nested_steps.append(step)

            if len(nested_steps) == 1:
                return nested_steps[0]

            # Wrap in a sequence
            seq = SequenceNode(name=item.name)
            seq.steps = nested_steps
            seq.children = nested_steps
            return seq

        # Handle dict that might be a nested pipeline config
        if isinstance(item, dict):
            # This shouldn't normally happen at this stage,
            # but handle it gracefully
            raise TypeError(
                f"Unexpected dict in pipeline data: {item}. Storage requests should have been parsed already."
            )

        raise TypeError(f"Unknown item type in pipeline: {type(item)}: {item}")

    @staticmethod
    def _parse_storage_request(item) -> StorageNode | None:
        """Parse a storage request dict into a StorageNode.

        Storage request formats:
        - {"store": "key"}
        - {"store_debug": "key"}
        - {"storage_debug_mode": True/False}
        - {"from_store": "key"}
        """
        if not isinstance(item, dict):
            return None

        if len(item) != 1:
            return None

        key, value = list(item.items())[0]

        if key == "store":
            if not isinstance(value, str):  # pragma: no cover
                raise TypeError("'store' value must be a string")
            return StorageNode(operation="store", key=value)

        elif key == "store_debug":
            if not isinstance(value, str):  # pragma: no cover
                raise TypeError("'store_debug' value must be a string")
            return StorageNode(operation="store_debug", key=value)

        elif key == "storage_debug_mode":
            if not isinstance(value, bool):  # pragma: no cover
                raise TypeError("'storage_debug_mode' value must be bool")
            return StorageNode(operation="toggle_debug", debug_value=value)

        elif key == "from_store":
            if not isinstance(value, str):  # pragma: no cover
                raise TypeError("'from_store' value must be string")
            return StorageNode(operation="load", key=value)

        return None

    @staticmethod
    def _parse_conversion_request(item) -> ConversionNode | None:
        """Parse a conversion request string into a ConversionNode.

        Conversion request formats:
        - "to_native"
        - "from_native"
        """
        if not isinstance(item, str):
            return None

        if item == "to_native":
            return ConversionNode(operation="to_native")
        elif item == "from_native":
            return ConversionNode(operation="from_native")

        return None


def build_ir(
    data,
    *,
    name: str | None = None,
    **kwargs,
) -> SequenceNode:  # pragma: no cover
    """Convenience function to build IR from pipeline data.

    Args:
        data: Pipeline data (transformers, dicts, etc.)
        name: Optional pipeline name
        **kwargs: Additional configuration options

    Returns:
        Root SequenceNode of the built IR.
    """
    builder = IRBuilder(data, name=name, **kwargs)
    return builder.build()
