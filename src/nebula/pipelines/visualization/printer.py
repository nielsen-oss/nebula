"""Terminal printer for pipeline visualization.

Prints a text representation of the pipeline IR to the terminal,
showing structure, transformers, and configuration options.

This replaces the old _show_pipeline function with a cleaner
implementation that operates on the IR rather than the raw
pipeline configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..pipe_cfg import PIPE_CFG

if TYPE_CHECKING:  # pragma: no cover
    from ..ir.nodes import (
        ConversionNode,
        ForkNode,
        FunctionNode,
        MergeNode,
        PipelineNode,
        SequenceNode,
        StorageNode,
        TransformerNode,
    )

__all__ = ["PipelinePrinter", "print_pipeline"]

MERGE_KWS = {
    # merging params
    "allow_missing_columns",
    "cast_subsets_to_input_schema",
    "repartition_output_to_original",
    "coalesce_output_to_original",
    # join
    "end",
    "on",
    "how",
    "broadcast",
    "left_on",
    "right_on",
    "suffix",
    # mix
    "storage",
    "skip",
    "perform",
    "skip_if_empty",
}


class PipelinePrinter:
    """Prints pipeline IR to terminal.

    Example:
        printer = PipelinePrinter(pipeline._ir)
        printer.print(add_params=True)
    """

    def __init__(
        self,
        ir: "SequenceNode",
        *,
        max_param_length: int,
        indent_size: int = 4,
    ):
        """Initialize the printer.

        Args:
            ir: The pipeline IR root node.
            indent_size: Number of spaces per indentation level.
            max_param_length: Maximum length for parameter strings.
        """
        self.ir = ir
        self.indent_size = indent_size
        self.max_param_length = max_param_length

    def print(self, *, add_params: bool = False, add_ids: bool = False) -> None:
        """Print the pipeline to terminal.

        Args:
            add_params: If True, include transformer parameters.
            add_ids: If True, include node IDs (useful for debugging).
        """
        lines = self._collect_lines(add_params=add_params, add_ids=add_ids)
        for line in lines:
            print(line)

    def to_string(self, *, add_params: bool = False, add_ids: bool = False) -> str:
        """Get pipeline representation as string.

        Args:
            add_params: If True, include transformer parameters.
            add_ids: If True, include node IDs.

        Returns:
            Multi-line string representation.
        """
        lines = self._collect_lines(add_params=add_params, add_ids=add_ids)
        return "\n".join(lines)

    def _collect_lines(self, *, add_params: bool, add_ids: bool) -> list[str]:
        """Collect all output lines."""
        lines = []
        self._visit_node(self.ir, 0, lines, add_params, add_ids)
        return lines

    def _indent(self, level: int) -> str:
        """Get indentation string for given level."""
        return " " * (level * self.indent_size)

    def _visit_node(
        self,
        node: "PipelineNode",
        level: int,
        lines: list[str],
        add_params: bool,
        add_ids: bool,
    ) -> None:
        """Visit a node and add its representation to lines."""
        from ..ir.nodes import (
            ConversionNode,
            ForkNode,
            FunctionNode,
            InputNode,
            MergeNode,
            OutputNode,
            SequenceNode,
            StorageNode,
            TransformerNode,
        )

        indent = self._indent(level)

        if isinstance(node, SequenceNode):
            self._visit_sequence(node, level, lines, add_params, add_ids)
        elif isinstance(node, TransformerNode):
            self._visit_transformer(node, level, lines, add_params, add_ids)
        elif isinstance(node, FunctionNode):
            self._visit_function(node, level, lines, add_params, add_ids)
        elif isinstance(node, StorageNode):
            self._visit_storage(node, level, lines, add_ids)
        elif isinstance(node, ConversionNode):
            self._visit_conversion(node, level, lines, add_ids)
        elif isinstance(node, ForkNode):
            self._visit_fork(node, level, lines, add_params, add_ids)
        elif isinstance(node, MergeNode):
            self._visit_merge(node, level, lines, add_params, add_ids)
        elif isinstance(node, InputNode):
            lines.append(f"{indent}[Input: {node.name}]")
        elif isinstance(node, OutputNode):
            lines.append(f"{indent}[Output: {node.name}]")
        else:
            lines.append(f"{indent}[Unknown: {type(node).__name__}]")

    def _visit_sequence(
        self,
        node: "SequenceNode",
        level: int,
        lines: list[str],
        add_params: bool,
        add_ids: bool,
    ) -> None:
        """Visit a sequence node."""
        indent = self._indent(level)

        # Add header
        msg_step_count = self._get_msg_step_count(node, zero_to_null=True)
        header = f"*** {node.name if node.name else 'Pipeline'} ***"
        if msg_step_count:
            header += f" ({msg_step_count})"
        if add_ids:
            header += f" [{node.id}]"
        lines.append(f"{indent}{header}")

        # Visit children (skip input/output for cleaner display at root)
        for step in node.steps:
            from ..ir.nodes import InputNode, OutputNode

            if isinstance(step, (InputNode, OutputNode)) and level == 0:
                continue  # Skip at root level
            self._visit_node(step, level, lines, add_params, add_ids)

    def _visit_transformer(
        self,
        node: "TransformerNode",
        level: int,
        lines: list[str],
        add_params: bool,
        add_ids: bool,
    ) -> None:
        """Visit a transformer node."""
        indent = self._indent(level)
        name = node.transformer_name
        line = f"{indent} - {name}"

        if add_params:
            params = node.get_params_for_print(self.max_param_length)
            if params:
                line += f" -> PARAMS: {params}"

        if add_ids:
            line += f" [{node.id}]"

        lines.append(line)

        # Add description if present
        if node.description:
            lines.append(f"{indent}     Description: {node.description}")

    def _visit_function(
        self,
        node: "FunctionNode",
        level: int,
        lines: list[str],
        add_params: bool,
        add_ids: bool,
    ) -> None:
        """Visit a function node."""
        indent = self._indent(level)
        name = node.func_name

        line = f"{indent} - {name}"

        if add_params and (node.args or node.kwargs):
            params = node.get_params_for_print(self.max_param_length)
            line += f" -> {params}"

        if add_ids:
            line += f" [{node.id}]"

        lines.append(line)
        # Add description if present
        if node.description:
            lines.append(f"{indent}     Description: {node.description}")

    def _visit_storage(
        self,
        node: "StorageNode",
        level: int,
        lines: list[str],
        add_ids: bool,
    ) -> None:
        """Visit a storage node."""
        indent = self._indent(level)
        line = f"{indent}   --> {node.display_message}"
        if add_ids:
            line += f" [{node.id}]"
        lines.append(line)

    def _visit_conversion(
        self,
        node: "ConversionNode",
        level: int,
        lines: list[str],
        add_ids: bool,
    ) -> None:
        """Visit a conversion node."""
        indent = self._indent(level)
        line = f"{indent}   --> {node.display_message}"
        if add_ids:
            line += f" [{node.id}]"
        lines.append(line)

    def _visit_fork(  # noqa: PLR0912
        self,
        node: "ForkNode",
        level: int,
        lines: list[str],
        add_params: bool,
        add_ids: bool,
    ) -> None:
        """Visit a fork node."""
        indent = self._indent(level)

        # Fork header
        if node.fork_type == "split":
            header = "------ SPLIT ------"
            if node.split_function:
                func_name = getattr(node.split_function, "__name__", "unknown")
                header += f" (function: {func_name})"
        elif node.fork_type == "branch":
            storage = node.config.get("storage")
            if storage:
                header = f"------ BRANCH (from storage: {storage}) ------"
            else:
                header = "------ BRANCH (from the primary DF) ------"
        elif node.fork_type == "apply_to_rows":
            col = node.config.get("input_col")
            op = node.config.get("operator")
            val = node.config.get("value")
            header = f"------ APPLY TO ROWS ({col} {op} {val}) ------"
        else:
            header = f"------ FORK ({node.fork_type}) ------"

        lines.append(f"{indent}{header}")

        # Show config if add_params
        if add_params:
            no_show = {"storage"}
            for key, value in sorted(node.config.items()):
                if value and (key not in no_show.union(MERGE_KWS)):
                    lines.append(f"{indent}  - {key}: {value}")

        if node.fork_type == "split":
            # Visit splits
            for split_name, branch_steps in node.branches.items():
                msg_step_count = self._get_msg_step_count(branch_steps)
                lines.append(f"{indent}**SPLIT <<< {split_name} >>> ({msg_step_count}):")
                for step in branch_steps:
                    self._visit_node(step, level + 1, lines, add_params, add_ids)

        else:  # branch | apply_to_rows
            flow_name = "Branch" if node.fork_type == "branch" else "Apply To rows"
            branch_steps = list(node.branches.values())[0]
            msg_step_count = self._get_msg_step_count(branch_steps)
            lines.append(f"{indent}>> {flow_name} ({msg_step_count}):")
            for step in branch_steps:
                self._visit_node(step, level + 1, lines, add_params, add_ids)

            # Visit otherwise
            if node.otherwise:
                msg_step_count = self._get_msg_step_count(node.otherwise)
                lines.append(f"{indent}>> Otherwise ({msg_step_count})")
                for step in node.otherwise:
                    self._visit_node(step, level + 1, lines, add_params, add_ids)

    def _visit_merge(
        self,
        node: "MergeNode",
        level: int,
        lines: list[str],
        add_params: bool,
        add_ids: bool,
    ) -> None:
        """Visit a merge node."""
        indent = self._indent(level)

        if node.merge_type == "append":
            line = f"{indent}<<< Append DFs >>>"
        elif node.merge_type == "join":
            line = f"{indent}<<< Join DFs >>>"
        elif node.merge_type == "dead-end":
            line = f"{indent}<<< Dead End (no merge) >>>"
        else:
            line = f"{indent}<<< Merge ({node.merge_type}) >>>"

        if add_ids:
            line += f" [{node.id}]"

        lines.append(line)

        if add_params:
            for key, value in node.config.items():
                if value and key in MERGE_KWS:
                    lines.append(f"{indent}  - {key}: {value}")

    def _get_msg_step_count(self, obj, zero_to_null: bool = False) -> str:
        from ..ir.nodes import SequenceNode

        if isinstance(obj, SequenceNode):
            n_nodes = self._count_transformations(obj)
        else:
            n_nodes = sum(self._count_transformations(i) for i in obj)
        if zero_to_null and (n_nodes == 0):
            return ""
        plural = "s" if n_nodes != 1 else ""
        return f"{n_nodes} transformation{plural}"

    @staticmethod
    def _count_transformations(node: "PipelineNode") -> int:
        """Count transformers and functions in a node tree."""
        from ..ir.nodes import ForkNode, FunctionNode, TransformerNode

        count = 0
        for n in node.walk():
            if isinstance(n, (TransformerNode, FunctionNode)):
                count += 1
            elif isinstance(n, ForkNode):
                # Count in branches
                for branch_steps in n.branches.values():
                    for step in branch_steps:
                        count += PipelinePrinter._count_transformations(step)
                # Count in otherwise
                if n.otherwise:
                    for step in n.otherwise:
                        count += PipelinePrinter._count_transformations(step)
        return count


def print_pipeline(
    ir: "SequenceNode",
    *,
    add_params: bool = False,
    indent_size: int = 4,
) -> None:  # pragma: no cover
    """Convenience function to print a pipeline IR.

    Args:
        ir: The pipeline IR root node.
        add_params: If True, include transformer parameters.
        indent_size: Number of spaces per indentation level.
    """
    max_param_length = PIPE_CFG["max_param_length"]
    printer = PipelinePrinter(ir, max_param_length=max_param_length, indent_size=indent_size)
    printer.print(add_params=add_params)
