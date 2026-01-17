"""Pipeline execution hooks for monitoring and extensibility.

Hooks allow users to inject custom behavior at key points during
pipeline execution without modifying the executor. Common use cases:
- Custom logging
- Metrics collection
- Error alerting (e.g., Slack notifications)
- Checkpointing

The default NoOpHooks implementation does nothing, ensuring zero
overhead when hooks aren't needed.

Example:
    class SlackAlertHooks(NoOpHooks):
        def on_error(self, node_id, error, context):
            send_slack_message(f"Pipeline failed at {node_id}: {error}")

    pipeline.run(df, hooks=SlackAlertHooks())
"""

from __future__ import annotations

from typing import Protocol

from ..ir.nodes import (
    ForkNode,
    FunctionNode,
    PipelineNode,
    StorageNode,
    TransformerNode,
)

__all__ = ["PipelineHooks", "NoOpHooks", "LoggingHooks"]

KWS_PRINT = {
    "splits_no_merge",
    "splits_skip_if_empty",
    "cast_subsets_to_input_schema",
    "repartition_output_to_original",
    "coalesce_output_to_original",
}


class PipelineHooks(Protocol):
    """Protocol defining the hooks interface.

    Implement this protocol to create custom hooks. You only need
    to implement the methods you care about - use NoOpHooks as a
    base class for convenience.
    """

    def on_pipeline_start(self, root_node: "PipelineNode", context: dict) -> None:
        """Called when pipeline execution begins.

        Args:
            root_node: The root node of the pipeline IR.
            context: Dictionary with 'df' and other metadata.
        """
        ...

    def on_pipeline_end(self, root_node: "PipelineNode", duration_ms: float, context: dict) -> None:
        """Called when pipeline execution completes successfully.

        Args:
            root_node: The root node of the pipeline IR.
            duration_ms: Total execution time in milliseconds.
            context: Dictionary with 'df' (result) and other metadata.
        """
        ...

    def on_node_start(self, node: "PipelineNode", context: dict) -> None:
        """Called before a node begins execution.

        Args:
            node: The node about to execute.
            context: Dictionary with 'df' and other metadata.
        """
        ...

    def on_node_end(self, node: "PipelineNode", duration_ms: float, context: dict) -> None:
        """Called after a node completes successfully.

        Args:
            node: The node that just completed.
            duration_ms: Node execution time in milliseconds.
            context: Dictionary with 'df' (result) and other metadata.
        """
        ...

    def on_error(self, node: "PipelineNode", error: Exception, context: dict) -> None:
        """Called when a node raises an exception.

        Args:
            node: The node that raised the exception.
            error: The exception that was raised.
            context: Dictionary with 'df' (input to failed node) and metadata.

        Note: This is called before the exception propagates. To suppress
        the exception, you would need custom executor logic.
        """
        ...

    def on_checkpoint(self, node: "PipelineNode", context: dict) -> None:
        """Called when a checkpoint is saved.

        Args:
            node: The node after which checkpoint was saved.
            context: Dictionary with checkpoint details.
        """
        ...

    def on_skip(self, node: "PipelineNode", reason: str, context: dict) -> None:
        """Called when a node is skipped (e.g., during resume).

        Args:
            node: The node being skipped.
            reason: Why the node was skipped.
            context: Dictionary with metadata.
        """
        ...


class NoOpHooks:
    """Default hooks implementation that does nothing.

    Use this as a base class when you only want to implement
    specific hooks:

        class MyHooks(NoOpHooks):
            def on_error(self, node, error, context):
                # Custom error handling
                send_alert(error)
    """

    def on_pipeline_start(self, root_node: "PipelineNode", context: dict) -> None:
        pass

    def on_pipeline_end(self, root_node: "PipelineNode", duration_ms: float, context: dict) -> None:
        pass

    def on_node_start(self, node: "PipelineNode", context: dict) -> None:
        pass

    def on_node_end(self, node: "PipelineNode", duration_ms: float, context: dict) -> None:
        pass

    def on_error(self, node: "PipelineNode", error: Exception, context: dict) -> None:
        pass

    def on_checkpoint(self, node: "PipelineNode", context: dict) -> None:
        pass

    def on_skip(self, node: "PipelineNode", reason: str, context: dict) -> None:
        pass


class LoggingHooks(NoOpHooks):
    """Hooks implementation that logs to the nebula logger.

    This provides similar output to the current pipeline execution,
    making migration smoother.
    """

    def __init__(self, *, max_param_length: int, show_params: bool, logger=None):
        """Initialize with optional custom logger.

        Args:
            max_param_length:
                Max string length of the parameters converted to string.
            show_params:
                Whether printing parameters in the terminal.
            logger:
                Logger instance. Defaults to nebula.logger.logger.
        """
        if logger is None:
            from nebula.logger import logger as nebula_logger

            self.logger = nebula_logger
        else:  # pragma: no cover
            self.logger = logger

        self.show_params: bool = show_params
        self.max_param_length: int = max_param_length

    def on_pipeline_start(  # noqa: D102
        self, root_node: "PipelineNode", context: dict
    ) -> None:
        name = root_node.metadata.get("name")
        msg = "Starting pipeline" + (f" '{name}'" if name else "")
        self.logger.info(msg)

    def on_pipeline_end(  # noqa: D102
        self, root_node: "PipelineNode", duration_ms: float, context: dict
    ) -> None:
        name = root_node.metadata.get("name")
        pre = "Pipeline" + (f" '{name}'" if name else "")
        duration_sec = duration_ms / 1000
        self.logger.info(f"{pre} completed in {duration_sec:.1f}s")

    def on_node_start(self, node: "PipelineNode", context: dict) -> None:
        if isinstance(node, TransformerNode):
            name = f"'{node.transformer_name}'"
            if self.show_params:
                params = node.get_params_for_print(self.max_param_length)
                if params:
                    name += f" PARAMS: {params}"
            self.logger.info(f"Running {name} ...")
        elif isinstance(node, FunctionNode):
            name = f"'{node.func_name}'"
            if self.show_params and (node.args or node.kwargs):
                name += ": " + node.get_params_for_print(self.max_param_length)
            self.logger.info(f"Running {name} ...")
        elif isinstance(node, StorageNode):
            self.logger.info(f"   --> {node.display_message}")
        elif isinstance(node, ForkNode):
            cfg_log = {k: v for k, v in node.config.items() if v and (k in KWS_PRINT)}
            msg = f"Entering {node.fork_type}"
            if cfg_log:
                msg += f": {cfg_log}"
            self.logger.info(msg)

    def on_node_end(  # noqa: D102
        self, node: "PipelineNode", duration_ms: float, context: dict
    ) -> None:
        if isinstance(node, (TransformerNode, FunctionNode)):
            name = node.transformer_name if isinstance(node, TransformerNode) else node.func_name
            duration_sec = duration_ms / 1000
            self.logger.info(f"Completed '{name}' in {duration_sec:.1f}s")

    def on_error(  # noqa: D102
        self, node: "PipelineNode", error: Exception, context: dict
    ) -> None:
        self.logger.error(f"Error at node {node.id}: {error}")

    def on_skip(  # noqa: D102
        self, node: "PipelineNode", reason: str, context: dict
    ) -> None:
        self.logger.info(f"Skipping node {node.id}: {reason}")
