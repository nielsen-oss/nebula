"""Execution context - mutable state during pipeline execution.

The ExecutionContext carries the DataFrame and execution state through
the pipeline. It's passed to each node during execution and can be
used for:
- Tracking completed nodes (for checkpoint/restart)
- Storing intermediate results
- Passing metadata between nodes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Any

__all__ = ["ExecutionContext"]


@dataclass
class ExecutionContext:
    """Mutable state carried through pipeline execution.

    Attributes:
        df: The current DataFrame being transformed.
        completed_nodes: Set of node IDs that have been executed.
        resume_from: If set, skip nodes until this ID is reached.
        should_skip: Internal flag for resume logic.
        start_time_ns: Pipeline start time in nanoseconds.
        node_start_times: Dict tracking individual node start times.
        checkpoint_storage: If set, save checkpoints to this storage key prefix.
        metadata: Extensible dict for custom data.
        fail_cache: Cache of DataFrames for error recovery.
    """

    df: Any = None
    completed_nodes: set[str] = field(default_factory=set)
    resume_from: str | None = None
    should_skip: bool = False
    start_time_ns: int = field(default_factory=perf_counter_ns)
    node_start_times: dict[str, int] = field(default_factory=dict)
    checkpoint_storage: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    fail_cache: dict[str, Any] = field(default_factory=dict)

    def elapsed_ms(self) -> float:
        """Get elapsed time since pipeline start in milliseconds."""
        return (perf_counter_ns() - self.start_time_ns) / 1_000_000

    def start_node(self, node_id: str) -> None:
        """Mark a node as starting execution."""
        self.node_start_times[node_id] = perf_counter_ns()

    def end_node(self, node_id: str) -> float:
        """Mark a node as completed and return its duration in ms."""
        start = self.node_start_times.pop(node_id, self.start_time_ns)
        duration_ms = (perf_counter_ns() - start) / 1_000_000
        self.completed_nodes.add(node_id)
        return duration_ms

    def should_skip_node(self, node_id: str) -> tuple[bool, str]:
        """Check if a node should be skipped during resume.

        Returns:
            Tuple of (should_skip: bool, reason: str)
        """
        if not self.resume_from:
            return False, ""

        if self.should_skip:
            if node_id == self.resume_from:
                # Found the resume point - stop skipping
                self.should_skip = False
                return False, ""
            return True, f"Resuming from {self.resume_from}"

        return False, ""

    def cache_for_failure(self, key: str, df: Any) -> None:
        """Cache a DataFrame for potential error recovery.

        This is used to store DataFrames before risky operations
        (like transformers or merges) so they can be recovered
        and stored in nebula_storage if an error occurs.
        """
        self.fail_cache[key] = df

    def clear_fail_cache(self) -> None:
        """Clear the failure cache after successful operation."""
        self.fail_cache.clear()

    def clone_with_df(self, new_df: Any) -> "ExecutionContext":
        """Create a shallow copy with a different DataFrame.

        Useful for parallel branch execution where each branch
        needs its own context but shares checkpoint tracking.
        """
        return ExecutionContext(
            df=new_df,
            completed_nodes=self.completed_nodes,  # Shared reference
            resume_from=self.resume_from,
            should_skip=self.should_skip,
            start_time_ns=self.start_time_ns,
            node_start_times=self.node_start_times,  # Shared reference
            checkpoint_storage=self.checkpoint_storage,
            metadata=self.metadata,  # Shared reference
            fail_cache=self.fail_cache,  # Shared reference
        )
