"""Pipeline Intermediate Representation (IR) nodes.

The IR is a tree of nodes representing the pipeline structure.
Once built, it can be traversed by different visitors:
- Executor: runs the pipeline
- Printer: outputs to terminal
- GraphvizRenderer: creates visual diagram

Design decisions:
- Nodes are dataclasses for clarity and immutability intent
- Each node has a unique ID based on content hash + position
- Parent references enable restart/checkpoint functionality
- Metadata dict allows extensibility without changing signatures
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Literal, TYPE_CHECKING

from nebula.auxiliaries import truncate_long_string
from nebula.pipelines.util import get_transformer_name

if TYPE_CHECKING:
    from nebula.base import Transformer, LazyWrapper

__all__ = [
    "NodeType",
    "PipelineNode",
    "TransformerNode",
    "FunctionNode",
    "StorageNode",
    "ForkNode",
    "MergeNode",
    "SequenceNode",
    "InputNode",
    "OutputNode",
]


class NodeType(Enum):
    """Enumeration of all node types in the IR."""
    INPUT = auto()
    OUTPUT = auto()
    TRANSFORMER = auto()
    FUNCTION = auto()
    STORAGE = auto()
    FORK = auto()
    MERGE = auto()
    SEQUENCE = auto()


@dataclass
class PipelineNode:
    """Base node in pipeline IR.
    
    Attributes:
        id: Unique, stable identifier (content hash + position).
        node_type: Type discriminator for visitor dispatch.
        position: Path in tree, e.g., (0, 'splits', 'low', 2).
        metadata: Extensible dict for name, description, rendering hints.
        children: Forward edges to child nodes.
        parent: Backward edge to parent (enables restart navigation).
    """
    id: str = ""
    node_type: NodeType = NodeType.SEQUENCE
    position: tuple[int | str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    children: list[PipelineNode] = field(default_factory=list)
    parent: PipelineNode | None = field(default=None, repr=False)

    def add_child(self, child: PipelineNode) -> None:
        """Add a child node and set its parent reference."""
        child.parent = self
        self.children.append(child)

    def walk(self):
        """Depth-first traversal yielding all nodes."""
        yield self
        for child in self.children:
            yield from child.walk()

    def find_by_id(self, node_id: str) -> PipelineNode | None:
        """Find a node by its ID."""
        for node in self.walk():
            if node.id == node_id:
                return node
        return None


@dataclass
class InputNode(PipelineNode):
    """Represents the input DataFrame entry point."""
    name: str = "DF input"

    def __post_init__(self):
        self.node_type = NodeType.INPUT
        if not self.metadata.get("name"):
            self.metadata["name"] = self.name


@dataclass
class OutputNode(PipelineNode):
    """Represents the output DataFrame exit point."""
    name: str = "DF output"

    def __post_init__(self):
        self.node_type = NodeType.OUTPUT
        if not self.metadata.get("name"):
            self.metadata["name"] = self.name


@dataclass
class TransformerNode(PipelineNode):
    """Wraps a Transformer or LazyWrapper instance.
    
    Attributes:
        transformer: The actual Transformer or LazyWrapper instance.
        description: Optional user-provided description.
    """
    transformer: Transformer | LazyWrapper | None = None
    description: str | None = None
    params_for_print: str | None = None

    def __post_init__(self):
        self.node_type = NodeType.TRANSFORMER
        if self.description:
            self.metadata["description"] = self.description

    @property
    def transformer_name(self) -> str:
        """Get the transformer class name."""
        if self.transformer is None:
            return "Unknown"
        # Handle LazyWrapper
        if hasattr(self.transformer, 'trf'):
            return f"(Lazy) {self.transformer.trf.__name__}"
        return self.transformer.__class__.__name__

    def get_params_for_print(self, max_param_length: int):
        if self.params_for_print is None:
            self.params_for_print = get_transformer_name(
                self.transformer, add_params=True, max_len=max_param_length
            )
        return self.params_for_print


@dataclass
class FunctionNode(PipelineNode):
    """Wraps a bare Python function as a transformer.
    
    Supports four call signatures:
    1. func(df) -> df
    2. func(df, *args) -> df
    3. func(df, *args, **kwargs) -> df
    4. func(df, *args, **kwargs, description: str) -> df

    Attributes:
        func: The callable function.
        args: Positional arguments (after df).
        kwargs: Keyword arguments.
    """
    func: Callable | None = None
    args: tuple = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    description: str | None = None
    params_for_print: str | None = None

    def __post_init__(self):
        self.node_type = NodeType.FUNCTION

    @property
    def func_name(self) -> str:
        """Get the function name."""
        if self.func is None:
            return "Unknown"
        return getattr(self.func, "__name__", str(self.func))

    def get_params_for_print(self, max_param_length: int):
        if self.params_for_print is None:
            params_parts = []
            if self.args:
                params_parts.append(f"ARGS={self.args}")
            if self.kwargs:
                params_parts.append(f"KWARGS={self.kwargs}")
            params_str = ", ".join(params_parts)
            self.params_for_print = truncate_long_string(params_str, max_param_length)
        return self.params_for_print


@dataclass
class StorageNode(PipelineNode):
    """Storage operations: store, load, debug toggle.
    
    Operations:
    - 'store': Save df to nebula_storage with key
    - 'store_debug': Save df only if debug mode is active
    - 'load': Replace df with one from storage
    - 'toggle_debug': Enable/disable debug storage mode
    
    Attributes:
        operation: The storage operation type.
        key: Storage key (for store/load operations).
        debug_value: True/False for toggle_debug operation.
    """
    operation: Literal['store', 'store_debug', 'load', 'toggle_debug'] = 'store'
    key: str = ''
    debug_value: bool | None = None

    def __post_init__(self):
        self.node_type = NodeType.STORAGE

    @property
    def display_message(self) -> str:
        """Human-readable description of the operation."""
        if self.operation == 'store':
            return f'Store df with key "{self.key}"'
        elif self.operation == 'store_debug':
            return f'Store df (debug) with key "{self.key}"'
        elif self.operation == 'load':
            return f'Load df from key "{self.key}"'
        elif self.operation == 'toggle_debug':
            action = "Activate" if self.debug_value else "Deactivate"
            return f"{action} storage debug mode"
        return f"Unknown operation: {self.operation}"


@dataclass
class ForkNode(PipelineNode):
    """Represents a point where the pipeline splits into branches.
    
    Fork types:
    - 'split': DataFrame split by user function into named subsets
    - 'branch': Secondary pipeline from primary or stored df
    - 'apply_to_rows': Filter rows, apply transforms, merge back
    
    Attributes:
        fork_type: The type of fork operation.
        config: Fork-specific configuration (condition, storage key, etc.).
        branches: Named branches, each containing a list of nodes.
        otherwise: Optional pipeline for non-matching rows.
        split_function: For 'split' type, the function that splits df.
    """
    fork_type: Literal['split', 'branch', 'apply_to_rows'] = 'split'
    config: dict[str, Any] = field(default_factory=dict)
    branches: dict[str, list[PipelineNode]] = field(default_factory=dict)
    otherwise: list[PipelineNode] | None = None
    split_function: Callable | None = None

    def __post_init__(self):
        self.node_type = NodeType.FORK


@dataclass
class MergeNode(PipelineNode):
    """Merge point after a fork - combines branch results.
    
    Merge types:
    - 'append': Vertical concatenation (union by name)
    - 'join': SQL-style join with on/how parameters
    - 'dead-end': No merge, branch result is discarded
    
    Attributes:
        merge_type: How to combine the branches.
        config: Merge-specific config (join keys, allow_missing_cols, etc.).
    """
    merge_type: Literal['append', 'join', 'dead-end'] = 'append'
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.node_type = NodeType.MERGE

    @property
    def display_message(self) -> str:
        """Human-readable description of the merge."""
        if self.merge_type == 'append':
            return "Append DataFrames"
        elif self.merge_type == 'join':
            how = self.config.get('how', 'inner')
            on = self.config.get('on', [])
            return f"Join DataFrames ({how} on {on})"
        elif self.merge_type == 'dead-end':
            return "Dead end (no merge)"
        return f"Unknown merge: {self.merge_type}"


@dataclass
class SequenceNode(PipelineNode):
    """A linear sequence of nodes.
    
    This simplifies the IR for the common case of sequential transformers.
    The steps are also stored in children for uniform traversal.
    
    Attributes:
        steps: Ordered list of nodes to execute sequentially.
        name: Optional pipeline name for display.
    """
    steps: list[PipelineNode] = field(default_factory=list)
    name: str | None = None

    def __post_init__(self):
        self.node_type = NodeType.SEQUENCE
        # Steps are also children for uniform traversal
        self.children = self.steps
        if self.name:
            self.metadata["name"] = self.name

    def add_step(self, node: PipelineNode) -> None:
        """Add a step to the sequence."""
        node.parent = self
        self.steps.append(node)
        self.children = self.steps  # Keep in sync
