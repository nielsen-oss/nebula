"""Node ID generation for pipeline IR.

IDs are designed to be:
1. Stable: Same transformer at same position = same ID across runs
2. Unique: No collisions within a pipeline
3. Human-readable: Can identify what/where a node is
4. Suitable for checkpointing: Can resume from a specific node

ID Format: "{content_hash}@{path}"
- content_hash: Short hash of node content (transformer name, params, etc.)
- path: Position in tree (e.g., "0.splits.low.2")

Examples:
- "SelectColumns_a1b2@0"           → First transformer at root
- "Filter_c3d4@0.splits.low.2"     → Third item in 'low' split
- "store:df_x@1"                   → Storage op at position 1
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .nodes import PipelineNode

__all__ = ["generate_node_id", "generate_content_hash", "position_to_path"]


def generate_content_hash(content: str, length: int = 6) -> str:
    """Generate a short hash from content string.

    Args:
        content: String to hash (e.g., transformer name + params).
        length: Length of hash to return (default 6 chars).

    Returns:
        Short hex hash string.
    """
    full_hash = hashlib.sha256(content.encode()).hexdigest()
    return full_hash[:length]


def position_to_path(position: tuple[int | str, ...]) -> str:
    """Convert position tuple to dot-separated path string.

    Args:
        position: Tuple like (0, 'splits', 'low', 2)

    Returns:
        String like "0.splits.low.2"
    """
    if not position:
        return "root"
    return ".".join(str(p) for p in position)


def _get_transformer_content(transformer) -> str:
    """Extract content string from a transformer for hashing."""
    # Handle LazyWrapper
    if hasattr(transformer, 'trf'):
        name = transformer.trf.__name__
        params = transformer.kwargs
    elif hasattr(transformer, '__class__'):
        name = transformer.__class__.__name__
        # Try to get init params if available
        params = getattr(transformer, '_transformer_init_params', {})
    else:  # pragma: no cover
        name = str(transformer)
        params = {}

    # Sort params for consistent hashing
    params_str = str(sorted(params.items())) if params else ""
    return f"{name}:{params_str}"


def _get_function_content(func, args: tuple, kwargs: dict) -> str:
    """Extract content string from a function for hashing."""
    name = getattr(func, "__name__", str(func))
    args_str = str(args) if args else ""
    kwargs_str = str(sorted(kwargs.items())) if kwargs else ""
    return f"{name}:{args_str}:{kwargs_str}"


def _get_storage_content(operation: str, key: str, debug_value: bool | None) -> str:
    """Extract content string from storage operation for hashing."""
    if operation == 'toggle_debug':
        return f"storage:toggle:{debug_value}"
    return f"storage:{operation}:{key}"


def _get_fork_content(fork_type: str, config: dict) -> str:
    """Extract content string from fork for hashing."""
    config_str = str(sorted(config.items())) if config else ""
    return f"fork:{fork_type}:{config_str}"


def _get_merge_content(merge_type: str, config: dict) -> str:
    """Extract content string from merge for hashing."""
    config_str = str(sorted(config.items())) if config else ""
    return f"merge:{merge_type}:{config_str}"


def generate_node_id(
        node_type: str,
        position: tuple[int | str, ...],
        *,
        transformer: Any = None,
        func: Any = None,
        args: tuple = (),
        kwargs: dict = None,
        operation: str = "",
        key: str = "",
        debug_value: bool | None = None,
        fork_type: str = "",
        merge_type: str = "",
        config: dict = None,
        name: str = "",
) -> str:
    """Generate a unique, stable ID for a pipeline node.

    The ID combines a content hash with the position path, making it
    both stable (same content = same hash) and unique (different positions
    even with same content).

    Args:
        node_type: Type of node ('transformer', 'function', 'storage', etc.)
        position: Position tuple in the tree
        transformer: For transformer nodes
        func: For function nodes
        args: For function nodes
        kwargs: For function nodes
        operation: For storage nodes
        key: For storage nodes
        debug_value: For storage toggle nodes
        fork_type: For fork nodes
        merge_type: For merge nodes
        config: For fork/merge nodes
        name: Optional name override

    Returns:
        Node ID string like "SelectColumns_a1b2@0.splits.low"
    """
    kwargs = kwargs or {}
    config = config or {}
    path = position_to_path(position)

    # Generate content-based hash
    if node_type == 'transformer':
        content = _get_transformer_content(transformer)
        prefix = content.split(":")[0]  # Just the name
    elif node_type == 'function':
        content = _get_function_content(func, args, kwargs)
        prefix = f"fn_{getattr(func, '__name__', 'unknown')}"
    elif node_type == 'storage':
        content = _get_storage_content(operation, key, debug_value)
        prefix = f"{operation}"
        if key:
            prefix += f":{key}"
    elif node_type == 'fork':
        content = _get_fork_content(fork_type, config)
        prefix = f"fork_{fork_type}"
    elif node_type == 'merge':
        content = _get_merge_content(merge_type, config)
        prefix = f"merge_{merge_type}"
    elif node_type == 'sequence':
        content = f"seq:{name}" if name else "seq"
        prefix = f"seq_{name}" if name else "seq"
    elif node_type == 'input':
        content = f"input:{name}"
        prefix = "input"
    elif node_type == 'output':
        content = f"output:{name}"
        prefix = "output"
    else:  # pragma: no cover
        content = f"unknown:{node_type}"
        prefix = "unknown"

    content_hash = generate_content_hash(content)

    # Clean prefix (remove special chars)
    prefix = prefix.replace(" ", "_").replace(":", "_")

    return f"{prefix}_{content_hash}@{path}"


def assign_ids_to_tree(root: "PipelineNode") -> None:
    """Assign IDs to all nodes in a tree.

    This walks the tree and assigns position-based IDs to each node.
    Should be called after the tree is fully built.

    Args:
        root: Root node of the pipeline tree.
    """
    _assign_ids_recursive(root, ())


def _assign_ids_recursive(
        node: "PipelineNode",
        position: tuple[int | str, ...]
) -> None:
    """Recursively assign IDs based on position."""
    from .nodes import (
        TransformerNode, FunctionNode, StorageNode,
        ForkNode, MergeNode, SequenceNode, InputNode, OutputNode
    )

    node.position = position

    # Generate ID based on node type
    if isinstance(node, TransformerNode):
        node.id = generate_node_id(
            'transformer', position,
            transformer=node.transformer
        )
    elif isinstance(node, FunctionNode):
        node.id = generate_node_id(
            'function', position,
            func=node.func, args=node.args, kwargs=node.kwargs
        )
    elif isinstance(node, StorageNode):
        node.id = generate_node_id(
            'storage', position,
            operation=node.operation, key=node.key, debug_value=node.debug_value
        )
    elif isinstance(node, ForkNode):
        node.id = generate_node_id(
            'fork', position,
            fork_type=node.fork_type, config=node.config
        )
        # Process branches
        for branch_name, branch_nodes in node.branches.items():
            for i, child in enumerate(branch_nodes):
                child_pos = position + ('branches', branch_name, i)
                _assign_ids_recursive(child, child_pos)
        # Process otherwise
        if node.otherwise:
            for i, child in enumerate(node.otherwise):
                child_pos = position + ('otherwise', i)
                _assign_ids_recursive(child, child_pos)
        return  # Don't process children normally for ForkNode

    elif isinstance(node, MergeNode):
        node.id = generate_node_id(
            'merge', position,
            merge_type=node.merge_type, config=node.config
        )
    elif isinstance(node, SequenceNode):
        node.id = generate_node_id(
            'sequence', position,
            name=node.name
        )
    elif isinstance(node, InputNode):
        node.id = generate_node_id('input', position, name=node.name)
    elif isinstance(node, OutputNode):
        node.id = generate_node_id('output', position, name=node.name)
    else:  # pragma: no cover
        node.id = generate_node_id('unknown', position)

    # Process children (for non-fork nodes)
    for i, child in enumerate(node.children):
        child_pos = position + (i,)
        _assign_ids_recursive(child, child_pos)
