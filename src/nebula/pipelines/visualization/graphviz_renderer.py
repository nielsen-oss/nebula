"""Graphviz renderer for pipeline visualization.

Creates visual DAG diagrams of pipeline IR using Graphviz.
This replaces the old _DAG and _graphviz modules with a cleaner
implementation that operates on the IR directly.

Dependencies:
- graphviz: pip install graphviz pyyaml
- System graphviz: apt-get install graphviz (or equivalent)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Any

import yaml

from nebula.auxiliaries import split_string_in_chunks
from nebula.pipelines.pipe_aux import get_transformer_name

try:
    from graphviz import Digraph

    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    Digraph = None

if TYPE_CHECKING:
    from ..ir.nodes import (
        PipelineNode, SequenceNode, TransformerNode, FunctionNode,
        StorageNode, ForkNode, MergeNode, InputNode, OutputNode,
    )

__all__ = ["HAS_GRAPHVIZ", "GraphvizRenderer", "render_pipeline"]

_KWS_FONT_NAME: str = "courier"
_KWS_FONT_SIZE: float = 11
_KWS_KEY_FONT_COLOR: str = "purple"  # Only the keys, not the values

# Style constants
_FONT_STYLE = {
    "fontname": "helvetica, verdana",
    "fontsize": "12",
}

_STYLES = {
    'input': {
        'shape': 'ellipse',
        'color': 'blue',
    },
    'output': {
        'shape': 'ellipse',
        'color': 'blue',
    },
    'transformer': {
        'shape': 'rectangle',
        'color': 'black',
        'style': 'rounded',
    },
    'function': {
        'shape': 'rectangle',
        # 'color': 'purple',
        'color': 'black',
        'style': 'rounded',
    },
    'storage_store': {
        'shape': 'cylinder',
        'color': 'red',
    },
    'storage_store_debug': {
        'shape': 'cylinder',
        'color': 'orange',
    },
    'storage_load': {
        'shape': 'cylinder',
        'color': 'blue',
    },
    'storage_toggle': {
        'shape': 'cylinder',
        'color': 'green',
    },
    'fork_split': {
        'shape': 'house',
        'color': 'black',
    },
    'fork_branch': {
        'shape': 'octagon',
        'color': 'black',
    },
    'fork_apply_to_rows': {
        'shape': 'rectangle',
        'color': 'grey',
        'style': 'rounded',
    },
    'merge': {
        'shape': 'rectangle',
        'color': '#067d17',
        'style': 'rounded',
    },
    'branch_name': {
        'shape': 'plaintext',
        'color': 'black',
    },
    'otherwise': {
        'shape': 'plaintext',
        'color': 'black',
    },
}


def __msg_as_yaml(o):
    value_yaml = yaml.safe_dump(o)
    splits = [i.rstrip() for i in value_yaml.split("\n") if i.strip()]
    ret = '  <br ALIGN="LEFT"/>  '
    ret += '  <br ALIGN="LEFT"/>  '.join(splits)
    ret += '<br ALIGN="LEFT"/>'
    return ret


def __single_line(v) -> bool:
    # True if is string or is not an iterable. False otherwise.
    return (not isinstance(v, Iterable)) or isinstance(v, str)


def __get_kws_row_html(a: str, b) -> str:
    """Create the single row of keyword."""
    s = f'<FONT COLOR="{_KWS_KEY_FONT_COLOR}">'
    s += f"{a}: </FONT>{b}"

    if isinstance(b, str):
        if "ALIGN" not in b:
            s += '<br ALIGN="LEFT"/>'
    else:
        s += '<br ALIGN="LEFT"/>'
    return s


def __get_kws_html(data: list[tuple[str, Any]]) -> str:
    """Create the block: keywords without the tile and the final <>."""
    s = f'<FONT POINT-SIZE="{_KWS_FONT_SIZE}" FACE="{_KWS_FONT_NAME}">'

    for k, v in data:
        s += __get_kws_row_html(k, v)

    s += "</FONT>"
    return s


def _transformer_params_to_yaml_format(full_msg) -> str:
    params_split = [i.split("=", 1) for i in full_msg]
    new_params = []
    key: str
    value: str
    for key, value in params_split:
        v_loaded = yaml.safe_load(value)
        # Do not create a new line if the value is a scalar / string.
        if __single_line(v_loaded):
            new_params.append((key, v_loaded))
            continue
        value_formatted = __msg_as_yaml(v_loaded)
        new_params.append((key, value_formatted))

    return __get_kws_html(new_params)


def _function_params_to_yaml_format(args, kwargs) -> str:
    params_split = []
    if args:
        params_split.append(("args", str(args)))
    if kwargs:
        params_split.append(("kwargs", str(kwargs)))
    new_params = []
    key: str
    value: str
    for key, value in params_split:
        v_loaded = yaml.safe_load(value)
        # Do not create a new line if the value is a scalar / string.
        if __single_line(v_loaded):
            new_params.append((key, v_loaded))
            continue
        value_formatted = __msg_as_yaml(v_loaded)
        new_params.append((key, value_formatted))

    return __get_kws_html(new_params)


# def _generic_flat_params_to_yaml_format(params: dict[str, Any]) -> str:
#     new_params = []
#     key: str
#     value: str
#     for key, value in params.items():
#         v_loaded = yaml.safe_load(value)
#         # Do not create a new line if the value is a scalar / string.
#         if __single_line(v_loaded):
#             new_params.append((key, v_loaded))
#             continue
#         value_formatted = __msg_as_yaml(v_loaded)
#         new_params.append((key, value_formatted))
#
#     return __get_kws_html(new_params)


def _generic_flat_params_to_yaml_format(params: dict[str, Any]) -> str:
    ret = []
    for k, v in params.items():
        parsed = str(v) + '<br ALIGN="LEFT"/>'
        ret.append((k, parsed))
    return __get_kws_html(ret)


class GraphvizRenderer:
    """Renders pipeline IR as Graphviz diagram.
    
    Example:
        renderer = GraphvizRenderer(pipeline._ir)
        dot = renderer.render(add_params=True)
        dot.render('pipeline', format='png')
    """

    def __init__(
            self,
            ir: "SequenceNode",
            *,
            max_param_length: int = -1,  # -1 = no limit
    ):
        """Initialize the renderer.
        
        Args:
            ir: The pipeline IR root node.
            max_param_length: Max length for parameter strings (-1 = no limit).
        """
        if not HAS_GRAPHVIZ:
            raise ImportError(
                "graphviz package not installed. "
                "Install with: pip install graphviz"
            )

        self.ir = ir
        self.max_param_length = max_param_length
        self._node_counter = 0
        self._node_ids: dict[str, str] = {}  # IR node id -> graphviz node name

    def render(
            self,
            *,
            add_params: bool = False,
            add_description: bool = False,
    ) -> "Digraph":
        """Render the pipeline IR as a Graphviz diagram.
        
        Args:
            add_params: If True, include transformer parameters.
            add_description: If True, include transformer descriptions.
        
        Returns:
            Graphviz Digraph object (can be rendered to various formats).
        """
        self._node_counter = 0
        self._node_ids = {}

        dot = Digraph()
        dot.attr('node', **_FONT_STYLE)

        # Build the graph
        self._render_node(self.ir, dot, add_params, add_description)

        return dot

    def _get_gv_node_name(self, ir_node_id: str) -> str:
        """Get or create graphviz node name for IR node ID."""
        if ir_node_id not in self._node_ids:
            self._node_ids[ir_node_id] = str(self._node_counter)
            self._node_counter += 1
        return self._node_ids[ir_node_id]

    def _render_node(
            self,
            node: "PipelineNode",
            dot: "Digraph",
            add_params: bool,
            add_description: bool,
            parent_gv_name: str | None = None,
    ) -> str | None:
        """Render a node and return its graphviz node name."""
        from ..ir.nodes import (
            SequenceNode, TransformerNode, FunctionNode,
            StorageNode, ForkNode, MergeNode, InputNode, OutputNode)

        if isinstance(node, SequenceNode):
            return self._render_sequence(node, dot, add_params, add_description, parent_gv_name)
        elif isinstance(node, TransformerNode):
            return self._render_transformer(node, dot, add_params, add_description, parent_gv_name)
        elif isinstance(node, FunctionNode):
            return self._render_function(node, dot, add_params, add_description, parent_gv_name)
        elif isinstance(node, StorageNode):
            return self._render_storage(node, dot, parent_gv_name)
        elif isinstance(node, ForkNode):
            return self._render_fork(node, dot, add_params, add_description, parent_gv_name)
        elif isinstance(node, MergeNode):
            return self._render_merge(node, dot, add_params)
        elif isinstance(node, InputNode):
            return self._render_input(node, dot, parent_gv_name)
        elif isinstance(node, OutputNode):
            return self._render_output(node, dot, parent_gv_name)

        return None

    def _add_node(
            self,
            dot: "Digraph",
            node: "PipelineNode",
            label: str,
            style_key: str,
            parent_gv_name: str | None = None,
    ) -> str:
        """Add a node to the graph and optionally connect to parent."""
        gv_name = self._get_gv_node_name(node.id)
        style = _STYLES.get(style_key, {})

        dot.node(gv_name, label=label, **style)

        if parent_gv_name is not None:
            dot.edge(parent_gv_name, gv_name)

        return gv_name

    def _render_sequence(
            self,
            node: "SequenceNode",
            dot: "Digraph",
            add_params: bool,
            add_description: bool,
            parent_gv_name: str | None,
    ) -> str | None:
        """Render a sequence - just renders children in order."""
        last_gv_name = parent_gv_name

        for step in node.steps:
            last_gv_name = self._render_node(
                step, dot, add_params, add_description, last_gv_name
            )

        return last_gv_name

    def _render_transformer(
            self,
            node: "TransformerNode",
            dot: "Digraph",
            add_params: bool,
            add_description: bool,
            parent_gv_name: str | None,
    ) -> str:
        """Render a transformer node."""
        name = node.transformer_name

        params = None
        if add_params:
            params = get_transformer_name(
                node.transformer, add_params=True, as_list=True, max_len=-1
            )
            if params:
                params = _transformer_params_to_yaml_format(params)

        if add_params or add_description:
            label = self._build_html_label(
                name,
                params=params,
                description=node.description if add_description else None,
            )
        else:
            label = name

        return self._add_node(dot, node, label, 'transformer', parent_gv_name)

    def _render_function(
            self,
            node: "FunctionNode",
            dot: "Digraph",
            add_params: bool,
            add_description: bool,
            parent_gv_name: str | None,
    ) -> str:
        """Render a function node."""
        name = node.func_name
        bold: bool = add_params or add_description
        if add_params and (node.args or node.kwargs):
            params = _function_params_to_yaml_format(node.args, node.kwargs)
            label = self._build_html_label(name, params=params, description=node.description)
        else:
            label = f"<<B>{name}</B>>" if bold else name
        return self._add_node(dot, node, label, 'function', parent_gv_name)

    def _render_storage(
            self,
            node: "StorageNode",
            dot: "Digraph",
            parent_gv_name: str | None,
    ) -> str:
        """Render a storage node."""
        label = node.display_message

        # Determine style based on operation
        if node.operation == 'store':
            style_key = 'storage_store'
        elif node.operation == 'store_debug':
            style_key = 'storage_store_debug'
        elif node.operation == 'load':
            style_key = 'storage_load'
        else:
            style_key = 'storage_toggle'

        return self._add_node(dot, node, label, style_key, parent_gv_name)

    def _render_fork(
            self,
            node: "ForkNode",
            dot: "Digraph",
            add_params: bool,
            add_description: bool,
            parent_gv_name: str | None,
    ) -> str:
        """Render a fork node with its branches."""
        # Determine style based on fork type
        style_key = f'fork_{node.fork_type}'

        # Create label
        if node.fork_type == 'split':
            func_name = getattr(node.split_function, '__name__', 'split')
            label = f"Split: {func_name}"
        elif node.fork_type == 'branch':
            storage = node.config.get('storage')
            if storage:
                label = f"Branch from storage '{storage}'"
            else:
                label = "Branch"
        elif node.fork_type == 'apply_to_rows':
            params = {k: v for k, v in node.config.items() if v}
            params = _generic_flat_params_to_yaml_format(params)
            label = self._build_html_label("Apply to rows", params=params)
        else:
            label = f"Fork: {node.fork_type}"

        fork_gv_name = self._add_node(dot, node, label, style_key, parent_gv_name)

        # Render branches
        branch_end_names = []
        for branch_name, branch_steps in node.branches.items():
            # Add branch name node
            branch_label_gv = self._get_gv_node_name(f"{node.id}_branch_{branch_name}")
            dot.node(branch_label_gv, label=f'[{branch_name.title()}]', **_STYLES['branch_name'])
            dot.edge(fork_gv_name, branch_label_gv)

            # Render branch steps
            last_gv_name = branch_label_gv
            for step in branch_steps:
                last_gv_name = self._render_node(
                    step, dot, add_params, add_description, last_gv_name
                )

            if last_gv_name:
                branch_end_names.append(last_gv_name)

        # Render otherwise if present
        if node.otherwise:
            otherwise_label_gv = self._get_gv_node_name(f"{node.id}_otherwise")
            dot.node(otherwise_label_gv, label='[Otherwise]', **_STYLES['otherwise'])
            dot.edge(fork_gv_name, otherwise_label_gv)

            last_gv_name = otherwise_label_gv
            for step in node.otherwise:
                last_gv_name = self._render_node(
                    step, dot, add_params, add_description, last_gv_name
                )

            if last_gv_name:
                branch_end_names.append(last_gv_name)

        elif node.fork_type in ('branch', 'apply_to_rows'):
            # No otherwise branch - add direct fork-to-merge edge
            # This shows the "pass-through" path for the original dataframe
            branch_end_names.append(fork_gv_name)

        # Store branch ends in metadata for merge node connection
        node.metadata['_branch_end_names'] = branch_end_names

        # Return fork node name - merge will connect from branch ends
        return fork_gv_name

    def _render_merge(
            self,
            node: "MergeNode",
            dot: "Digraph",
            add_params: bool,
    ) -> str:
        """Render a merge node."""
        if node.merge_type == 'append':
            label = "Append DFs"
        elif node.merge_type == 'join':
            label = "Join"
        elif node.merge_type == 'dead-end':
            label = "Dead End"
        else:
            label = f"Merge: {node.merge_type}"

        if add_params:
            params = {k: v for k, v in node.config.items() if v}
            if params:
                params = _generic_flat_params_to_yaml_format(params)
                label = self._build_html_label(label, params=params)
            else:
                label = f"<<B>{label}</B>>"

        merge_gv_name = self._add_node(dot, node, label, 'merge', None)

        # Connect from branch ends if available
        # Look for fork node in parent chain
        if node.parent:
            fork_node = self._find_sibling_fork(node)
            if fork_node:
                branch_ends = fork_node.metadata.get('_branch_end_names', [])
                for branch_end in branch_ends:
                    dot.edge(branch_end, merge_gv_name)

        return merge_gv_name

    @staticmethod
    def _find_sibling_fork(node: "PipelineNode") -> "ForkNode | None":
        """Find the ForkNode sibling that this MergeNode closes."""
        from ..ir.nodes import ForkNode, SequenceNode

        parent = node.parent
        if not isinstance(parent, SequenceNode):
            return None

        # Find this node's index in parent's steps
        try:
            my_index = parent.steps.index(node)
        except ValueError:
            return None

        # Look backwards for the most recent ForkNode
        for i in range(my_index - 1, -1, -1):
            if isinstance(parent.steps[i], ForkNode):
                return parent.steps[i]

        return None

    def _render_input(
            self,
            node: "InputNode",
            dot: "Digraph",
            parent_gv_name: str | None,
    ) -> str:
        """Render an input node."""
        return self._add_node(dot, node, node.name, 'input', parent_gv_name)

    def _render_output(
            self,
            node: "OutputNode",
            dot: "Digraph",
            parent_gv_name: str | None,
    ) -> str:
        """Render an output node."""
        return self._add_node(dot, node, node.name, 'output', parent_gv_name)

    @staticmethod
    def _find_parent_fork(node: "PipelineNode") -> "ForkNode | None":
        """Find the nearest ForkNode ancestor."""
        from ..ir.nodes import ForkNode

        current = node.parent
        while current:
            if isinstance(current, ForkNode):
                return current
            current = current.parent
        return None

    @staticmethod
    def _get_description(desc: str) -> str:
        s = '<FONT POINT-SIZE="10">'
        chunks = split_string_in_chunks(desc, 40)
        msg = '<br ALIGN="LEFT"/>'.join(chunks) + '<br ALIGN="LEFT"/>'
        return f"{s}<I>{msg}</I></FONT>"

    def _build_html_label(
            self,
            title: str,
            params: str | None = None,
            description: str | None = None,
    ) -> str:
        """Build an HTML label for graphviz node."""
        parts = [f"<B>{title}</B>"]

        if description:
            parts.append(f"<br/>{self._get_description(description)}")
        else:
            if params:
                parts.append("<br/>")

        if params:
            parts.append(params)

        return "<" + "".join(parts) + ">"


def render_pipeline(
        ir: "SequenceNode",
        *,
        add_params: bool = False,
        add_description: bool = False,
) -> "Digraph":
    """Convenience function to render a pipeline IR.

    Args:
        ir: The pipeline IR root node.
        add_params: If True, include transformer parameters.
        add_description: If True, include transformer descriptions.

    Returns:
        Graphviz Digraph object.
    """
    renderer = GraphvizRenderer(ir)
    return renderer.render(add_params=add_params, add_description=add_description)
