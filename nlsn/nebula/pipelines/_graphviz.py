"""Pipeline graphviz module.

This module was written in 3 days, must be reworked.

To change length & layout (...)
dot.edge_attr.update(minlen="0")
dot.graph_attr['rankdir'] = 'LR'
"""

from typing import Any, Iterable

import yaml
from graphviz import Digraph

from nlsn.nebula.auxiliaries import split_string_in_chunks
from nlsn.nebula.pipelines.pipe_aux import Node, NodeType
from nlsn.nebula.pipelines.util import get_transformer_name

__all__ = ["create_graph"]

_STORE_SHAPE: str = "cylinder"  # "octagon", "box", "hexagon", "diamond"
_SPLIT_SHAPE: str = "house"
_MERGE_SHAPE: str = "rectangle"  # "invhouse", "invtrapezium", "ellipse", "invtriangle"

_MERGE_COLOR: str = "#067d17"  # "#595959", "#f40953", "#32bbb9" "purple"
_INPUT_OUTPUT_DF_COLOR: str = "blue"

_MERGE_BLOCK_STYLE: dict[str, str] = {
    "shape": _MERGE_SHAPE,
    "color": _MERGE_COLOR,
    "style": "rounded",
}

_FONT_STYLE: dict[str, str] = {  # keys & values must be <str>
    "fontname": "helvetica, verdana",
    "fontsize": "12",
}

_KWS_FONT_NAME: str = "courier"
_KWS_FONT_SIZE: float = 11
_KWS_KEY_FONT_COLOR: str = "purple"  # Only the keys, not the values

_STYLES_KEYS: set = {"shape", "color", "style", "fillcolor"}


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


def __get_block_html(title: str, data: list[tuple[str, Any]]) -> str:
    """Create the block: title + keywords."""
    s = f"<B>{title}</B><br/>"
    if data:
        s += __get_kws_html(data)
    return f"<{s}>"


def __get_split_merge_msg(x: Node) -> str:
    return __get_block_html("*** Merge Splits ***", sorted(x.data.items()))


def __get_branch_merge_msg(x: Node) -> str:
    end = x.data["end"]
    if end == "dead-end":
        return "Not merge branch"

    if end == "append":
        allow_missing = bool(x.data.get("allow_missing_columns"))
        data = [("Allow missing columns", allow_missing)]
        return __get_block_html("*** Append DFs ***", data)
    elif end == "join":
        on_cols = x.data["on"]
        if __single_line(on_cols):
            on_formatted = on_cols
        else:
            on_formatted = __msg_as_yaml(on_cols)
        data = [("on", on_formatted), ("how", x.data["how"])]
        return __get_block_html("*** Join DFs ***", data)
    else:
        raise ValueError


def __get_a2r_msg(x: Node) -> str:
    a2r: dict = x.data
    keys = ["input_col", "operator", "value", "comparison_column"]
    data = []
    for k in keys:
        if k in a2r:
            data.append((k, a2r[k]))
    return __get_block_html("Apply to rows", data)


def __get_a2r_merge_msg(x: Node) -> str:
    if x.data.apply_to_rows.get("dead-end"):
        return "Not merge branch"

    data = [("Allow missing columns", x.data.allow_missing_cols)]
    if x.data.repartition_output_to_original:
        data += [("Repartition to original", True)]
    if x.data.coalesce_output_to_original:
        data += [("Coalesce to original", True)]
    return __get_block_html("*** Append rows ***", data)


def __get_debug_mode_msg(x: bool) -> str:
    if x:
        return "Activate debug mode"
    return "Deactivate debug mode"


def __get_replace_with_stored_df_msg(x: str) -> str:
    return f'Replace DF with "{x}"'


def __get_transformer_name(x: Node, add_params: bool = False) -> list[str]:
    return get_transformer_name(
        x.data, add_params=add_params, as_list=add_params, max_len=-1
    )


_STYLES: dict[int, dict] = {
    NodeType.TRANSFORMER.value: {
        "shape": "rectangle",
        "color": "black",
        "style": "rounded",
        "msg_func": __get_transformer_name,
    },
    NodeType.STORE.value: {
        "shape": _STORE_SHAPE,
        "color": "red",
    },
    NodeType.STORE_DEBUG.value: {
        "shape": _STORE_SHAPE,
        "color": "orange",
        "msg_func": lambda x: f'"{x.data}"',
    },
    NodeType.STORAGE_DEBUG_MODE.value: {
        "shape": _STORE_SHAPE,
        "color": "green",
        "msg_func": lambda x: __get_debug_mode_msg(x.data),
    },
    NodeType.REPLACE_WITH_STORED_DF.value: {
        "shape": _STORE_SHAPE,
        "color": "blue",
        "msg_func": lambda x: __get_replace_with_stored_df_msg(x.data),
    },
    NodeType.SPLIT_FUNCTION.value: {
        "shape": _SPLIT_SHAPE,
        "color": "black",
        "msg_func": lambda x: x.data.__name__,
    },
    NodeType.SPLIT_NAME.value: {
        "shape": "plaintext",
        "color": "black",
    },
    NodeType.SPLIT_MERGE.value: {
        **_MERGE_BLOCK_STYLE,
        "msg_func": __get_split_merge_msg,
    },
    NodeType.BRANCH_PRIMARY_DF.value: {
        "shape": "octagon",
        "color": "black",
        "msg_func": lambda x: "Branch DF",
    },
    NodeType.BRANCH_SECONDARY_DF.value: {
        "shape": "ellipse",
        "color": _INPUT_OUTPUT_DF_COLOR,
        "msg_func": lambda x: f'Secondary DF\n"{x.data["storage"]}"',
    },
    NodeType.BRANCH_MERGE.value: {
        **_MERGE_BLOCK_STYLE,
        "msg_func": __get_branch_merge_msg,
    },
    NodeType.APPLY_TO_ROWS.value: {
        "shape": "rectangle",
        "color": "grey",
        "style": "rounded",
        "msg_func": __get_a2r_msg,
    },
    NodeType.MERGE_APPLY_TO_ROWS.value: {
        **_MERGE_BLOCK_STYLE,
        "msg_func": __get_a2r_merge_msg,
    },
    NodeType.OTHERWISE.value: {
        "shape": "plaintext",
        "color": "black",
    },
    NodeType.INPUT_DF.value: {
        "shape": "ellipse",
        "color": _INPUT_OUTPUT_DF_COLOR,
        "msg_func": lambda x: x.kws["name"],
    },
    NodeType.OUTPUT_DF.value: {
        "shape": "ellipse",
        "color": _INPUT_OUTPUT_DF_COLOR,
        "msg_func": lambda x: x.kws["name"],
    },
}


def _update_dot(dot: Digraph, el: Node, label: str) -> None:
    node_name = str(el.n)
    enum_value: int = el.t.value

    style = {k: v for k, v in _STYLES[enum_value].items() if k in _STYLES_KEYS}

    try:
        dot.node(node_name, label=label, **style, **_FONT_STYLE)
    except Exception as e:
        msg = "Unable to create the node:\n"
        msg += f"node_name={node_name} ({type(node_name)}),\n"
        msg += f"label={label} ({type(label)}),\n"
        msg += f"font_style={_FONT_STYLE}\n style={style}"
        raise type(e)(f"{e}\n{msg}")

    child: Node
    for child in el.children:
        edge_style = {}
        if el.t not in {
            NodeType.BRANCH_PRIMARY_DF,
            NodeType.BRANCH_SECONDARY_DF,
        }:
            if child.t == NodeType.BRANCH_MERGE and child.data.get("end") == "dead-end":
                edge_style["style"] = "invis"
        if child.t == NodeType.BRANCH_SECONDARY_DF:
            edge_style["style"] = "invis"

        dot.edge(node_name, str(child.n), len="0", **edge_style)


def _get_transformer_params_formatted(full_msg):
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


def _add_transformer_description(el) -> str:
    s = '<FONT POINT-SIZE="11">'
    label = ""
    if hasattr(el.data, "get_description"):
        trf_desc = el.data.get_description()
        if trf_desc:
            if trf_desc.strip() == "":
                return ""
            chunks = split_string_in_chunks(trf_desc, 40)
            trf_desc = '<br ALIGN="LEFT"/>'.join(chunks) + '<br ALIGN="LEFT"/>'
            label = f"{s}<I>{trf_desc}</I></FONT>"

    return label


def create_graph(
        dag: list[Node | dict],
        add_transformer_params: bool = False,
        add_transformer_description: bool = False,
) -> Digraph:
    """Create the graphviz plot from the pipeline dag."""
    dot = Digraph()

    el: Node | dict

    for el in dag:
        if isinstance(el, dict):
            continue

        enum_value: int = el.t.value
        msg_func = _STYLES[enum_value].get("msg_func")

        if msg_func is not None:
            if enum_value == NodeType.TRANSFORMER.value and add_transformer_params:
                full_msg = msg_func(el, True)
                label: str = full_msg[0]
                label = f"<B>{label}</B><br/>"  # <br ALIGN="LEFT"/>'
                if add_transformer_description:
                    label += _add_transformer_description(el)
                if len(full_msg) > 1:
                    # params = "<br/>".join(full_msg[1:])  # old implementation
                    label += _get_transformer_params_formatted(full_msg[1:])

                label = f"<{label}>"

            else:
                label = msg_func(el)
        else:
            label = el.data

        _update_dot(dot, el, label)

    return dot
