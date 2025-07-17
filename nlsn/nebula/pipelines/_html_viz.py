"""HTML rendering of the pipeline.

! Experimental !


stringa lunga
"""

# pylint: disable=pointless-statement
# pylint: disable=unused-argument

import html
from pathlib import Path
from typing import Dict, Iterable

import yaml

from nlsn.nebula.auxiliaries import split_string_in_chunks
from nlsn.nebula.base import Transformer
from nlsn.nebula.pipelines.auxiliaries import StoreRequest, is_storage_request
from nlsn.nebula.pipelines.transformer_type_util import is_transformer
from nlsn.nebula.pipelines.util import get_transformer_name

__all__ = ["render_html"]


class _IDCounter:
    """Generate sequential ids with a prefix."""

    def __init__(self, prefix):
        self.prefix = prefix
        self.count = 0

    def get_id(self) -> str:
        self.count += 1
        return f"{self.prefix}-{self.count}"


# _CONTAINER_ID_COUNTER = _IDCounter("blk-container-id")
_STEP_ID_COUNTER = _IDCounter("step-id")


def _stripped_str(o) -> str:
    if isinstance(o, str):
        return o.strip()
    else:
        return ""


_BLOCK_COLORS = {
    "merge_splits": ("#ffe0b3", "#fff5e6"),
    "store": ("#d9dbdb", "#e1e3e3"),
    "store_debug": ("#f2eecb", "#f5edab"),
    "storage_debug_mode_on": ("#cdf7f0", "#b4faee"),
    "storage_debug_mode_off": ("#f9d2fc", "#f3b7f7"),
    "apply_to_rows": ("#bef7c1", "#b5f7b8"),  # green
    "merge_apply_to_rows": ("#ffe0b3", "#fff5e6"),
    "branch": ("#bef7c1", "#b5f7b8"),  # green
    "merge_append_branch": ("#ffe0b3", "#fff5e6"),
    "merge_join_branch": ("#ffe0b3", "#fff5e6"),
}


def __bold_key_value(k: str, v: str, after_colon=" ") -> str:
    return (
        f'<span><span style="font-weight: bold;">{k}</span>:{after_colon}{v}</span><br>'
    )


def _get_coalesce_repartition_to_original(obj) -> str:
    if obj.coalesce_output_to_original:
        return __bold_key_value("Coalesce to original", "True")
    elif obj.repartition_output_to_original:
        return __bold_key_value("Repartition to original", "True")
    return ""


def _add_blank_blk(
    label: str, color: str = "black", bg_color: str = "white", border: str = "white"
) -> str:
    """Add a non-collapsible w/o info blank block."""
    stl = f'style="color: {color}; background-color: {bg_color}; border-color: {border}; padding: 3px"'
    # label = html.escape(label)
    ret = f"""
    <div class="blk-item">
        <div class="blk-step" {stl}>
            <label>{label}</label>
        </div>
    </div>
    """
    return ret


def _add_generic_blk(
    name: str, content: str, info: str, *, bg_color_main=None, bg_color_content=None
) -> str:
    idn = _STEP_ID_COUNTER.get_id()

    # if bg_color_main:
    #     bg_color_main = f'style="background-color: {bg_color_main};"'
    # else:
    #     bg_color_main = ""
    # if bg_color_content:
    #     bg_color_content = f'style="background-color: {bg_color_content};"'
    # else:
    #     bg_color_content = ""

    if info:
        info = html.escape(info)
        span = f"""<span class="blk-step-doc-link">i <span>{info}</span></span>"""
    else:
        span = ""

    # name = html.escape(name)
    main_el = f"<div>{span}{name}</div>"  # keep these divs
    ret = '<div class="blk-item">'

    if content:
        # content = html.escape(content)
        ret += f"""
            <div class="blk-step blk-toggleable">
                <input class="blk-toggleable__control blk-hidden--visually" id="blk-{idn}" type="checkbox">
                <label for="blk-{idn}" class="blk-toggleable__label blk-toggleable__label-arrow">
                    {main_el}
                </label>
                <div class="blk-toggleable__content">
                    <pre>{content}</pre>
                </div>
            </div>
        """
    else:
        ret += f"""
        <div class="blk-step" style="padding: 6px;">{main_el}</div>
        """
    ret += "</div>"
    return ret


def __msg_as_yaml(o):
    value_yaml = yaml.safe_dump(o)
    splits = [i.rstrip() for i in value_yaml.split("\n") if i.strip()]
    # ret = "  <br>  "
    ret = "  <br>  ".join(splits)
    # ret += "<br>"
    return ret


def __single_line(v) -> bool:
    # True if is string or is not an iterable. False otherwise.
    return (not isinstance(v, Iterable)) or isinstance(v, str)


def _add_transformer_blk(obj: Transformer) -> str:
    trf_name, *trf_params = get_transformer_name(
        obj, add_params=True, max_len=-1, as_list=True
    )
    if hasattr(obj, "get_description"):
        desc = obj.get_description()
    else:
        desc = ""

    try:
        doc = obj.__init__.__doc__.split("\n")[0].strip().rstrip(".")
    except (AttributeError, IndexError):
        doc = ""

    if desc:
        formatted_desc = "<br>".join(split_string_in_chunks(desc, limit=30))
        content = __bold_key_value("DESCRIPTION", formatted_desc)
    else:
        content = ""

    params_split = [i.split("=", 1) for i in trf_params]
    key: str
    value: str
    for key, value in params_split:
        content += __key_bold_value_yaml(key, value)
    return _add_generic_blk(trf_name, content, doc)


def __key_bold_value_yaml(k, v) -> str:
    v_loaded = yaml.safe_load(v)
    if __single_line(v_loaded):
        return __bold_key_value(k, v_loaded)
    v_formatted = __msg_as_yaml(v_loaded)
    return __bold_key_value(k, v_formatted, after_colon="<br>  ")


def _add_storage_blk(obj, storage_request) -> str:
    content, info = "", ""
    if storage_request == StoreRequest.STORE_DF:
        title = "Store"
        content = __bold_key_value("KEY", obj["store"])
        info = "Store the current dataframe in the storage"
        key_color = "store"

    elif storage_request == StoreRequest.STORE_DF_DEBUG:
        title = "Store (debug mode)"
        content = __bold_key_value("KEY", obj["store_debug"])
        info = "Store the current dataframe in the storage if the debug mode is enabled"
        key_color = "store_debug"

    elif storage_request == StoreRequest.ACTIVATE_DEBUG:
        key_color = "storage_debug_mode_on"
        title = "Activate storage debug mode"

    elif storage_request == StoreRequest.DEACTIVATE_DEBUG:
        key_color = "storage_debug_mode_off"
        title = "Deactivate storage debug mode"

    else:  # pragma: no cover
        raise ValueError("Unknown Enum in _StoreRequest")

    bg_color_main, bg_color_content = _BLOCK_COLORS[key_color]
    out = _add_generic_blk(
        title,
        content,
        info,
        bg_color_main=bg_color_main,
        bg_color_content=bg_color_content,
    )

    return out


def _add_pipe_blk(name: str, desc: str) -> str:
    name = html.escape(name)
    # desc = html.escape(desc)
    idn = _STEP_ID_COUNTER.get_id()
    ret = f"""
    <div class="blk-label-container">
        <div class="blk-label blk-toggleable">
            <input class="blk-toggleable__control blk-hidden--visually" id="blk-{idn}" type="checkbox">
            <label for="blk-{idn}" class="blk-toggleable__label blk-toggleable__label-arrow">
                {name}
            </label>
            <div class="blk-toggleable__content ">
                <pre>{desc}</pre>
            </div>
        </div>
    </div>
    """
    return ret


def _add_split_pipe(out, obj):
    # Add the split function block on top
    split_func_name: str = obj.split_function.__name__
    split_func_name = html.escape(split_func_name)
    out += '<div class="blk-serial">'
    out += _add_blank_blk(split_func_name, border="lightgrey")
    out += "</div>"

    dead_end = obj.splits_no_merge

    out += '<div class="blk-parallel">'
    for split_name, el in obj.splits.items():
        out += '<div class="blk-parallel-item">'
        out += '<div class="blk-item">'
        out += '<div class="blk-serial">'
        out += _add_blank_blk(split_name, border="white")
        out = _create_html(out, el)
        if split_name in dead_end:
            out += _add_blank_blk(
                "Not merged split", color="red", border="grey", bg_color="white"
            )
        out += "</div></div></div>"

    # Add the merge block
    out += "</div>"
    out += '<div class="blk-serial" style="padding-top:10px;">'
    merge_block_main = '<span style="font-weight: bold;">Append splits</span><br>'

    merge_details = __bold_key_value(
        "Allow missing columns", f"{bool(obj.allow_missing_cols)}"
    )
    merge_details += __bold_key_value(
        "Cast each split to input schema", f"{bool(obj.cast_subset_to_input_schema)}"
    )
    merge_details += _get_coalesce_repartition_to_original(obj)
    bg_color_main, bg_color_content = _BLOCK_COLORS["merge_splits"]
    out += _add_generic_blk(
        merge_block_main,
        merge_details,
        "",
        bg_color_main=bg_color_main,
        bg_color_content=bg_color_content,
    )
    out += "</div>"
    return out


def _get_a2r_top_block(a2r: Dict[str, str]) -> str:
    s = '<div class="blk-serial" style="padding-top:10px;">'
    header = '<span style="font-weight: bold;">Apply to rows</span><br>'

    content = ""
    keys = ["input_col", "operator", "value", "comparison_column"]
    for k in keys:
        if k in a2r:
            content += __bold_key_value(k, f"{a2r[k]}")

    bg_color_main, bg_color_content = _BLOCK_COLORS["apply_to_rows"]
    s += _add_generic_blk(
        header,
        content,
        "",
        bg_color_main=bg_color_main,
        bg_color_content=bg_color_content,
    )
    s += "</div>"
    return s


def _add_a2r_pipe(out, obj):
    # Add the split function block on top
    a2r_cond: str = _get_a2r_top_block(obj.apply_to_rows)
    # a2r_cond = html.escape(a2r_cond)
    out += '<div class="blk-serial">'
    out += _add_blank_blk(a2r_cond, border="lightgrey")
    out += "</div>"

    pipes = [["Branched flow", obj.stages]]
    other = ["Main flow"]
    if obj.otherwise:
        other.append(obj.otherwise)
    else:
        other.append([])
    pipes.append(other)

    is_dead_end = obj.apply_to_rows.get("dead-end")

    out += '<div class="blk-parallel">'
    for split_name, el in pipes:
        out += '<div class="blk-parallel-item">'
        out += '<div class="blk-item">'
        out += '<div class="blk-serial">'
        out += _add_blank_blk(split_name, border="white")
        out = _create_html(out, el)

        if is_dead_end and (split_name == "Branched flow"):
            out += _add_blank_blk(
                "Not appended rows", color="red", border="grey", bg_color="white"
            )

        out += "</div></div></div>"

    # Add the merge block
    out += "</div>"
    out += '<div class="blk-serial" style="padding-top:10px;">'
    merge_block_main = '<span style="font-weight: bold;">Append rows</span><br>'
    merge_details = __bold_key_value(
        "Allow missing columns", f"{bool(obj.allow_missing_cols)}"
    )
    merge_details += _get_coalesce_repartition_to_original(obj)
    bg_color_main, bg_color_content = _BLOCK_COLORS["merge_apply_to_rows"]
    out += _add_generic_blk(
        merge_block_main,
        merge_details,
        "",
        bg_color_main=bg_color_main,
        bg_color_content=bg_color_content,
    )
    out += "</div>"
    return out


def _get_branch_top_block(branch: Dict[str, str]) -> str:
    storage = branch.get("storage")
    if storage:
        title = f"Branch from a stored dataframe: {storage}"
    else:
        title = "Branch the main flow"

    s = '<div class="blk-serial" style="padding-top:10px;">'
    header = f'<span style="font-weight: bold;">{title}</span><br>'

    bg_color_main, bg_color_content = _BLOCK_COLORS["branch"]
    s += _add_generic_blk(
        header, "", "", bg_color_main=bg_color_main, bg_color_content=bg_color_content
    )
    s += "</div>"
    return s


def _add_branch_pipe(out, obj):
    # Add the split function block on top
    branch_cond: str = _get_branch_top_block(obj.branch)
    # branch_cond = html.escape(branch_cond)
    out += '<div class="blk-serial">'
    out += _add_blank_blk(branch_cond, border="lightgrey")
    out += "</div>"

    storage = obj.branch.get("storage")
    if storage:
        branched_flow = f"Source: {storage}"
    else:
        branched_flow = "Branched flow"

    pipes = [[branched_flow, obj.stages]]
    other = ["Main flow"]
    if obj.otherwise:
        other.append(obj.otherwise)
    else:
        other.append([])
    pipes.append(other)

    end_type = obj.branch["end"]
    is_dead_end = end_type == "dead-end"

    out += '<div class="blk-parallel">'
    for split_name, el in pipes:
        out += '<div class="blk-parallel-item">'
        out += '<div class="blk-item">'
        out += '<div class="blk-serial">'
        out += _add_blank_blk(split_name, border="white")
        out = _create_html(out, el)

        if is_dead_end and (split_name != "Main flow"):
            out += _add_blank_blk(
                "Not merged branch", color="red", border="grey", bg_color="white"
            )

        out += "</div></div></div>"

    out += "</div>"
    if is_dead_end:
        return out

    # Add the merge block
    out += '<div class="blk-serial" style="padding-top:10px;">'

    if end_type == "append":
        merge_block_main = (
            '<span style="font-weight: bold;">Append dataframes</span><br>'
        )
        merge_details = __bold_key_value(
            "Allow missing columns", f"{bool(obj.allow_missing_cols)}"
        )
        merge_details += _get_coalesce_repartition_to_original(obj)
        bg_color_main, bg_color_content = _BLOCK_COLORS["merge_append_branch"]
    else:
        merge_block_main = '<span style="font-weight: bold;">Join dataframes</span><br>'
        join_on = obj.branch["on"]
        merge_details = __bold_key_value("How", obj.branch["how"])
        merge_details += __key_bold_value_yaml(
            "On", f"{join_on}"
        )  # __bold_key_value("On", join_on)
        bg_color_main, bg_color_content = _BLOCK_COLORS["merge_join_branch"]

    out += _add_generic_blk(
        merge_block_main,
        merge_details,
        "",
        bg_color_main=bg_color_main,
        bg_color_content=bg_color_content,
    )
    out += "</div>"
    return out


def _create_html(out, obj, df_input_name=None):
    if hasattr(obj, "branch"):
        name = _stripped_str(obj.name)
        name = name if name else "Pipeline"
        desc = _stripped_str(obj.description)

        n_trf = f"{obj.get_number_transformers()} transformers"
        if desc:
            desc = "<br>".join(split_string_in_chunks(desc, limit=30)) + f"<br>{n_trf}"
        else:
            desc = n_trf

        out += '<div class="blk-item blk-dashed-wrapped">'
        if obj._sub_pipe_type == "linear":
            out += _add_pipe_blk(name, desc)
            out += '<div class="blk-serial">'
            for el in obj.stages:
                out = _create_html(out, el)

        elif obj._sub_pipe_type in {"split", "a2r", "branch"}:
            out += '<div class="blk-item blk-dashed-wrapped">'
            out += _add_pipe_blk(name, desc)

            if obj._sub_pipe_type == "split":
                out = _add_split_pipe(out, obj)

            elif obj._sub_pipe_type == "a2r":
                out = _add_a2r_pipe(out, obj)

            elif obj._sub_pipe_type == "branch":
                out = _add_branch_pipe(out, obj)

            else:
                raise RuntimeError  # pragma: no cover

        else:
            raise RuntimeError  # pragma: no cover

        out += "</div></div>"

    else:
        # Check whether 'obj' is a storage request
        _storage_request: StoreRequest = is_storage_request(obj)

        if is_transformer(obj):
            out += _add_transformer_blk(obj)
            return out

        # If 'obj' is an iterable, recursively parse it
        elif isinstance(obj, (list, tuple)):
            for el in obj:
                out = _create_html(out, el)

        elif _storage_request.value > 0:
            out += _add_storage_blk(obj, _storage_request)

        else:  # pragma: no cover
            raise RuntimeError

    return out


def render_html(obj, df_input_name, df_output_name) -> str:
    """HTML pipeline rendering."""
    # idn = _CONTAINER_ID_COUNTER.get_id()
    css = Path(__file__).with_suffix(".css").read_text("utf-8")
    out = f"<style>{css}</style>\n\n"

    _len_css = len(out)

    out += """
    <div id="dag-container" class="blk-top-container">
    <div class="blk-container">
    """

    out = _create_html(out, obj, df_input_name)

    # df_output_name_strip = _stripped_str(df_input_name)
    # if df_output_name_strip:
    #     out += _add_blank_blk(df_output_name_strip)

    out += "</div></div>"
    return out
