"""Pipeline Loop Processing Module.

This module provides functionality to process nested pipeline configurations
with loop structures.
It allows for dynamic expansion of pipeline components based on loop
parameters, with support for nested loops and parameter precedence rules.

Key Features:
-------------
- Recursive processing of nested pipeline structures
- Parameter substitution using <<parameter_name>> syntax
- Support for both linear and product-based loop expansions
- Nested loop handling with clear precedence rules
- Preservation of non-loop pipeline components

Loop Structure:
--------------
A loop must be defined under a "loop" key with the following structure:
    {
        "loop": {
            "values": {
                "param1": [value1, value2, ...],
                "param2": [value1, value2, ...],
            },
            "mode": "linear"|"product",  # Optional, defaults to "linear"
            ... # Additional pipeline configuration
        }
    }

Parameter Precedence:
-------------------
When nested loops use the same parameter names, the innermost loop takes precedence.
This allows for hierarchical parameter overriding while maintaining outer loop parameters
for non-overridden values.

Example:
--------
Here's an example demonstrating nested loops and parameter precedence:

d = {
    "loop": {
        "values": {
            "x1": [1, 2],  # This will fill only the transformer TR_1
            "x2": [3, 4]  # This will fill both TR_1 and TR_2
        },
        "pipeline": [
            {
                "transformer": "TR_1",
                "params": {"a": "<<x1>>", "b": "<<x2>>"},
            },
            {
                "pipeline": [
                    {
                        "loop": {
                            "values": {
                                "x1": [10, 20],  # Takes precedence over outer "x1"
                                "x3": [30, 40]
                            },
                            "transformer": "TR_2",
                            # "b" is filled by the outermost parameters
                            "params": {"a": "<<x1>>", "b": "<<x2>>", "c": "<<x3>>"},
                        }
                    }
                ]
            }
        ]
    }
}

In this example:
1. The outer loop defines parameters x1 and x2
2. TR_1 uses both outer loop parameters
3. The inner loop redefines x1 and introduces x3
4. TR_2 uses:
   - x1 from the inner loop (precedence)
   - x2 from the outer loop (not overridden)
   - x3 from the inner loop

Functions:
---------
process_pipeline(pipe: dict) -> dict
    Main entry point for processing a pipeline configuration.

expand_loops(d: dict) -> dict
    Recursively processes nested loops in a dictionary.

substitute_params(d: dict, params: dict) -> dict
    Substitutes parameters in a dictionary with their values.

process_loop(loop_dict: dict) -> List[dict]
    Processes a single loop configuration.

prepare_loop_params(d: dict) -> dict
    Validates and prepares loop parameters.

Notes:
-----
- Parameters must be enclosed in double angle brackets (<<param_name>>)
- Loop values must be lists of equal length when using linear generation
- Product generation type creates all possible combinations of parameters
- Original input structures are not modified; new copies are created
- Original types defined in the loop dictionary like int / float / are
    preserved when the placeholder is the full parameter, like "<<x>>"
    otherwise, if inserted in a string like "value_<<x>>" it is cast to string
"""

from copy import deepcopy
from itertools import product
from typing import Any

__all__ = ["expand_loops"]


def validate_loop_params(d: dict) -> None:
    """Validate parameters for a loop without modifying the input.

    Args:
        d (dict): A dictionary containing loop parameters

    Raises:
        TypeError: If values or mode are of the wrong type
        ValueError: If values have different lengths or invalid mode
        KeyError: If required keys are missing
    """
    if "values" not in d:
        raise KeyError("The 'values' key is missing from the input dictionary.")

    values = d["values"]
    if not isinstance(values, dict):
        raise TypeError("The 'values' must be a dictionary.")
    if not values:
        raise ValueError("The 'values' dictionary is empty.")

    for k, v in values.items():
        if not isinstance(v, list):
            raise TypeError(f"Value for '{k}' must be list, got '{type(v)}'.")

    mode = d.get("mode")
    if mode is not None:
        if not isinstance(mode, str):
            raise TypeError("The 'mode' must be a string.")
        if mode not in {"linear", "product"}:
            raise ValueError("The 'mode' must be 'linear' or 'product'.")

    if (mode == "linear") or not mode:
        lens = {len(i) for i in values.values()}
        if len(lens) != 1:
            raise ValueError(f"'values' must have the same length. Found: {lens}")


def convert_product_to_linear(d_orig: dict) -> dict:
    """Convert a product-type loop to linear format if needed.

    Args:
        d_orig (dict): A validated dictionary containing loop parameters

    Returns:
        dict: Original dict if linear, new dict with cross-product if product
    """
    mode: str | None = d_orig.get("mode")
    if mode is None:
        return d_orig

    d_new = deepcopy(d_orig)
    del d_new["mode"]

    if mode == "linear":
        return d_new

    values = d_new["values"]
    d_new["values"] = dict(zip(values.keys(), zip(*product(*values.values()))))
    return d_new


def prepare_loop_params(d: dict) -> dict:
    """Validate and prepare loop parameters."""
    validate_loop_params(d)
    return convert_product_to_linear(d)


def _replace(s, param_name: str, param_value):
    if not isinstance(s, str):  # int / float / datetime | ...
        return s
    placeholder = f"<<{param_name}>>"
    if s == placeholder:
        return param_value  # don't change the type
    return s.replace(placeholder, str(param_value))


def _replace_multiple_params(s: str, params: dict):
    # 's' enters as a string, but could be replaced by something else
    for param_name, param_value in params.items():
        s = _replace(s, param_name, param_value)
    return s


def _substitute_params(o, params: dict):
    if isinstance(o, str):
        return _replace_multiple_params(o, params)
    elif isinstance(o, list):
        ret = []
        for v in o:
            ret.append(_substitute_params(v, params))
        return ret
    elif isinstance(o, dict):
        ret = {}
        for k, v in o.items():
            new_key = _replace_multiple_params(k, params) if isinstance(k, str) else k
            ret[new_key] = _substitute_params(v, params)
        return ret
    else:
        return o


def substitute_params(d: dict, params: dict) -> dict:
    """Recursively substitute parameters enclosed in <<>> with their values."""
    return _substitute_params(d, params)


def process_loop(loop_dict: dict) -> list[dict]:
    """Process a single loop and return list of expanded dictionaries."""
    loop_dict = prepare_loop_params(loop_dict)
    values: dict = loop_dict["values"]

    # Get the first value's length since we know they're
    # all the same after prepare_loop_params
    n_iterations = len(next(iter(values.values())))

    # Create parameter combinations for each iteration
    ret = []
    for i in range(n_iterations):
        params = {k: v[i] for k, v in values.items()}

        # Create a copy of loop_dict without the loop-specific keys
        loop_content = {k: v for k, v in loop_dict.items() if k not in {"loop", "values"}}

        # Substitute parameters in the copy
        expanded_dict: dict = substitute_params(loop_content, params)
        ret.append(expanded_dict)

    return ret


def _expand_loops(o) -> tuple[Any, bool]:
    """Recursively process all loops, from innermost to outermost."""
    # No need for deepcopy - we'll build a new dict as we go
    if isinstance(o, dict):
        if o.get("skip") or (o.get("perform") is False):
            if "loop" in o:
                return {}, False
            return o, False

        ret = {}
        for k, value in o.items():
            ret[k], _ = _expand_loops(value)

        if "loop" in o:
            loop_content = o["loop"]
            # Process any nested loops in the loop content first
            loop_content, _ = _expand_loops(loop_content)
            # Now process this loop
            expanded = process_loop(loop_content)
            return expanded, True

    elif isinstance(o, list):
        ret = []
        for value in o:
            res, is_expanded = _expand_loops(value)
            if is_expanded and isinstance(res, list):
                ret.extend(res)
            else:
                ret.append(res)
    else:
        ret = o
    return ret, False


def expand_loops(o: dict) -> dict:
    """Process the entire pipeline dictionary."""
    # We only need to copy the top-level structure
    pipe = o["pipeline"]
    is_split = isinstance(pipe, dict)
    ret = {"pipeline": {} if is_split else []}

    # Copy all non-pipeline keys
    for k, v in o.items():
        if k != "pipeline":
            ret[k] = v

    # Process the top-level pipeline
    if is_split:
        for k, item in pipe.items():
            processed, _ = _expand_loops(item)
            ret["pipeline"][k] = processed
    else:
        for item in pipe:
            processed, is_expanded = _expand_loops(item)
            if processed:  # if loop is skipped it return an empty dict
                if is_expanded and isinstance(processed, list):
                    ret["pipeline"].extend(processed)
                else:
                    ret["pipeline"].append(processed)

    return ret
