"""Functions to check the user input."""

from typing import Any, Dict, Optional, Set, Union

__all__ = [
    "assert_branch_inputs",
    "assert_apply_to_rows_inputs",
    "ensure_no_branch_or_apply_to_rows_otherwise",
    "ensure_no_branch_or_apply_to_rows_in_split_pipeline",
    "validate_skip_perform",
]

_APPLY_TO_ROWS_KEYS: Dict[str, Set[str]] = {
    "mandatory": {"input_col", "operator"},
    "optional": {"value", "comparison_column", "dead-end", "skip_if_empty"},
}

_BRANCH_KEYS: Dict[str, Set[str]] = {
    "mandatory": {"end"},
    "optional": {"storage", "on", "how", "broadcast", "skip", "perform"},
}

_BRANCH_END_VALUES = {
    "join": {"mandatory": {"on", "how"}, "optional": {"broadcast", "skip", "perform"}},
    "dead-end": {"mandatory": set(), "optional": {"skip", "perform"}},
    "append": {"mandatory": set(), "optional": {"skip", "perform"}},
}


def _assert_is_dict(name: str, o):
    if not isinstance(o, dict):
        raise TypeError(f"'{name}' must be <dict>, found {type(o)}")


def _check_keys(name: str, o: Union[set, dict], cmp: Dict[str, Set[str]]):
    """Checks for necessary and optional keys."""
    mandatory: Set[str] = cmp["mandatory"]
    optional: Set[str] = cmp["optional"]

    all_keys = mandatory.union(optional)
    keys = set(o)
    if not keys.issubset(all_keys):
        excess = keys - all_keys
        msg = f"'{name}' keys must be a subset of {all_keys}. "
        msg += f"Unknown key(s): {excess}"
        raise KeyError(msg)

    if not mandatory.issubset(keys):
        miss = mandatory - keys
        msg = f"'{name}' must contain the key(s) {mandatory}. Missing: {miss}"
        raise KeyError(msg)


def assert_apply_to_rows_inputs(o: Dict[str, Union[str, bool, None]]) -> None:
    """Check the validity of an 'apply_to_rows' configuration dictionary.

    Args:
        o: Apply to rows configuration.

    Raises:
        TypeError: If the pipeline configuration is not a dictionary.
        KeyError: If the provided required keys are not a subset of:
            {"input_col", "operator", "value", "comparison_column"}.
        ValueError: If both 'value' and 'comparison_column' are provided,
                    or if 'input_col' == 'comparison_column',
                    or if 'skip_if_empty' is not a boolean.
    """
    _assert_is_dict("apply_to_rows", o)
    _check_keys("apply_to_rows", o, _APPLY_TO_ROWS_KEYS)

    value = o.get("value")
    comparison_column = o.get("comparison_column")
    if (value is not None) and (comparison_column is not None):
        msg = "Only one among 'value' and 'comparison_column' "
        msg += "can be set for 'apply_to_rows'."
        raise ValueError(msg)

    input_col = o["input_col"]
    if input_col == comparison_column:
        msg = "'input_col' and 'comparison_column' cannot have the same value."
        raise ValueError(msg)

    skip_if_empty = o.get("skip_if_empty", False)
    if skip_if_empty not in [True, False]:
        raise ValueError("'skip_if_empty' must be either True or False.")


def assert_branch_inputs(o: dict) -> None:
    """Check the validity of a 'branch' configuration dictionary.

    Args:
        o (dict):
            Branch configuration.

    Raises:
        TypeError: If the pipeline configuration is not a dictionary.
        KeyError: If the provided required keys are not a subset of:
            {"storage", "end", "on", "how", "skip", "perform"}.
        ValueError: If the value for 'end' is not in the allowed.
    """
    _assert_is_dict("branch", o)
    _check_keys("branch", o, _BRANCH_KEYS)

    end_value = o["end"]
    allowed_ends = set(_BRANCH_END_VALUES)
    if end_value not in allowed_ends:
        raise ValueError(f"The 'end' value must be in {allowed_ends}")

    keys: Set[str] = set(o).copy() - {"end", "storage"}
    sub_name = f"'branch[{end_value}]'"
    _check_keys(sub_name, keys, _BRANCH_END_VALUES[end_value])
    validate_skip_perform(o.get("skip"), o.get("perform"))


def ensure_no_branch_or_apply_to_rows_otherwise(
    branch: Optional[Dict[str, Union[str, bool]]],
    apply_to_rows: Optional[Dict[str, Union[str, bool]]],
    otherwise: Optional[Dict[str, Any]],
) -> None:
    """Ensure that 'branch', 'apply_to_rows' and 'otherwise' are valid.

    Args:
        branch: Branch configuration.
        apply_to_rows: Apply to rows configuration.
        otherwise: Otherwise configuration.

    Raises:
        AssertionError: If the combination of inputs is invalid.
    """
    if branch and apply_to_rows:
        msg = "The user cannot provide at the same time "
        msg += "'branch' and 'apply_to_rows'"
        raise AssertionError(msg)

    if otherwise:
        if not (branch or apply_to_rows):
            msg = "'Otherwise' can be provided only for "
            msg += "'branch' and 'apply_to_rows'"
            raise AssertionError(msg)

        msg_dead_end = "'Otherwise' cannot be provided in '{}' where "
        msg_dead_end += "'end' == 'dead-end', append another pipeline instead"

        if branch:
            if branch.get("storage"):
                msg = "'Otherwise' cannot be provided in 'branch' where "
                msg += "the 'storage' is set, do the transformation "
                msg += "before branching instead"
                raise AssertionError(msg)
            if branch["end"] == "dead-end":
                raise AssertionError(msg_dead_end.format("branch"))
        if apply_to_rows:
            if apply_to_rows.get("dead-end"):
                raise AssertionError(msg_dead_end.format("apply_to_rows"))


def ensure_no_branch_or_apply_to_rows_in_split_pipeline(
    branch: Optional[Dict[str, Any]], apply_to_rows: Optional[Dict[str, Any]]
) -> None:
    """Ensure that 'branch' and 'apply_to_rows' are not passed in split-pipelines.

    Args:
        branch: Branch configuration.
        apply_to_rows: Apply to rows configuration.

    Raises:
        AssertionError: If either 'branch' or 'apply_to_rows' is provided.
    """
    if branch:
        msg = "'branch' cannot be provided for 'split pipelines'."
        raise AssertionError(msg)

    if apply_to_rows:
        msg = "'apply_to_rows' cannot be provided for 'split pipelines'."
        raise AssertionError(msg)


def validate_skip_perform(skip: Optional[bool], perform: Optional[bool]) -> bool:
    """Validates that the skip and perform flags are not set contradictorily.

    Return True if the user requests NOT to perform the operation, False otherwise.
    """
    # if both are explicitly provided, check the consistency
    if isinstance(skip, bool) and isinstance(perform, bool):
        if (bool(skip) + bool(perform)) != 1:
            raise AssertionError("skip and perform cannot be contradictory")

    # If perform is explicitly False, the user wants to skip.
    if perform is False:
        return True

    return bool(skip)
