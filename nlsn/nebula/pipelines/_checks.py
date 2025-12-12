"""Functions to check the user input."""

from typing import Any

from nlsn.nebula.auxiliaries import validate_keys

__all__ = [
    "assert_branch_inputs",
    "assert_apply_to_rows_inputs",
    "ensure_no_branch_or_apply_to_rows_in_split_pipeline",
    "ensure_no_branch_or_apply_to_rows_otherwise",
    "should_skip_operation",
]


def _assert_is_dict(name: str, o):
    if not isinstance(o, dict):
        raise TypeError(f"'{name}' must be a dict, got {type(o).__name__}")


def assert_apply_to_rows_inputs(o: dict[str, str | bool | None]) -> None:
    """Check the validity of an 'apply_to_rows' configuration dictionary."""
    _assert_is_dict("apply_to_rows", o)

    validate_keys(
        "apply_to_rows",
        o,
        mandatory={"input_col", "operator"},
        optional={"value", "comparison_column", "dead-end", "skip_if_empty"}
    )

    # Rest of validation...
    value = o.get("value")
    comparison_column = o.get("comparison_column")
    if (value is not None) and (comparison_column is not None):
        raise ValueError(
            "Only one of 'value' or 'comparison_column' can be set for 'apply_to_rows'"
        )

    input_col = o["input_col"]
    if input_col == comparison_column:
        raise ValueError("'input_col' and 'comparison_column' cannot have the same value")

    skip_if_empty = o.get("skip_if_empty", False)
    if skip_if_empty not in [True, False]:
        raise ValueError("'skip_if_empty' must be either True or False")


def assert_branch_inputs(o: dict) -> None:
    """Check the validity of a 'branch' configuration dictionary."""
    _assert_is_dict("branch", o)

    validate_keys(
        "branch",
        o,
        mandatory={"end"},
        optional={"storage", "on", "how", "broadcast", "skip", "perform"}
    )

    end_value = o["end"]
    allowed_ends = {"join", "dead-end", "append"}
    if end_value not in allowed_ends:
        raise ValueError(f"'end' must be one of {allowed_ends}, got: {end_value}")

    # Validate end-specific keys
    keys = set(o) - {"end", "storage"}
    if end_value == "join":
        validate_keys(
            f"branch[end='{end_value}']",
            keys,
            mandatory={"on", "how"},
            optional={"broadcast", "skip", "perform"}
        )
    elif end_value in {"dead-end", "append"}:
        validate_keys(
            f"branch[end='{end_value}']",
            keys,
            mandatory=set(),
            optional={"skip", "perform"}
        )

    should_skip_operation(o.get("skip"), o.get("perform"))


def ensure_no_branch_or_apply_to_rows_otherwise(
        branch: dict[str, str | bool] | None,
        apply_to_rows: dict[str, str | bool] | None,
        otherwise: dict[str, Any] | None,
) -> None:
    """Ensure that 'branch', 'apply_to_rows' and 'otherwise' are valid.

    Args:
        branch: Branch configuration.
        apply_to_rows: Apply to rows configuration.
        otherwise: Otherwise configuration.

    Raises:
        ValueError: If the combination of inputs is invalid.
    """
    if branch and apply_to_rows:
        msg = "The user cannot provide at the same time "
        msg += "'branch' and 'apply_to_rows'"
        raise ValueError(msg)

    if otherwise:
        if not (branch or apply_to_rows):
            msg = "'Otherwise' can be provided only for "
            msg += "'branch' and 'apply_to_rows'"
            raise ValueError(msg)

        msg_dead_end = "'Otherwise' cannot be provided in '{}' where "
        msg_dead_end += "'end' == 'dead-end', append another pipeline instead"

        if branch:
            if branch.get("storage"):
                msg = "'Otherwise' cannot be provided in 'branch' where "
                msg += "the 'storage' is set, do the transformation "
                msg += "before branching instead"
                raise ValueError(msg)
            if branch["end"] == "dead-end":
                raise ValueError(msg_dead_end.format("branch"))
        if apply_to_rows:
            if apply_to_rows.get("dead-end"):
                raise ValueError(msg_dead_end.format("apply_to_rows"))


def ensure_no_branch_or_apply_to_rows_in_split_pipeline(
        branch: dict[str, Any] | None, apply_to_rows: dict[str, Any] | None
) -> None:
    """Ensure that 'branch' and 'apply_to_rows' are not passed in split-pipelines.

    Args:
        branch: Branch configuration.
        apply_to_rows: Apply to rows configuration.

    Raises:
        ValueError: If either 'branch' or 'apply_to_rows' is provided.
    """
    if branch:
        msg = "'branch' cannot be provided for 'split pipelines'."
        raise ValueError(msg)

    if apply_to_rows:
        msg = "'apply_to_rows' cannot be provided for 'split pipelines'."
        raise ValueError(msg)


def should_skip_operation(skip: bool | None, perform: bool | None) -> bool:
    """Return True if operation should be skipped."""
    if isinstance(skip, bool) and isinstance(perform, bool):
        if skip == perform:  # Both True or both False = contradiction
            raise ValueError(
                "'skip' and 'perform' cannot both be True or both be False"
            )

    if perform is False:
        return True

    return bool(skip)
