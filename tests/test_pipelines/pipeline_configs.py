"""Pipeline configurations for apply_to_rows and branch tests."""

from nebula import TransformerPipeline
from nebula.transformers import AddLiterals, DropColumns

from .auxiliaries import ThisTransformerIsBroken

__all__ = [
    # apply_to_rows
    "pipe_apply_to_rows_basic",
    "pipe_apply_to_rows_dead_end",
    "pipe_apply_to_rows_otherwise",
    "pipe_apply_to_rows_skip",
    "pipe_apply_to_rows_skip_if_empty",
    "pipe_apply_to_rows_comparison_column",
    "pipe_apply_to_rows_missing_cols_error",
    # branch: dead-end
    "pipe_branch_dead_end",
    "pipe_branch_dead_end_from_storage",
    # branch: append
    "pipe_branch_append",
    "pipe_branch_append_new_column",
    "pipe_branch_append_missing_cols_error",
    "pipe_branch_append_from_storage",
    # branch: join
    "pipe_branch_join",
    "pipe_branch_join_from_storage",
    # branch: otherwise
    "pipe_branch_append_otherwise",
    "pipe_branch_join_otherwise",
    # branch: skip/perform
    "pipe_branch_skip",
    "pipe_branch_not_perform",
    "pipe_branch_skip_otherwise",
]


# --- apply_to_rows configs ---


def pipe_apply_to_rows_basic() -> TransformerPipeline:
    """Modify rows where idx > 5."""
    return TransformerPipeline(
        [AddLiterals(data=[{"value": "modified", "alias": "c1"}])],
        apply_to_rows={"input_col": "idx", "operator": "gt", "value": 5},
    )


def pipe_apply_to_rows_dead_end() -> TransformerPipeline:
    """Null rows stored but not merged back."""
    return TransformerPipeline(
        [
            AddLiterals(data=[{"alias": "processed", "value": True}]),
            {"store": "df_null_rows"},
        ],
        apply_to_rows={"input_col": "c1", "operator": "is_null", "dead-end": True},
    )


def pipe_apply_to_rows_otherwise() -> TransformerPipeline:
    """Idx > 5 -> "matched", otherwise -> "not_matched"."""
    return TransformerPipeline(
        [AddLiterals(data=[{"value": "matched", "alias": "c1"}])],
        apply_to_rows={"input_col": "idx", "operator": "gt", "value": 5},
        otherwise=AddLiterals(data=[{"value": "not_matched", "alias": "c1"}]),
    )


def pipe_apply_to_rows_skip() -> TransformerPipeline:
    """Skipped branch — output equals input."""
    return TransformerPipeline(
        [ThisTransformerIsBroken()],
        apply_to_rows={"skip": True, "input_col": "idx", "operator": "gt", "value": -100},
    )


def pipe_apply_to_rows_skip_if_empty() -> TransformerPipeline:
    """No rows match (idx < -100) — skipped via skip_if_empty."""
    return TransformerPipeline(
        [AddLiterals(data=[{"value": "should_not_appear", "alias": "c1"}])],
        apply_to_rows={"input_col": "idx", "operator": "lt", "value": -100, "skip_if_empty": True},
    )


def pipe_apply_to_rows_comparison_column() -> TransformerPipeline:
    """Rows where c1 > c2 get a "result" column."""
    return TransformerPipeline(
        [AddLiterals(data=[{"value": "c1_gt_c2", "alias": "result"}])],
        apply_to_rows={"input_col": "c1", "operator": "gt", "comparison_column": "c2"},
        allow_missing_columns=True,
    )


def pipe_apply_to_rows_missing_cols_error() -> TransformerPipeline:
    """Should raise — new column without allow_missing_columns."""
    return TransformerPipeline(
        [AddLiterals(data=[{"value": "x", "alias": "new_col"}])],
        apply_to_rows={"input_col": "idx", "operator": "gt", "value": 5},
        allow_missing_columns=False,
    )


# --- branch configs ---


def pipe_branch_dead_end() -> TransformerPipeline:
    """Branch stores result but doesn't merge back."""
    return TransformerPipeline(
        [
            AddLiterals(data=[{"value": "from_branch", "alias": "branch_col"}]),
            {"store": "df_branch_result"},
        ],
        branch={"end": "dead-end"},
    )


def pipe_branch_dead_end_from_storage() -> TransformerPipeline:
    """Branch reads from storage, dead-end."""
    return TransformerPipeline(
        [
            AddLiterals(data=[{"value": "from_branch", "alias": "branch_col"}]),
            {"store": "df_branch_result"},
        ],
        branch={"storage": "df_source", "end": "dead-end"},
    )


def pipe_branch_append() -> TransformerPipeline:
    """Branch result appended to main DataFrame."""
    return TransformerPipeline(
        [AddLiterals(data=[{"value": "from_branch", "alias": "c1"}])],
        branch={"end": "append"},
    )


def pipe_branch_append_new_column() -> TransformerPipeline:
    """Branch adds new column with allow_missing_columns."""
    return TransformerPipeline(
        [AddLiterals(data=[{"value": "branch_value", "alias": "new_col"}])],
        branch={"end": "append"},
        allow_missing_columns=True,
    )


def pipe_branch_append_missing_cols_error() -> TransformerPipeline:
    """Should raise — new column without allow_missing_columns."""
    return TransformerPipeline(
        [AddLiterals(data=[{"value": "branch_value", "alias": "new_col"}])],
        branch={"end": "append"},
        allow_missing_columns=False,
    )


def pipe_branch_append_from_storage() -> TransformerPipeline:
    """Branch reads from storage and appends."""
    return TransformerPipeline(
        [AddLiterals(data=[{"value": "from_storage_branch", "alias": "c1"}])],
        branch={"storage": "df_source", "end": "append"},
    )


def pipe_branch_join() -> TransformerPipeline:
    """Branch joined to main on 'idx'."""
    return TransformerPipeline(
        [
            DropColumns(columns=["c1", "c2"]),
            AddLiterals(data=[{"value": "joined", "alias": "new_col"}]),
        ],
        branch={"end": "join", "on": "idx", "how": "inner"},
    )


def pipe_branch_join_from_storage() -> TransformerPipeline:
    """Branch reads from storage and joins."""
    return TransformerPipeline(
        [
            DropColumns(columns=["c1", "c2"]),
            AddLiterals(data=[{"value": "joined_from_storage", "alias": "new_col"}]),
        ],
        branch={"storage": "df_source", "end": "join", "on": "idx", "how": "inner"},
    )


def pipe_branch_append_otherwise() -> TransformerPipeline:
    """Branch appends, main also transformed via otherwise."""
    return TransformerPipeline(
        [AddLiterals(data=[{"value": "branch_transformed", "alias": "c1"}])],
        branch={"end": "append"},
        otherwise=AddLiterals(data=[{"value": "main_transformed", "alias": "c1"}]),
    )


def pipe_branch_join_otherwise() -> TransformerPipeline:
    """Branch joins, main transformed via otherwise."""
    return TransformerPipeline(
        [
            DropColumns(columns=["c1", "c2"]),
            AddLiterals(data=[{"value": "joined", "alias": "new_col"}]),
        ],
        branch={"end": "join", "on": "idx", "how": "inner"},
        otherwise=AddLiterals(data=[{"value": "main_marker", "alias": "other_col"}]),
    )


def pipe_branch_skip() -> TransformerPipeline:
    """Branch skipped, main passes through."""
    return TransformerPipeline(
        [AddLiterals(data=[{"value": "should_not_appear", "alias": "c1"}])],
        branch={"end": "append", "skip": True},
    )


def pipe_branch_not_perform() -> TransformerPipeline:
    """Branch not performed (perform=False), main passes through."""
    return TransformerPipeline(
        [AddLiterals(data=[{"value": "should_not_appear", "alias": "c1"}])],
        branch={"end": "append", "perform": False},
    )


def pipe_branch_skip_otherwise() -> TransformerPipeline:
    """Branch skipped, but otherwise still runs."""
    return TransformerPipeline(
        [AddLiterals(data=[{"value": "should_not_appear", "alias": "c1"}])],
        branch={"end": "append", "skip": True},
        otherwise=AddLiterals(data=[{"value": "otherwise_applied", "alias": "c1"}]),
    )
