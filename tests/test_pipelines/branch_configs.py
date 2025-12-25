"""Pipeline configurations for branch tests.

Each function returns a configured TransformerPipeline ready to run.

Branch creates a secondary pipeline that:
- Optionally reads from storage (or uses the main DataFrame)
- Transforms the data
- Merges back via: append, join, or dead-end (no merge)
"""

from nebula.pipelines.pipelines import TransformerPipeline
from nebula.transformers import DropColumns, AddLiterals

__all__ = [
    # Dead-end variants
    "pipe_branch_dead_end",
    "pipe_branch_dead_end_from_storage",
    # Append variants
    "pipe_branch_append",
    "pipe_branch_append_new_column",
    "pipe_branch_append_missing_cols_error",
    "pipe_branch_append_from_storage",
    # Join variants
    "pipe_branch_join",
    "pipe_branch_join_from_storage",
    # Otherwise variants
    "pipe_branch_append_otherwise",
    "pipe_branch_join_otherwise",
    # Skip/perform variants
    "pipe_branch_skip",
    "pipe_branch_not_perform",
    "pipe_branch_skip_otherwise",
]


# ============================================================================
# Dead-end: branch runs but result is discarded
# ============================================================================

def pipe_branch_dead_end() -> TransformerPipeline:
    """Branch that stores result but doesn't merge back.

    Main DataFrame passes through unchanged.
    Branch result is stored for later inspection.
    """
    return TransformerPipeline(
        [
            AddLiterals(data=[{"value": "from_branch", "alias": "branch_col"}]),
            {"store": "df_branch_result"},
        ],
        branch={"end": "dead-end"},
    )


def pipe_branch_dead_end_from_storage() -> TransformerPipeline:
    """Branch reads from storage instead of main DataFrame.

    Requires ns.set("df_source", ...) before running.
    Main DataFrame passes through unchanged.
    """
    return TransformerPipeline(
        [
            AddLiterals(data=[{"value": "from_branch", "alias": "branch_col"}]),
            {"store": "df_branch_result"},
        ],
        branch={
            "storage": "df_source",
            "end": "dead-end",
        },
    )


# ============================================================================
# Append: branch result is appended (union) to main DataFrame
# ============================================================================

def pipe_branch_append() -> TransformerPipeline:
    """Branch result is appended to main DataFrame.

    Output has 2x rows (original + branch with modified c1).
    """
    return TransformerPipeline(
        [AddLiterals(data=[{"value": "from_branch", "alias": "c1"}])],
        branch={"end": "append"},
    )


def pipe_branch_append_new_column() -> TransformerPipeline:
    """Branch adds a new column, requires allow_missing_columns.

    Output has 2x rows. Original rows have null in new_col.
    """
    return TransformerPipeline(
        [AddLiterals(data=[{"value": "branch_value", "alias": "new_col"}])],
        branch={"end": "append"},
        allow_missing_columns=True,
    )


def pipe_branch_append_missing_cols_error() -> TransformerPipeline:
    """Should raise: branch adds column without allow_missing_columns."""
    return TransformerPipeline(
        [AddLiterals(data=[{"value": "branch_value", "alias": "new_col"}])],
        branch={"end": "append"},
        allow_missing_columns=False,
    )


def pipe_branch_append_from_storage() -> TransformerPipeline:
    """Branch reads from storage and appends to main DataFrame.

    Requires ns.set("df_source", ...) before running.
    """
    return TransformerPipeline(
        [AddLiterals(data=[{"value": "from_storage_branch", "alias": "c1"}])],
        branch={
            "storage": "df_source",
            "end": "append",
        },
    )


# ============================================================================
# Join: branch result is joined to main DataFrame
# ============================================================================

def pipe_branch_join() -> TransformerPipeline:
    """Branch result is joined to main DataFrame on 'idx'.

    Branch drops c1, c2 and adds new_col.
    Result has original columns + new_col.
    """
    return TransformerPipeline(
        [
            DropColumns(columns=["c1", "c2"]),
            AddLiterals(data=[{"value": "joined", "alias": "new_col"}]),
        ],
        branch={
            "end": "join",
            "on": "idx",
            "how": "inner",
        },
    )


def pipe_branch_join_from_storage() -> TransformerPipeline:
    """Branch reads from storage and joins to main DataFrame.

    Requires ns.set("df_source", ...) before running.
    """
    return TransformerPipeline(
        [
            DropColumns(columns=["c1", "c2"]),
            AddLiterals(data=[{"value": "joined_from_storage", "alias": "new_col"}]),
        ],
        branch={
            "storage": "df_source",
            "end": "join",
            "on": "idx",
            "how": "inner",
        },
    )


# ============================================================================
# Otherwise: separate transform for main DataFrame
# ============================================================================

def pipe_branch_append_otherwise() -> TransformerPipeline:
    """Branch appends, but main DataFrame also gets transformed.

    - Main DataFrame: c1 = "main_transformed"
    - Branch: c1 = "branch_transformed"
    - Result: union of both
    """
    return TransformerPipeline(
        [AddLiterals(data=[{"value": "branch_transformed", "alias": "c1"}])],
        branch={"end": "append"},
        otherwise=AddLiterals(data=[{"value": "main_transformed", "alias": "c1"}]),
    )


def pipe_branch_join_otherwise() -> TransformerPipeline:
    """Branch joins, but main DataFrame also gets transformed.

    - Main DataFrame: gets other_col = "main_marker"
    - Branch: provides new_col = "joined"
    - Result: join of transformed main + branch
    """
    return TransformerPipeline(
        [
            DropColumns(columns=["c1", "c2"]),
            AddLiterals(data=[{"value": "joined", "alias": "new_col"}]),
        ],
        branch={
            "end": "join",
            "on": "idx",
            "how": "inner",
        },
        otherwise=AddLiterals(data=[{"value": "main_marker", "alias": "other_col"}]),
    )


# ============================================================================
# Skip/Perform: conditionally disable the branch
# ============================================================================

def pipe_branch_skip() -> TransformerPipeline:
    """Branch is skipped entirely, main DataFrame passes through.

    The branch transform should NOT be applied.
    """
    return TransformerPipeline(
        [AddLiterals(data=[{"value": "should_not_appear", "alias": "c1"}])],
        branch={
            "end": "append",
            "skip": True,
        },
    )


def pipe_branch_not_perform() -> TransformerPipeline:
    """Branch is skipped (perform=False) entirely, main DataFrame passes through.

    The branch transform should NOT be applied.
    """
    return TransformerPipeline(
        [AddLiterals(data=[{"value": "should_not_appear", "alias": "c1"}])],
        branch={
            "end": "append",
            "perform": False,
        },
    )


def pipe_branch_skip_otherwise() -> TransformerPipeline:
    """Branch is skipped, but otherwise pipeline still runs.

    Only the otherwise transform is applied.
    """
    return TransformerPipeline(
        [AddLiterals(data=[{"value": "should_not_appear", "alias": "c1"}])],
        branch={
            "end": "append",
            "skip": True,
        },
        otherwise=AddLiterals(data=[{"value": "otherwise_applied", "alias": "c1"}]),
    )
