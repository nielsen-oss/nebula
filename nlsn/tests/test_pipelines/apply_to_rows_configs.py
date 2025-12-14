"""Pipeline configurations for apply_to_rows tests.

These replace the YAML files and _shared.py factory functions.
Each function returns a configured TransformerPipeline ready to run.
"""

from nlsn.nebula.pipelines.pipelines import TransformerPipeline
from nlsn.nebula.transformers import AddLiterals

__all__ = [
    "pipe_apply_to_rows_basic",
    "pipe_apply_to_rows_dead_end",
    "pipe_apply_to_rows_otherwise",
    "pipe_apply_to_rows_skip_if_empty",
    "pipe_apply_to_rows_comparison_column",
    "pipe_apply_to_rows_missing_cols_error",
]


def pipe_apply_to_rows_basic() -> TransformerPipeline:
    """Basic apply_to_rows: modify rows where idx > 5."""
    return TransformerPipeline(
        [AddLiterals(data=[{"value": "modified", "alias": "c1"}])],
        apply_to_rows={
            "input_col": "idx",
            "operator": "gt",
            "value": 5,
        },
    )


def pipe_apply_to_rows_dead_end() -> TransformerPipeline:
    """Apply to null rows, store them, but don't merge back.

    This tests:
    - is_null operator (no value needed)
    - dead-end behavior (matched rows excluded from output)
    - storage within apply_to_rows
    """
    return TransformerPipeline(
        [
            AddLiterals(data=[{"alias": "processed", "value": True}]),
            {"store": "df_null_rows"},
        ],
        apply_to_rows={
            "input_col": "c1",
            "operator": "is_null",
            "dead-end": True,
        },
    )


def pipe_apply_to_rows_otherwise() -> TransformerPipeline:
    """Apply different transforms to matching vs non-matching rows.

    - Rows where idx > 5: c1 = "matched"
    - Rows where idx <= 5: c1 = "not_matched"
    """
    return TransformerPipeline(
        [AddLiterals(data=[{"value": "matched", "alias": "c1"}])],
        apply_to_rows={
            "input_col": "idx",
            "operator": "gt",
            "value": 5,
        },
        otherwise=AddLiterals(data=[{"value": "not_matched", "alias": "c1"}]),
    )


def pipe_apply_to_rows_skip_if_empty() -> TransformerPipeline:
    """Skip the branch entirely if no rows match the condition.

    Uses idx < -100 which matches nothing, so the transform is skipped
    and the output equals the input unchanged.
    """
    return TransformerPipeline(
        [AddLiterals(data=[{"value": "should_not_appear", "alias": "c1"}])],
        apply_to_rows={
            "input_col": "idx",
            "operator": "lt",
            "value": -100,
            "skip_if_empty": True,
        },
    )


def pipe_apply_to_rows_comparison_column() -> TransformerPipeline:
    """Compare two columns instead of column vs literal value.

    Rows where c1 > c2 get a new column added.
    Uses allow_missing_columns since the branch adds a column.
    """
    return TransformerPipeline(
        [AddLiterals(data=[{"value": "c1_gt_c2", "alias": "result"}])],
        apply_to_rows={
            "input_col": "c1",
            "operator": "gt",
            "comparison_column": "c2",
        },
        allow_missing_columns=True,
    )


def pipe_apply_to_rows_missing_cols_error() -> TransformerPipeline:
    """Should raise error: new column without allow_missing_columns.

    The branch adds a column that doesn't exist in the non-matching rows,
    but allow_missing_columns=False (default), so append should fail.
    """
    return TransformerPipeline(
        [AddLiterals(data=[{"value": "x", "alias": "new_col"}])],
        apply_to_rows={
            "input_col": "idx",
            "operator": "gt",
            "value": 5,
        },
        allow_missing_columns=False,
    )