"""Shared code for pipelines."""

from functools import partial

from nlsn.nebula.base import Transformer
from nlsn.nebula.pipelines.pipelines import TransformerPipeline
from nlsn.nebula.shared_transformers import Count, Distinct, DropColumns, WithColumn


class RuntimeErrorTransformer(Transformer):
    def __init__(self):
        """Transformer that raises RuntimeError."""
        super().__init__()

    def transform(self, df):
        """Raise RuntimeError."""
        raise RuntimeError("RuntimeErrorTransformer")


def _get_apply_to_rows_is_null_dead_end():
    pipe = TransformerPipeline(
        [
            WithColumn(column_name="new", value=-1),
            {"store": "df_fork"},
        ],
        apply_to_rows={
            "input_col": "c1",
            "operator": "isNull",
            "dead-end": True,
            "skip_if_empty": True,
        },
    )
    return pipe


def _get_apply_to_rows_gt():
    pipe = TransformerPipeline(
        [
            WithColumn(column_name="c1", value="x"),
        ],
        apply_to_rows={"input_col": "idx", "operator": "gt", "value": 5},
        coalesce_output_to_original=True,
    )
    return pipe


def _get_apply_to_rows_comparison_col():
    pipe = TransformerPipeline(
        [
            WithColumn(column_name="new_column", value="new_value"),
        ],
        apply_to_rows={
            "input_col": "c1",
            "operator": "gt",
            "comparison_column": "c2",
            "skip_if_empty": True,
        },
        allow_missing_columns=True,
        repartition_output_to_original=True,
    )
    return pipe


def _get_apply_to_rows_error():
    pipe = TransformerPipeline(
        [
            WithColumn(column_name="new_column", value="new_value"),
        ],
        apply_to_rows={"input_col": "c1", "operator": "gt", "comparison_column": "c2"},
        allow_missing_columns=False,
    )
    return pipe


def _get_apply_to_rows_otherwise():
    pipe = TransformerPipeline(
        [
            WithColumn(column_name="c1", value="x"),
        ],
        apply_to_rows={
            "input_col": "idx",
            "operator": "gt",
            "value": 5,
            "skip_if_empty": True,
        },
        otherwise=WithColumn(column_name="c1", value="other"),
        coalesce_output_to_original=True,
    )
    return pipe


def _get_apply_to_rows_skip_if_empty():
    pipe = TransformerPipeline(
        [
            WithColumn(column_name="c1", value="x"),
        ],
        apply_to_rows={
            "input_col": "idx",
            "operator": "lt",
            "value": -100,
            "skip_if_empty": True,
        },
        coalesce_output_to_original=True,
    )
    return pipe


def __assemble_branch_pipe(fork_pipe: TransformerPipeline) -> TransformerPipeline:
    first_stage_pipe = TransformerPipeline(Distinct())
    last_stage_pipe = TransformerPipeline(Count())
    pipe_full = TransformerPipeline(
        [
            first_stage_pipe,
            fork_pipe,
            last_stage_pipe,
        ]
    )
    return pipe_full


def _get_branch_dead_end_without_storage():
    fork_pipe = TransformerPipeline(
        [
            WithColumn(column_name="new", value=-1),
            {"store": "df_fork"},
        ],
        branch={"end": "dead-end"},
    )
    return __assemble_branch_pipe(fork_pipe)


def _get_branch_append_without_storage():
    fork_pipe = TransformerPipeline(
        [
            WithColumn(column_name="c1", value="c"),
        ],
        branch={"end": "append"},
    )
    return __assemble_branch_pipe(fork_pipe)


def _get_branch_append_without_storage_error():
    fork_pipe = TransformerPipeline(
        [
            WithColumn(column_name="new_column", value="new_value"),
        ],
        branch={"end": "append"},
        allow_missing_columns=False,
    )
    return __assemble_branch_pipe(fork_pipe)


def _get_branch_append_without_storage_new_col():
    fork_pipe = TransformerPipeline(
        [
            WithColumn(column_name="new_column", value="new_value"),
        ],
        branch={"end": "append"},
        allow_missing_columns=True,
    )
    return __assemble_branch_pipe(fork_pipe)


def _get_branch_join_without_storage():
    fork_pipe = TransformerPipeline(
        [
            DropColumns(columns=["c1", "c2"]),
            WithColumn(column_name="new", value=-1),
        ],
        branch={"end": "join", "on": "idx", "how": "inner"},
    )
    return __assemble_branch_pipe(fork_pipe)


def _get_branch_dead_end_with_storage():
    fork_pipe = TransformerPipeline(
        [
            WithColumn(column_name="new", value=-1),
            {"store": "df_fork"},
        ],
        branch={"storage": "df_x", "end": "dead-end"},
    )
    return __assemble_branch_pipe(fork_pipe)


def _get_branch_append_with_storage():
    fork_pipe = TransformerPipeline(
        [
            WithColumn(column_name="c1", value="c"),
        ],
        branch={"storage": "df_x", "end": "append"},
    )
    return __assemble_branch_pipe(fork_pipe)


def _get_branch_append_with_storage_error():
    fork_pipe = TransformerPipeline(
        [
            WithColumn(column_name="new_column", value="new_value"),
        ],
        branch={"storage": "df_x", "end": "append"},
        allow_missing_columns=False,
    )
    return __assemble_branch_pipe(fork_pipe)


def _get_branch_append_with_storage_new_col():
    fork_pipe = TransformerPipeline(
        [
            WithColumn(column_name="new_column", value="new_value"),
        ],
        branch={"storage": "df_x", "end": "append"},
        allow_missing_columns=True,
    )
    return __assemble_branch_pipe(fork_pipe)


def _get_branch_join_with_storage():
    fork_pipe = TransformerPipeline(
        [
            DropColumns(columns=["c1", "c2"]),
            WithColumn(column_name="new", value=-1),
        ],
        branch={
            "storage": "df_x",
            "end": "join",
            "on": "idx",
            "how": "inner",
            "broadcast": True,
        },
    )
    return __assemble_branch_pipe(fork_pipe)


def _get_branch_append_otherwise():
    fork_pipe = TransformerPipeline(
        [
            WithColumn(column_name="c1", value="c"),
        ],
        branch={"end": "append"},
        otherwise=TransformerPipeline(WithColumn(column_name="c1", value="other")),
    )
    return __assemble_branch_pipe(fork_pipe)


def _get_branch_join_otherwise():
    fork_pipe = TransformerPipeline(
        [
            DropColumns(columns=["c1", "c2"]),
            WithColumn(column_name="new", value=-1),
        ],
        branch={"end": "join", "on": "idx", "how": "inner", "broadcast": False},
        otherwise=WithColumn(column_name="other_col", value="other"),
    )
    return __assemble_branch_pipe(fork_pipe)


def _get_branch_skip():
    fork_pipe = TransformerPipeline(
        [
            WithColumn(column_name="c1", value="c"),
        ],
        branch={"end": "append", "skip": True},
    )
    return __assemble_branch_pipe(fork_pipe)


def _get_branch_skip_otherwise():
    fork_pipe = TransformerPipeline(
        [
            WithColumn(column_name="c1", value="c"),
        ],
        branch={"end": "append", "skip": True},
        otherwise=TransformerPipeline(WithColumn(column_name="c1", value="other")),
    )
    return __assemble_branch_pipe(fork_pipe)


def _skip_flat_pipeline():
    pipe = TransformerPipeline(
        [
            WithColumn(column_name="c1", value="x"),
        ],
        skip=True,
    )
    return pipe


def _skip_split_pipeline(skip=True):
    pipe = TransformerPipeline(
        {
            "c_x": WithColumn(column_name="c_x", value="x"),
            "c_y": WithColumn(column_name="c_y", value="y"),
        },
        split_function=lambda x: x,
        skip=skip,
    )
    return pipe


def _skip_nested_pipeline():
    split_pipe = partial(_skip_split_pipeline, skip=True)
    pipe = TransformerPipeline(
        [
            WithColumn(column_name="c1", value="x"),
            split_pipe,
        ],
        skip=True,
    )
    return pipe


DICT_APPLY_TO_ROWS_PIPELINES = {
    "apply_to_rows_is_null_dead_end": _get_apply_to_rows_is_null_dead_end,
    "apply_to_rows_gt": _get_apply_to_rows_gt,
    "apply_to_rows_comparison_col": _get_apply_to_rows_comparison_col,
    "apply_to_rows_error": _get_apply_to_rows_error,
    "apply_to_rows_otherwise": _get_apply_to_rows_otherwise,
    "apply_to_rows_skip_if_empty": _get_apply_to_rows_skip_if_empty,
}

DICT_BRANCH_PIPELINE = {
    "branch_dead_end_without_storage": _get_branch_dead_end_without_storage,
    "branch_append_without_storage": _get_branch_append_without_storage,
    "branch_append_without_storage_error": _get_branch_append_without_storage_error,
    "branch_append_without_storage_new_col": _get_branch_append_without_storage_new_col,
    "branch_join_without_storage": _get_branch_join_without_storage,
    "branch_dead_end_with_storage": _get_branch_dead_end_with_storage,
    "branch_append_with_storage": _get_branch_append_with_storage,
    "branch_append_with_storage_error": _get_branch_append_with_storage_error,
    "branch_append_with_storage_new_col": _get_branch_append_with_storage_new_col,
    "branch_join_with_storage": _get_branch_join_with_storage,
    "branch_append_otherwise": _get_branch_append_otherwise,
    "branch_join_otherwise": _get_branch_join_otherwise,
    "branch_skip": _get_branch_skip,
    "branch_skip_otherwise": _get_branch_skip_otherwise,
    # Same of 2 previous pipelines but the skip is obtained by setting perform=True
    "branch_not_perform": _get_branch_skip,
    "branch_not_perform_otherwise": _get_branch_skip_otherwise,
}

DICT_SKIP_PIPELINE = {
    "skip_flat_pipeline": _skip_flat_pipeline,
    "skip_split_pipeline": _skip_split_pipeline,
    "skip_nested_pipeline": _skip_nested_pipeline,
    # Same pipelines but the skip is obtained by setting perform=True
    "dont_perform_flat_pipeline": _skip_flat_pipeline,
    "dont_perform_split_pipeline": _skip_split_pipeline,
    "dont_perform_nested_pipeline": _skip_nested_pipeline,
}

DICT_SKIP_TRANSFORMER = {
    "skip_transformer": TransformerPipeline([]),
    "dont_perform_transformer": TransformerPipeline([]),
}
