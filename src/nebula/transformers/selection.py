"""Column selection/manipulation."""

import re
from typing import Iterable

from nebula.auxiliaries import (
    assert_at_least_one_non_null,
    assert_only_one_non_none,
    ensure_flat_list,
)
from nebula.base import Transformer

__all__ = [
    "DropColumns",
    "RenameColumns",
    "SelectColumns",
]


class DropColumns(Transformer):
    def __init__(
            self,
            *,
            columns: str | list[str] | None = None,
            regex: str | None = None,
            glob: str | None = None,
            startswith: str | Iterable[str] | None = None,
            endswith: str | Iterable[str] | None = None,
            allow_excess_columns: bool = True,
    ):
        """Drop a subset of columns.

        Args:
            columns (str | list(str) | None):
                A list of columns to drop. Defaults to None.
            regex (str | None):
                Select the columns to drop by using a regex pattern.
                Defaults to None.
            glob (str | None):
                Select the columns to drop by using a bash-like pattern.
                Defaults to None.
            startswith (str | iterable(str) | None):
                Drop all the columns whose names start with the provided
                string(s). Defaults to None.
            endswith (str | iterable(str) | None):
                Drop all the columns whose names end with the provided
                string(s). Defaults to None.
            allow_excess_columns (bool):
                Whether to allow 'columns' argument to list columns that are
                not present in the dataframe. Default True.
                If 'columns' contains columns that are not present in the
                DataFrame and 'allow_excess_columns' is set to False, raise
                an AssertionError.

        Raises:
            AssertionError: If `allow_excess_columns` is False, and the column
            list contains columns that are not present in the DataFrame.
        """
        assert_at_least_one_non_null(
            columns=columns,
            regex=regex,
            glob=glob,
            startswith=startswith,
            endswith=endswith
        )
        super().__init__()
        self._set_columns_selections(
            columns=columns,
            regex=regex,
            glob=glob,
            startswith=startswith,
            endswith=endswith,
            allow_excess_columns=allow_excess_columns,
        )
        self._allow_excess_columns: bool = allow_excess_columns

    def _transform_nw(self, nw_df):
        selection: list[str] = self._get_selected_columns(nw_df)
        if self._allow_excess_columns:
            actual = set(nw_df.columns)
            selection = [col for col in selection if col in actual]
        return nw_df.drop(selection)


class RenameColumns(Transformer):

    def __init__(
            self,
            *,
            columns: str | list[str] | None = None,
            columns_renamed: str | list[str] | None = None,
            mapping: dict[str, str] | None = None,
            regex_pattern: str | None = None,
            regex_replacement: str | None = None,
            fail_on_missing_columns: bool = True,
    ):
        """Transformer to rename DataFrame columns.

         It can be called using either a pair of lists, a mapping dictionary
         or a regex-based renaming.

        Args:
            columns (str | list[str] | None):
                Original column names. Defaults to None.
            columns_renamed (str | list(str) | None):
                New column names. Defaults to None.
            mapping (dict(str, str) | None):
                A dictionary mapping original column names to new column names.
                Defaults to None.
            regex_pattern (str | None):
                The regex pattern used to match the part of the column names
                that will be replaced. Defaults to None.
            regex_replacement (str | None):
                The regex_replacement string for the matched regex pattern in
                column names. Defaults to None.
            fail_on_missing_columns (bool):
                Whether to raise an error if columns mentioned in `columns` or
                'mapping` are not present in the df. Defaults to True.
                Defaults to True.

            Renaming takes place in the following sequence:
            1. Columns matched with `Regex`
            2. Column rename mapping provided in mapping and
                (columns + columns_renamed).
                If duplicate mappings are provided in mapping as well as
                (columns + columns_renamed), the latter will overwrite the
                mapping of the former.
        """
        assert_only_one_non_none(
            columns=columns,
            mapping=mapping,
            regex_pattern=regex_pattern or regex_replacement
        )
        super().__init__()

        if bool(regex_pattern) != bool(regex_replacement):
            raise ValueError(
                "'replacement' must be provided when 'regex_pattern' is used."
            )
        self._regex_pattern: str | None = regex_pattern
        self._regex_repl: str | None = regex_replacement
        self._fail_on_missing_columns: bool = fail_on_missing_columns

        if bool(columns) != bool(columns_renamed):
            raise ValueError(
                "'columns_renamed' and 'columns' must be used together."
            )

        columns: list[str] = ensure_flat_list(columns)
        columns_renamed: list[str] = ensure_flat_list(columns_renamed)
        if len(columns) != len(columns_renamed):
            raise ValueError(
                f"len(columns)={len(columns)} != len(columns_renamed)={len(columns_renamed)}"
            )

        mapping = mapping if mapping else {}
        self._map_rename: dict[str, str] = (
            {**mapping, **dict(zip(columns, columns_renamed))} if columns else mapping
        )

    def _check_diff(self, df):
        diff = set(self._map_rename.keys()) - set(df.columns)
        if diff and self._fail_on_missing_columns:  # pragma: no cover
            diff_str = ", ".join(diff)
            msg = f"Some columns to be renamed are NOT present in the dataframe! {diff_str}"
            raise ValueError(msg)

    def _get_regex_mapping(self, nw_df) -> dict[str, str]:
        return {c: re.sub(self._regex_pattern, self._regex_repl, c) for c in nw_df.columns}

    def _transform_nw(self, nw_df):
        self._check_diff(nw_df)
        if self._regex_pattern:
            mapping = self._get_regex_mapping(nw_df)
            return nw_df.rename(mapping)
        return nw_df.rename(self._map_rename)


class SelectColumns(Transformer):

    def __init__(
            self,
            *,
            columns: str | list[str] | None = None,
            regex: str | None = None,
            glob: str | None = None,
            startswith: str | Iterable[str] | None = None,
            endswith: str | Iterable[str] | None = None,
    ):
        """Select a subset of columns.

        Args:
            columns (str | list(str) | None):
                A list of columns to select. Defaults to None.
            regex (str | None):
                Take the columns to select by using a regex pattern.
                Defaults to None.
            glob (str | None):
                Take the columns to select by using a bash-like pattern.
                Defaults to None.
            startswith (str | iterable(str) | None):
                Select all the columns whose names start with the provided
                string(s). Defaults to None.
            endswith (str | iterable(str) | None):
                Select all the columns whose names end with the provided
                string(s). Defaults to None.
        """
        assert_at_least_one_non_null(
            columns=columns,
            regex=regex,
            glob=glob,
            startswith=startswith,
            endswith=endswith
        )
        super().__init__()
        self._set_columns_selections(
            columns=columns,
            regex=regex,
            glob=glob,
            startswith=startswith,
            endswith=endswith,
        )

    def _transform_nw(self, nw_df):
        selection: list[str] = self._get_selected_columns(nw_df)
        return nw_df.select(selection)
