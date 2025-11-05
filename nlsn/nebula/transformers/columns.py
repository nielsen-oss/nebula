"""Transformers for Managing Columns without Affecting Row Values."""

import re
from typing import Any, Iterable, Optional, Union

import narwhals as nw

from nlsn.nebula.auxiliaries import (
    assert_at_least_one_non_null,
    assert_only_one_non_none,
    ensure_flat_list,
)
from nlsn.nebula.base import Transformer

__all__ = [
    "AddPrefixSuffixToColumnNames",
    "AddTypedColumns",
    "DropColumns",
    "RenameColumns",
    "SelectColumns",
]


class AddPrefixSuffixToColumnNames(Transformer):
    def __init__(
            self,
            *,
            columns: Optional[Union[str, list[str]]] = None,
            regex: Optional[str] = None,
            glob: Optional[str] = None,
            startswith: Optional[Union[str, Iterable[str]]] = None,
            endswith: Optional[Union[str, Iterable[str]]] = None,
            prefix: Optional[str] = None,
            suffix: Optional[str] = None,
            allow_excess_columns: bool = False,
    ):
        """Add the prefix and / or suffix to column names.

        This does not change the field values.

        Args:
            columns (str | list(str) | None):
                Column(s) to rename. Defaults to None.
            regex (str | None):
                Select the columns to rename by using a regex pattern.
                Defaults to None.
            glob (str | None):
                Select the columns to rename by using a bash-like pattern.
                Defaults to None.
            startswith (str | iterable(str) | None):
                Select all the columns whose names start with the provided
                string(s). Defaults to None.
            endswith (str | iterable(str) | None):
                Select all the columns whose names end with the provided
                string(s). Defaults to None.
            prefix (str | None):
                Prefix to add at the beginning of the newly created column(s).
                Defaults to None.
            suffix (str | None):
                Suffix to add at the end of the newly created column(s).
                Defaults to None.
            allow_excess_columns (bool):
                Whether to allow columns that are not contained in the
                dataframe (raises AssertionError by default).
                Defaults to False.
        """
        assert_at_least_one_non_null(prefix, suffix)

        super().__init__()
        self._prefix: str = prefix if prefix else ""
        self._suffix: str = suffix if suffix else ""
        self._set_columns_selections(
            columns=columns,
            regex=regex,
            glob=glob,
            startswith=startswith,
            endswith=endswith,
            allow_excess_columns=allow_excess_columns,
        )

    def _transform_nw(self, nw_df):
        selection: list[str] = self._get_selected_columns(nw_df)
        set_selection: set[str] = set(selection)

        rename_mapping = {}
        for col in nw_df.columns:
            if col in set_selection:
                rename_mapping[col] = self._prefix + col + self._suffix

        return nw_df.rename(rename_mapping)


class AddTypedColumns(Transformer):
    def __init__(
            self,
            *,
            columns: Optional[Union[list[tuple[str, str]], dict[str, Any]]],
    ):
        """Add typed columns if they do not exist in the DF.

        For each column, the data type must be specified.
        If 'columns' is null/empty, the transformer is a pass-through.

        Args:
            columns (list(tuple(str, str)) | dict(str, Any) | None):
            3 different input types:
            - list(tuple(str, str)): the first value represents the column
                name and the second one the data-type. These fields are
                filled with null values.
            - dict(str, any): the key represents the column name.
                If the value is a <string>, it represents the data-type; values
                will be filled with null.
                If the value is <dict>, it must be in the form:
                {"type": str, "value": any}, where the nested "value" indicates
                the filling value.
            - [] | {} | None: do nothing.
        """
        super().__init__()

        self._columns: dict[str, dict[str, Any]]
        self._skip: bool = False

        if not columns:
            self._skip = True
            return

        if isinstance(columns, dict):
            self._assert_keys_strings(columns)
            self._check_default_value(columns)
            # Sort for repeatability
            columns_raw = sorted(columns.items())

        else:
            if not isinstance(columns, (tuple, list)):
                msg = '"columns" must be <list> | <tuple> | <dict <str, str>>'
                raise AssertionError(msg)
            unique_len = {len(i) for i in columns}
            if unique_len != {2}:
                msg = 'If "columns" is a <list> | <tuple> it must contain '
                msg += "2-element iterables"
                raise AssertionError(msg)
            columns_raw = columns

        # Convert 'columns_raw' into a dictionary like:
        # {"column_name": {"type": datatype, "value": value}}
        self._columns = {}
        for k, obj in columns_raw:
            if isinstance(obj, dict):
                datatype = obj["type"]
                value = obj["value"]
            else:
                datatype = obj
                value = None

            self._columns.update({k: {"type": datatype, "value": value}})

    @staticmethod
    def _assert_keys_strings(dictionary):
        for k in dictionary.keys():
            if not isinstance(k, str):
                msg = "All keys in the dictionary must be <string>"
                raise AssertionError(msg)

    @staticmethod
    def _check_default_value(dictionary):
        _allowed = {"type", "value"}

        for nd in dictionary.values():
            if not isinstance(nd, dict):
                continue

            set_keys = nd.keys()
            if set_keys != _allowed:
                msg = f"Allowed keys in nested dictionary: {_allowed}. "
                msg += f"Found: {set_keys}."
                raise AssertionError(msg)

    def _transform_nw(self, nw_df):
        if self._skip:
            return nw_df

        # Current columns
        current_cols: set[str] = set(nw_df.columns)

        # Build a list of new columns to add
        new_cols_exprs = []
        for name, nd in self._columns.items():
            if name in current_cols:  # Do not add new col if already exist
                continue

            value = nd["value"]
            data_type = nd["type"]

            # Create literal column with specified value
            new_cols_exprs.append(nw.lit(value).cast(data_type).alias(name))

        if not new_cols_exprs:
            return nw_df

        # Add new columns to existing dataframe
        return nw_df.with_columns(new_cols_exprs)


class DropColumns(Transformer):
    def __init__(
            self,
            *,
            columns: Optional[Union[str, list[str]]] = None,
            regex: Optional[str] = None,
            glob: Optional[str] = None,
            startswith: Optional[Union[str, Iterable[str]]] = None,
            endswith: Optional[Union[str, Iterable[str]]] = None,
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
        assert_at_least_one_non_null(columns, regex, glob, startswith, endswith)
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
            columns: Optional[Union[str, list[str]]] = None,
            columns_renamed: Optional[Union[str, list[str]]] = None,
            mapping: Optional[dict[str, str]] = None,
            regex_pattern: Optional[str] = None,
            regex_replacement: Optional[str] = None,
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
        assert_only_one_non_none(columns, mapping, regex_pattern)
        super().__init__()

        if bool(regex_pattern) != bool(regex_replacement):
            raise AssertionError(
                "'replacement' must be provided when 'regex_pattern' is used."
            )
        self._regex_pattern: Optional[str] = regex_pattern
        self._regex_repl: Optional[str] = regex_replacement
        self._fail_on_missing_columns: bool = fail_on_missing_columns

        if bool(columns) != bool(columns_renamed):
            raise AssertionError(
                "'columns_renamed' and 'columns' must be used together."
            )

        columns: list[str] = ensure_flat_list(columns)
        columns_renamed: list[str] = ensure_flat_list(columns_renamed)
        if len(columns) != len(columns_renamed):
            raise AssertionError(
                f"len(columns)={len(columns)} != len(columns_renamed)={len(columns_renamed)}"
            )

        mapping = mapping if mapping else {}
        self._map_rename: dict[str, str] = (
            {**mapping, **dict(zip(columns, columns_renamed))} if columns else mapping
        )

    def _check_diff(self, df):
        diff = set(self._map_rename.keys()) - set(df.columns)
        if diff and self._fail_on_missing_columns:
            diff_str = ", ".join(diff)
            msg = f"Some columns to be renamed are NOT present in the dataframe! {diff_str}"
            raise AssertionError(msg)

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
            columns: Optional[Union[str, list[str]]] = None,
            regex: Optional[str] = None,
            glob: Optional[str] = None,
            startswith: Optional[Union[str, Iterable[str]]] = None,
            endswith: Optional[Union[str, Iterable[str]]] = None,
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
        assert_at_least_one_non_null(columns, regex, glob, startswith, endswith)
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
