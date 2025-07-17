"""Transformers for Managing Columns without Affecting Row Values."""

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import pyspark.sql.functions as F

from nlsn.nebula.auxiliaries import (
    assert_at_least_one_non_null,
    assert_is_string,
    assert_only_one_non_none,
)
from nlsn.nebula.base import Transformer

__all__ = [
    "AddPrefixSuffixToColumnNames",
    "AddTypedColumns",
    "ColumnsToLowercase",
    "DuplicateColumn",
    "ReplaceDotInColumnNames",
]


class AddPrefixSuffixToColumnNames(Transformer):
    def __init__(
        self,
        *,
        columns: Optional[Union[str, List[str]]] = None,
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
            prefix (str | None):
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

    def _transform(self, df):
        list_selection: List[str] = self._get_selected_columns(df)

        set_selection: Set[str] = set(list_selection)

        out_cols = []
        for c in df.columns:
            if c in set_selection:
                new_col = F.col(c).alias(self._prefix + c + self._suffix)
            else:
                new_col = c
            out_cols.append(new_col)

        return df.select(out_cols)


class AddTypedColumns(Transformer):
    def __init__(
        self,
        *,
        columns: Optional[Union[List[Tuple[str, str]], Dict[str, Any]]],
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

        self._columns: Dict[str, Dict[str, Any]]
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

    def _transform(self, df):
        if self._skip:
            return df

        # Current columns
        current_cols: Set[str] = set(df.columns)

        new_cols = []
        for name, nd in self._columns.items():
            if name in current_cols:  # Do not add new col if already exist
                continue

            value = nd["value"]
            data_type = nd["type"]
            new_cols.append(F.lit(value).cast(data_type).alias(name))

        return df.select("*", *new_cols)


class ColumnsToLowercase(Transformer):
    def __init__(
        self,
        *,
        columns: Optional[Union[str, List[str]]] = None,
        regex: Optional[str] = None,
        glob: Optional[str] = None,
        startswith: Optional[Union[str, Iterable[str]]] = None,
        endswith: Optional[Union[str, Iterable[str]]] = None,
        trim: bool = False,
    ):
        """Rename column names to lowercase.

        Args:
            columns (str | list(str) | None):
                A list of columns to lowercase. Defaults to None.
            regex (str | None):
                Select the columns to lowercase by using a regex pattern.
                Defaults to None.
            glob (str | None):
                Select the columns to lowercase by using a bash-like pattern.
                Defaults to None.
            startswith (str | iterable(str) | None):
                Select all the columns whose names start with the provided
                string(s). Defaults to None.
            endswith (str | iterable(str) | None):
                Select all the columns whose names end with the provided
                string(s). Defaults to None.
            trim (bool):
                If True, trim the spaces from both ends.
                Defaults to False.
        """
        super().__init__()
        self._set_columns_selections(
            columns=columns,
            regex=regex,
            glob=glob,
            startswith=startswith,
            endswith=endswith,
        )
        self._trim: bool = trim

    def _transform(self, df):
        list_cols_to_lowercase: List[str] = self._get_selected_columns(df)

        set_cols_to_lowercase = set(list_cols_to_lowercase)

        new_cols = []
        for c in df.columns:
            if c in set_cols_to_lowercase:
                new_name = c.lower()
                if self._trim:
                    new_name = new_name.strip()
                new_cols.append(F.col(c).alias(new_name))
            else:
                new_cols.append(c)

        return df.select(*new_cols)


class DuplicateColumn(Transformer):
    def __init__(
        self,
        *,
        col_map: Optional[Dict[str, str]] = None,
        pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        """Duplicate columns and assign them new names.

        If a new duplicate column has an existing name, the old column will
        be dropped.

        Only one among col_map and pairs can be provided.

        Args:
            col_map (dict(str, str) | None):
                A dictionary specifying column duplications. Keys represent
                the original column names, and values represent the desired
                new column names. Note that due to the unordered nature
                of dictionaries, the order of duplications is not guaranteed.
                For example, if you have `{"x": "x_dup", "y": "x"}` and the
                "y" mapping is processed first, "x_dup" will be a duplicate of
                "y", not the original "x". To ensure a specific order of
                duplications, use the `pairs` argument instead.
                Defaults to None.
            pairs (list(iterable(str, str)) | None):
                A list of 2-element iterable, where each tuple represents a
                column duplication. The first element of each tuple is the
                original column name, and the second element is the new
                column name. Duplications are performed in the order they
                appear in the list, providing explicit control over the
                sequence of operations. This addresses potential ordering
                issues that can arise when using `col_map`.
                Defaults to None.
        """
        assert_only_one_non_none(col_map, pairs)

        super().__init__()

        self._pairs: List[Tuple[str, str]] = []
        if pairs:
            if not all(len(i) == 2 for i in pairs):
                msg = "'pairs' must be a list of 2-element iterables. Found different lengths"
                raise AssertionError(msg)
            self._pairs = pairs
        else:
            set_keys: Set[str] = set(col_map.keys())
            values = col_map.values()
            ints = set_keys.intersection(set(values))
            if ints:
                msg = (
                    f"Keys and values must be all different. Found intersection {ints}"
                )
                raise AssertionError(msg)

            self._pairs = list(col_map.items())

        values = [i[1] for i in self._pairs]
        set_values: Set[str] = set(values)
        if len(values) != len(set_values):
            raise AssertionError(f"Duplicated values in mapping: {values}")

    def _transform(self, df):
        new_column_names: Set[str] = {i[1] for i in self._pairs}
        old_cols = [i for i in df.columns if i not in new_column_names]
        new_cols = [F.col(k).alias(v) for k, v in self._pairs]
        return df.select(*old_cols, *new_cols)


class ReplaceDotInColumnNames(Transformer):
    def __init__(self, *, replacement: str):
        """Replace all the dots "." in dataframe column names.

        Generally, as a best practice, column names should not contain special
        characters except underscore (_) however, sometimes we may need to
        handle it.

        To access PySpark/Spark DataFrame Column Name with a dot from
        withColumn() & select(), you need to enclose the column name with
        backticks (`).
        Since it is not always possible, especially when <str>, <col> and
        <col.alias> are used during the selection, it is preferable to sanitize
        the columns by replacing the dot with other characters.

        Args:
            replacement (str):
                Replacement for the dot.
        """
        assert_is_string(replacement, "replacement")
        if "." in replacement:
            raise AssertionError("You're a maniac!")

        super().__init__()
        self._rep: str = replacement

    def _transform(self, df):
        cols = [F.col(f"`{c}`").alias(c.replace(".", self._rep)) for c in df.columns]
        return df.select(*cols)
