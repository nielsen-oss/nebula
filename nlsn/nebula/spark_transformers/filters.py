"""Row Filtering Operations."""

from functools import reduce
from operator import and_
from typing import Any, Iterable, List, Optional, Union

from pyspark.sql import functions as F

from nlsn.nebula.auxiliaries import assert_allowed
from nlsn.nebula.base import Transformer
from nlsn.nebula.spark_util import (
    drop_duplicates_no_randomness,
    ensure_spark_condition,
    get_spark_condition,
    null_cond_to_false,
)

__all__ = [
    "DiscardNulls",
    "DropAnyNullOrEmptyArray",
    "DropAnyNullOrEmptyString",
    "DropDuplicates",
    "Filter",
]


def select_all_columns(cols, regex, glob: Optional[str]) -> bool:
    """Check if the user is asking for all columns or a subset."""
    ret = glob == "*"  # means all columns
    ret |= not (cols or regex or glob)  # means no particular subset
    return ret


class DiscardNulls(Transformer):
    def __init__(
        self,
        *,
        how: str,
        columns: Optional[Union[str, List[str]]] = None,
        regex: Optional[str] = None,
        glob: Optional[str] = None,
        startswith: Optional[Union[str, Iterable[str]]] = None,
        endswith: Optional[Union[str, Iterable[str]]] = None,
    ):
        """Drop rows with null values.

        Input parameters are eventually used to select a subset of the columns.

        Args:
            how (str):
                "any" or "all". If "any", drop a row if it contains any nulls.
                If "all", drop a row only if all its values are null.
            columns (str | list(str) | None):
                List of columns to consider. Defaults to None.
            regex (str | None):
                Take the columns to consider by using a regex pattern.
                Defaults to None.
            glob (str | None):
                Take the columns to consider by using a bash-like pattern.
                Defaults to None.
            startswith (str | iterable(str) | None):
                Select all the columns to consider whose names start with the
                provided string(s). Defaults to None.
            endswith (str | iterable(str) | None):
                Select all the columns to consider whose names end with the
                provided string(s). Defaults to None.
        """
        assert_allowed(how, {"any", "all"}, "how")
        super().__init__()
        self._how: str = how
        self._set_columns_selections(
            columns=columns,
            regex=regex,
            glob=glob,
            startswith=startswith,
            endswith=endswith,
        )

    def _transform(self, df):
        subset: List[str] = self._get_selected_columns(df)
        if subset and set(subset) != set(list(df.columns)):
            return df.dropna(self._how, subset=subset)
        return df.dropna(self._how)


class DropAnyNullOrEmptyArray(Transformer):
    def __init__(
        self,
        *,
        columns: Optional[Union[str, List[str]]] = None,
        regex: Optional[str] = None,
        glob: Optional[str] = None,
        startswith: Optional[Union[str, Iterable[str]]] = None,
        endswith: Optional[Union[str, Iterable[str]]] = None,
    ):
        """Drop any rows where the arrays in the considered columns are empty or null.

        Args:
            columns (str | list(str) | None):
                A list of the columns. Defaults to None.
            regex (str | None):
                Select the columns by using a regex pattern.
                Defaults to None.
            glob (str | None):
                Select the columns by using a bash-like pattern.
                Defaults to None.
            startswith (str | iterable(str) | None):
                Select all the columns whose names start with the provided
                string(s). Defaults to None.
            endswith (str | iterable(str) | None):
                Select all the columns whose names end with the provided
                string(s). Defaults to None.
        """
        super().__init__()
        self._set_columns_selections(
            columns=columns,
            regex=regex,
            glob=glob,
            startswith=startswith,
            endswith=endswith,
        )

    def _transform(self, df):
        selection: List[str] = self._get_selected_columns(df)
        array_types = {c.name for c in df.schema if c.dataType.typeName() == "array"}
        subset: List[str] = [i for i in selection if i in array_types]

        li_size = [F.size(c) for c in subset]

        # keep non empty array and non null
        li_cond = [(c.isNotNull() & (c > 0)) for c in li_size]
        return df.filter(reduce(and_, li_cond))


class DropAnyNullOrEmptyString(Transformer):
    def __init__(
        self,
        *,
        trim: bool = False,
        columns: Optional[Union[str, List[str]]] = None,
        regex: Optional[str] = None,
        glob: Optional[str] = None,
        startswith: Optional[Union[str, Iterable[str]]] = None,
        endswith: Optional[Union[str, Iterable[str]]] = None,
    ):
        """Drop any rows where the strings in the considered columns are empty or null.

        Args:
            trim (bool):
                If True strips blank spaces before checking if there is an empty string.
                The trimming is just for checking; the string itself in the
                resulting dataframe is NOT trimmed.
            columns (str | list(str) | None):
                List of columns to select. Defaults to None.
            regex (str | None):
                Select the columns by using a regex pattern.
                Defaults to None.
            glob (str | None):
                Select the columns by using a bash-like pattern.
                Defaults to None.
            startswith (str | iterable(str) | None):
                Select all the columns whose names start with the provided
                string(s). Defaults to None.
            endswith (str | iterable(str) | None):
                Select all the columns whose names end with the provided
                string(s). Defaults to None.
        """
        super().__init__()
        self._trim: bool = trim
        self._set_columns_selections(
            columns=columns,
            regex=regex,
            glob=glob,
            startswith=startswith,
            endswith=endswith,
        )

    def _transform(self, df):
        selection: List[str] = self._get_selected_columns(df)
        str_types = {c.name for c in df.schema if c.dataType.typeName() == "string"}
        subset: List[str] = [i for i in selection if i in str_types]

        li_prepared = [F.trim(c) if self._trim else F.col(c) for c in subset]

        # keep str != "" and non null
        li_cond = [((c != "") & c.isNotNull()) for c in li_prepared]
        return df.filter(reduce(and_, li_cond))


class DropDuplicates(Transformer):
    def __init__(
        self,
        *,
        columns: Optional[Union[str, List[str]]] = None,
        regex: Optional[str] = None,
        glob: Optional[str] = None,
        startswith: Optional[Union[str, Iterable[str]]] = None,
        endswith: Optional[Union[str, Iterable[str]]] = None,
    ):
        """Perform spark `drop_duplicates` operation.

        Input parameters are eventually used to select a subset of the columns.
        In such cases, the 'drop_duplicates_no_randomness' function is used
        to minimize randomness; otherwise, a bare 'drop_duplicates()' or
        '.distinct()' method is used.

        Args:
            columns (str | list(str) | None):
                List of the subset columns. Defaults to None.
            regex (str | None):
                Select the subset columns to select by using a regex pattern.
                Defaults to None.
            glob (str | None):
                Select the subset columns by using a bash-like pattern.
                Defaults to None.
            startswith (str | iterable(str) | None):
                Select all the subset columns whose names start with the provided
                string(s). Defaults to None.
            endswith (str | iterable(str) | None):
                Select all the subset columns whose names end with the provided
                string(s). Defaults to None.
        """
        super().__init__()
        self._set_columns_selections(
            columns=columns,
            regex=regex,
            glob=glob,
            startswith=startswith,
            endswith=endswith,
        )

    def _transform(self, df):
        subset: List[str] = self._get_selected_columns(df)
        if subset and (set(subset) != set(list(df.columns))):
            return drop_duplicates_no_randomness(df, subset)
        return df.drop_duplicates()


class Filter(Transformer):
    def __init__(
        self,
        *,
        input_col: str,
        perform: str,
        operator: str,
        value: Optional[Any] = None,
        comparison_column: Optional[str] = None,
    ):
        """Keep or remove rows according to the given conditions.

        Args:
            input_col (str):
                 Specifies the input column to be used as the filter criteria.
            perform (str):
                "keep" or "remove" rows that match the condition.
            operator (str):
                valid operators:
                - "eq":             equal
                - "le":             less equal
                - "lt":             less than
                - "ge":             greater equal
                - "gt":             greater than
                - "isin":           iterable of valid values
                - "isnotin":           iterable of valid values
                - "array_contains": has at least one instance of <value> in a <ArrayType> column
                - "contains":       has at least one instance of <value> in a <StringType> column
                - "startswith":     The row value starts with <value> in a <StringType> column
                - "endswith":       The row value ends with <value> in a <StringType> column
                - "between":        is between 2 values, lower and upper bound inclusive
                - "like":           matches a SQL LIKE pattern
                - "rlike":          matches a regex pattern
                - "isNull"          *
                - "isNotNull"       *
                - "isNaN"           *
                - "isNotNaN"        *

                    * Does not require the optional "value" argument
                    "ne" (not equal) is not allowed

                value (any | None):
                    Value used for the comparison.
                comparison_column (str | None):
                    Name of column to be compared with `input_col`.

                Either `value` or `comparison_column` (not both) must be provided
                for python operators that require a comparison value.
        """
        # ne is not allowed, it requires a specific logic to handle the None / NaN
        if operator == "ne":
            msg = '"ne" operator is disallowed since can be ambiguous for null values'
            raise AssertionError(msg)

        assert_allowed(perform, {"keep", "remove"}, "perform")
        ensure_spark_condition(operator, value, comparison_column)

        super().__init__()
        self._input_col: str = input_col
        self._perform: str = perform
        self._op: str = operator
        self._value: Optional[Any] = value
        self._compare_col: Optional[str] = comparison_column

    def _transform(self, df):
        cond = get_spark_condition(
            df,
            self._input_col,
            self._op,
            value=self._value,
            compare_col=self._compare_col,
        )

        if self._perform == "remove":
            cond = ~null_cond_to_false(cond)
        return df.filter(cond)
