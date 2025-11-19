"""Row Filtering Operations."""

from typing import Iterable

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
    "DropDuplicates",
    "Filter",
]


class DiscardNulls(Transformer):
    def __init__(
            self,
            *,
            how: str,
            columns: str | list[str] | None = None,
            regex: str | None = None,
            glob: str | None = None,
            startswith: str | Iterable[str] | None = None,
            endswith: str | Iterable[str] | None = None,
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
        subset: list[str] = self._get_selected_columns(df)
        if subset and set(subset) != set(list(df.columns)):
            return df.dropna(self._how, subset=subset)
        return df.dropna(self._how)


class DropDuplicates(Transformer):
    def __init__(
            self,
            *,
            columns: str | list[str] | None = None,
            regex: str | None = None,
            glob: str | None = None,
            startswith: str | Iterable[str] | None = None,
            endswith: str | Iterable[str] | None = None,
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
        subset: list[str] = self._get_selected_columns(df)
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
            value=None,
            comparison_column: str | None = None,
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
        self._value = value
        self._compare_col: str | None = comparison_column

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
