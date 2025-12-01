"""Row Filtering Operations."""

from nlsn.nebula.auxiliaries import assert_allowed
from nlsn.nebula.base import Transformer
from nlsn.nebula.spark_util import (
    ensure_spark_condition,
    get_spark_condition,
    null_cond_to_false,
)

__all__ = [
    "Filter",
]


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
