"""Function to split a Polars dataframe in two subsets."""

import operator as py_operator
from typing import Any, Dict, Optional, Tuple

import polars as pl

from nlsn.nebula.auxiliaries import assert_allowed, assert_only_one_non_none

__all__ = ["polars_split_function"]

_VALID_NULL_OPERATORS = {
    "isNull",
    "isNotNull",
    "isNaN",
    "isNotNaN",
    "isNullOrNaN",
    "isNotNullOrNaN",
}

_STRINGS_METHODS: Dict[str, str] = {
    "contains": "contains",
    "startswith": "starts_with",
    "endswith": "ends_with",
}

_VALID_OPERATORS = _VALID_NULL_OPERATORS.union(
    {
        "eq",
        "ne",
        "le",
        "lt",
        "ge",
        "gt",
        "isin",
        "contains",
        "startswith",
        "endswith",
    }
)


def polars_split_function(
    df: "pl.DataFrame",
    input_col: str,
    operator: str,
    value: Any,
    compare_col: Optional[str],
) -> Tuple["pl.DataFrame", "pl.DataFrame"]:
    """Split a dataframe into two dataframes given a certain condition.

    Args:
        df (DataFrame): The input Polars DataFrame.
        input_col (str): The column to apply the condition on.
        operator (str): The comparison operator.
            Valid ones:
            - "eq", "ne", "le", "lt", "ge", "gt" (equality, greater, lower)
            - "isNull", "isNotNull", "isNullOrNaN", "isNotNullOrNaN",
                "isNaN", "isNotNaN": look for null / nan values
            - "isin": check if a value is in an iterable
            - "contains": look for a substring in a <StringType> Column
            - "startswith": look for a string that starts with.
            - "endswith": look for a string that ends with.
        value (Any): The value to compare against.
        compare_col (str | None): An optional column for comparison.

    Returns:
        tuple(DataFrame, DataFrame): A tuple containing two DataFrames:
            1. Rows that satisfy the condition.
            2. Rows that do not satisfy the condition.
    """
    assert_allowed(operator, _VALID_OPERATORS, "operator")

    s: pl.Series = df[input_col]
    msk: pl.Series
    if operator in _VALID_NULL_OPERATORS:
        if value is not None:
            raise AssertionError
        if compare_col is not None:
            raise AssertionError

        if "Null" in operator:
            msk = s.is_null()
            if "NaN" in operator:
                msk |= s.is_nan()
        else:
            msk = s.is_nan()

        if operator.startswith("isNot"):
            msk = ~msk

    else:
        assert_only_one_non_none(value, compare_col)

        if operator in {"eq", "ne", "le", "lt", "ge", "gt"}:
            if value is not None:
                cmp = value
            else:
                cmp = df[compare_col]

            msk = getattr(py_operator, operator)(s, cmp)

        else:
            if value is None:  # pragma: no cover
                raise AssertionError
            if operator == "isin":
                msk = s.is_in(value)
            elif operator in {"contains", "startswith", "endswith"}:  # Strings
                meth: str = _STRINGS_METHODS[operator]
                msk = getattr(s.str, meth)(value)
            else:  # pragma: no cover
                raise ValueError

        # Set to <False> all None in the boolean mask
        msk = msk.fill_null(False)
    return df.filter(msk), df.filter(~msk)
