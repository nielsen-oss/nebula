"""Function to split a Pandas dataframe in two subsets."""

import operator as py_operator
from typing import Any, Optional, Tuple

import pandas as pd

from nlsn.nebula.auxiliaries import assert_allowed, assert_only_one_non_none

__all__ = ["pandas_split_function"]

_VALID_OPERATORS = {
    "isNull",
    "isNotNull",
    "isNaN",
    "isNotNaN",
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


def pandas_split_function(
    df: "pd.DataFrame",
    input_col: str,
    operator: str,
    value: Any,
    compare_col: Optional[str],
) -> Tuple["pd.DataFrame", "pd.DataFrame"]:
    """Split a dataframe into two dataframes given a certain condition.

    Args:
        df (DataFrame): The input Pandas DataFrame.
        input_col (str): The column to apply the condition on.
        operator (str): The comparison operator.
            - "eq", "ne", "le", "lt", "ge", "gt" (equality, greater, lower)
            - "isNull", "isNotNull", "isNaN", "isNotNaN": look for null values
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

    s: pd.Series = df[input_col]
    msk: pd.Series
    if operator in {"isNull", "isNotNull", "isNaN", "isNotNaN"}:
        msg_err = "With null / nan operator the 'value' must not be provided"
        if value is not None:
            raise AssertionError(msg_err)
        if compare_col is not None:
            raise AssertionError(msg_err)
        msk = s.isna()
        if operator in {"isNotNull", "isNotNaN"}:
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
                msk = s.isin(value)
            elif operator in {"contains", "startswith", "endswith"}:
                msk = getattr(s.str, operator)(value)
            else:  # pragma: no cover
                raise ValueError(f"Unknown operator '{operator}'")
    return df[msk], df[~msk]
