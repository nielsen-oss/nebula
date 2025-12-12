from typing import Iterable

import narwhals as nw

from nlsn.nebula.auxiliaries import assert_allowed
from nlsn.nebula.df_types import get_dataframe_type

__all__ = [
    "df_is_empty",
    "get_condition",
    "null_cond_to_false",
    "validate_operation",
]

COMPARISON_OPERATORS = {"eq", "ne", "le", "lt", "ge", "gt"}
NULL_OPERATORS = {"is_null", "is_not_null", "is_nan", "is_not_nan"}
STRING_OPERATORS = {"contains", "starts_with", "ends_with"}
MEMBERSHIP_OPERATORS = {"is_between", "is_in", "is_not_in"}
_allowed_operators = (
        COMPARISON_OPERATORS
        | NULL_OPERATORS
        | STRING_OPERATORS
        | MEMBERSHIP_OPERATORS
)


def df_is_empty(df_input) -> bool:
    """Check whether a dataframe is empty."""
    if isinstance(df_input, (nw.DataFrame, nw.LazyFrame)):
        df = nw.to_native(df_input)
    else:
        df = df_input
    df_type_name: str = get_dataframe_type(df)
    if df_type_name == "pandas":
        return df.empty
    elif df_type_name == "polars":
        import polars as pl
        if isinstance(df, pl.LazyFrame):
            return df.limit(1).collect().is_empty()
        return df.is_empty()
    elif df_type_name == "spark":
        return df.isEmpty()
    else:  # pragma: no cover
        raise ValueError(f"Unsupported dataframe type: {df_type_name}")


def null_cond_to_false(cond: nw.Expr) -> nw.Expr:
    """Convert null values in a boolean condition to False.

    This is useful when you want null comparisons to be treated as False
    rather than propagating null through boolean logic.

    Args:
        cond: A boolean expression that may contain nulls.

    Returns:
        Expression where nulls are replaced with False.
    """
    return nw.when(cond.is_null()).then(nw.lit(False)).otherwise(cond)


def validate_operation(
        operator: str,
        value=None,
        compare_col: str | None = None,
) -> None:
    """Validate the input parameters for a filter condition.

    This function checks if the provided `operator`, `value`, and `compare_col` are
    valid for constructing a Narwhals condition. It ensures that the operator is
    supported, that the correct arguments are provided based on the operator,
    and that the values are of the expected types.

    Args:
        operator: The comparison operator to use. Valid operators include:
            - "eq", "ne", "le", "lt", "ge", "gt": Standard comparisons
            - "is_null", "is_not_null", "is_nan", "is_not_nan": Null/NaN checks
            - "is_in", "is_not_in": Set membership
            - "is_between": Range check (inclusive)
            - "contains", "starts_with", "ends_with": String matching
        value: The value to compare against. Required for operators that compare
            against a specific value. Cannot be used with `compare_col`.
        compare_col: The name of the column to compare against. Required for
            operators that compare two columns. Cannot be used with `value`.

    Raises:
        ValueError: If operator is invalid or incompatible args are provided.
        TypeError: If value has wrong type for the operator.
    """
    assert_allowed(operator, _allowed_operators, "operator")

    # Null/NaN operators don't need value or compare_col
    if operator in NULL_OPERATORS:
        if value is not None or compare_col is not None:
            raise ValueError(
                f"Operator '{operator}' does not accept 'value' or 'compare_col'"
            )
        return

    # All other operators need exactly one of value/compare_col
    if (value is not None) and (compare_col is not None):
        raise ValueError(
            "Exactly one of 'value' or 'compare_col' must be provided"
        )

    if (value is None) and (compare_col is None):
        raise ValueError(
            f"Operator '{operator}' requires either 'value' or 'compare_col'"
        )

    # Standard comparisons accept either value or compare_col - no extra validation
    if operator in COMPARISON_OPERATORS:
        return

    # Some operators don't support column comparison
    if operator in {"is_between"} and compare_col is not None:
        raise ValueError(
            f"Operator '{operator}' does not support column comparison"
        )

    # Type-specific validations (only when value is provided)
    if value is None:
        return  # compare_col is provided instead, skip type checks

    if operator in {"is_in", "is_not_in"}:
        if isinstance(value, str):
            raise TypeError(
                f"Operator '{operator}' requires an iterable, not a string. "
                "Use 'contains' for substring matching."
            )
        if not isinstance(value, Iterable):
            raise TypeError(
                f"Operator '{operator}' requires an iterable (list, tuple, set)"
            )
        if None in value:
            raise ValueError(
                f"Operator '{operator}' does not handle None values in the iterable"
            )

    elif operator == "is_between":
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                "Operator 'is_between' requires a list or tuple of [lower, upper]"
            )
        if len(value) != 2:
            raise ValueError(
                "Value for 'is_between' must have exactly 2 elements: [lower, upper]"
            )

    elif operator in STRING_OPERATORS:
        if not isinstance(value, str):
            raise TypeError(
                f"Operator '{operator}' requires a string value"
            )


def get_condition(
        col_name: str,
        operator: str,
        *,
        value=None,
        compare_col: str | None = None,
) -> nw.Expr:
    """Build a Narwhals boolean expression for filtering.

    This function constructs boolean conditions that work across
    pandas, Polars, and Spark via the Narwhals API.

    Args:
        col_name: Name of the column to filter on.
        operator: Comparison operator. Valid operators:
            - "eq", "ne", "le", "lt", "ge", "gt": Standard comparisons
            - "is_null", "is_not_null", "is_nan", "is_not_nan": Null/NaN checks
            - "is_in", "is_not_in": Set membership
            - "is_between": Range check (inclusive)
            - "contains", "starts_with", "ends_with": String matching
        value: Value to compare against. Required for most operators.
            Cannot be used with `compare_col`.
        compare_col: Name of column to compare against. Allows column-to-column
            comparisons. Cannot be used with `value`.

    Returns:
        A Narwhals boolean expression suitable for use in df.filter().

    Examples:
        >>> # Simple value comparison
        >>> cond = get_condition("age", "gt", value=18)
        >>> df.filter(cond)

        >>> # Column-to-column comparison
        >>> cond = get_condition("sales", "gt", compare_col="target")
        >>> df.filter(cond)

        >>> # String matching
        >>> cond = get_condition("name", "starts_with", value="J")
        >>> df.filter(cond)

        >>> # Set membership
        >>> cond = get_condition("status", "is_in", value=["active", "pending"])
        >>> df.filter(cond)

        >>> # Range check
        >>> cond = get_condition("score", "is_between", value=[0, 100])
        >>> df.filter(cond)
    """
    validate_operation(operator, value=value, compare_col=compare_col)

    col = nw.col(col_name)

    # Null/NaN checks
    if operator in NULL_OPERATORS:
        if operator == "is_null":
            return col.is_null()
        elif operator == "is_not_null":
            return ~col.is_null()
        elif operator == "is_nan":
            return null_cond_to_false(col.is_nan())
        else:  # is_not_nan
            return ~null_cond_to_false(col.is_nan())

    # String operations
    if operator in STRING_OPERATORS:
        if operator == "contains":
            return col.str.contains(value)
        elif operator == "starts_with":
            return col.str.starts_with(value)
        else:  # ends_with
            return col.str.ends_with(value)

    # Membership operations
    if operator in MEMBERSHIP_OPERATORS:
        if operator == "is_in":
            return col.is_in(value)
        elif operator == "is_not_in":
            # Use null_cond_to_false to handle nulls like Spark does
            return ~null_cond_to_false(col.is_in(value))
        else:  # is_between
            lower, upper = value
            return col.is_between(lower, upper)

    # Standard comparisons
    # Get what we're comparing against (value or column)
    comparator = nw.col(compare_col) if compare_col else value

    if operator == "eq":
        return col == comparator
    elif operator == "ne":
        return col != comparator
    elif operator == "lt":
        return col < comparator
    elif operator == "le":
        return col <= comparator
    elif operator == "gt":
        return col > comparator
    else:  # ge
        return col >= comparator
