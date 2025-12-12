from functools import partial, reduce
from typing import Iterable

import narwhals as nw

from nlsn.nebula.auxiliaries import assert_allowed, get_symmetric_differences_in_sets
from nlsn.nebula.df_types import get_dataframe_type

__all__ = [
    "append_dataframes",
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


def append_dataframes(
        dataframes,
        *,
        allow_missing_cols: bool,
        relax: bool = False,
        rechunk: bool = False,
        ignore_index: bool = False,
):
    """Append (concatenate vertically) a list of dataframes.

    This function handles dataframes from pandas, Polars, and Spark backends,
    with support for column mismatches and type coercion. All dataframes must
    be from the same backend (or Narwhals wrappers around the same backend).

    Args:
        dataframes (list):
            List of dataframes to concatenate vertically. Can be:
            - All native dataframes from the same backend
            - Mix of Narwhals wrappers and native frames (same underlying backend)
            - All Narwhals DataFrames/LazyFrames (same underlying backend)
        allow_missing_cols (bool):
            If True, allows column mismatches between dataframes. Missing columns
            are filled with null values. If False, raises ValueError when column
            sets don't match exactly.
            Behavior by backend:
            - pandas: Uses pd.concat naturally handles missing columns
            - Polars: Uses 'diagonal' mode to add null columns
            - Spark: Uses unionByName(allowMissingColumns=True)
        relax (bool):
            Polars-only parameter. If True, allows compatible type coercion during
            concatenation (e.g., int32 → int64, float32 → float64). Uses Polars'
            'vertical_relaxed' or 'diagonal_relaxed' modes. Ignored for pandas and
            Spark. Defaults to False.
        rechunk (bool):
            Polars-only parameter. If True, rechunks the concatenated result for
            better memory layout and performance. Ignored for pandas and Spark.
            Defaults to False.
        ignore_index (bool):
            Pandas-only parameter. If True, do not preserve the original index
            values when concatenating. Ignored for Polars (no index) and Spark.
            Defaults to False.

    Returns:
        DataFrame: Concatenated dataframe in the appropriate format:
            - Returns Narwhals DataFrame/LazyFrame if any input is Narwhals
            - Returns native dataframe if all inputs are native
            - Always uses the same backend as the input dataframes

    Raises:
        ValueError:
            If dataframes list is empty.
        TypeError:
            If multiple different backends are detected (e.g., pandas + Polars).
        ValueError:
            If column mismatch is found and allow_missing_cols=False.
        TypeError:
            If an unsupported dataframe backend is detected.
        Native Exception: If concatenating empty dfs without schema information.
            Note: Empty dataframes may cause schema inference errors. Filter them
            upstream or ensure they have explicit schemas before concatenation.

    Notes:
        - Backend-specific parameters (rechunk, relax, ignore_index) are silently
          ignored when not applicable to the current backend.
        - Type mismatches between dataframes are handled according to each backend's
          default behavior unless relax=True for Polars.
        - When mixing Narwhals wrappers with native frames, all must have the same
          underlying backend (e.g., all Polars, or all pandas).
        - Empty dataframes may cause schema inference errors in some backends,
          especially when concatenating multiple empty dataframes or when column
          types cannot be inferred. Users should filter empty dataframes upstream
          or ensure schemas are explicitly defined.

    Examples:
        >>> # Simple concatenation with matching columns
        >>> result = append_dataframes([df1, df2, df3])

        >>> # Allow missing columns (fills with nulls)
        >>> result = append_dataframes(
        ...     [df_with_cols_abc, df_with_cols_ab],
        ...     allow_missing_cols=True
        ... )

        >>> # Polars with type relaxation (int32 → int64)
        >>> result = append_dataframes(
        ...     [pl_df_int32, pl_df_int64],
        ...     relax=True
        ... )

        >>> # Pandas without preserving index
        >>> result = append_dataframes(
        ...     [pd_df1, pd_df2],
        ...     ignore_index=True
        ... )

        >>> # Mix Narwhals wrapper with native (returns Narwhals)
        >>> nw_df = nw.from_native(pl_df1)
        >>> result = append_dataframes([nw_df, pl_df2, pl_df3])
        >>> # result is nw.DataFrame wrapping Polars

        >>> # Polars with rechunking for better memory layout
        >>> result = append_dataframes(
        ...     [pl_df1, pl_df2, pl_df3],
        ...     rechunk=True
        ... )
    """
    if not dataframes:
        raise ValueError("Cannot append empty list of dataframes")

    if len(dataframes) == 1:
        return dataframes[0]

    to_native: bool = True
    native_dataframes = []
    sets_columns: list[set[str]] = []
    full_columns: set[str] = set()
    backends = set()
    for df in dataframes:
        if isinstance(df, (nw.DataFrame, nw.LazyFrame)):
            to_native = False
            df_native = nw.to_native(df)
        else:
            df_native = df

        native_dataframes.append(df_native)
        backends.add(get_dataframe_type(df_native))
        sets_columns.append(set(df_native.columns))
        full_columns.update(set(df_native.columns))

    n_backends = len(backends)

    if n_backends > 1:
        raise TypeError(f"Mixed backed found: {n_backends}")

    native_backend = backends.pop()

    diff: set[str] = get_symmetric_differences_in_sets(*sets_columns)

    # Handle the error
    if diff and not allow_missing_cols:
        msg = []
        for i, right_df in enumerate(native_dataframes[1:], start=1):
            left_df = native_dataframes[i - 1]
            col_left = set(right_df.columns)
            col_right = set(left_df.columns)
            missing_left = full_columns - col_left
            missing_right = full_columns - col_right
            if missing_left:
                msg.append(f"missing in df({i}): {sorted(missing_left)}")
            if missing_right:
                msg.append(f"missing in df({i}): {sorted(missing_right)}")

        raise ValueError(
            "Column mismatch between dataframes -> " + ", ".join(msg)
        )

    if native_backend == "pandas":
        import pandas as pd
        ret = pd.concat(native_dataframes, axis=0, ignore_index=ignore_index)

    elif native_backend == "polars":
        import polars as pl
        if allow_missing_cols:
            how = "diagonal"
        else:
            cols = native_dataframes[0].columns
            native_dataframes = [df.select(cols) for df in native_dataframes]
            how = "vertical"
        how += "_relaxed" if relax else ""
        ret = pl.concat(native_dataframes, rechunk=rechunk, how=how)

    elif native_backend == "spark":
        from pyspark.sql import DataFrame
        func = partial(DataFrame.unionByName, allowMissingColumns=allow_missing_cols)
        ret = reduce(func, native_dataframes)

    else:  # pragma: no cover
        raise TypeError(f"Unsupported dataframe type: {native_backend}")

    return ret if to_native else nw.from_native(ret)


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
