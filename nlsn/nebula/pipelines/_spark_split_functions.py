"""Function to split a Spark dataframe in two subsets."""

from typing import Any, Optional, Tuple

from pyspark.sql import DataFrame

from nlsn.nebula.spark_util import get_spark_condition, null_cond_to_false

__all__ = ["spark_split_function"]


def spark_split_function(
    df: "DataFrame",
    input_col: str,
    operator: str,
    value: Any,
    compare_col: Optional[str],
) -> Tuple["DataFrame", "DataFrame"]:
    """Split a dataframe into two dataframes given a certain condition.

    Args:
        df (DataFrame): The input Spark DataFrame.
        input_col (str): The column to apply the condition on.
        operator (str): The comparison operator.
        value (Any): The value to compare against.
        compare_col (str | None): An optional column for comparison.

    Returns:
        tuple(DataFrame, DataFrame): A tuple containing two DataFrames:
            1. Rows that satisfy the condition.
            2. Rows that do not satisfy the condition.
    """
    cond = get_spark_condition(
        df, input_col, operator, value=value, compare_col=compare_col
    )
    otherwise_cond = ~null_cond_to_false(cond)
    return df.filter(cond), df.filter(otherwise_cond)
