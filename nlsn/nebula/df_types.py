"""Create the 'GenericDataFrame' type.

Dynamically create a Union type like:
Union[pyspark.sql.DataFrame, pandas.DataFrame, polars.DataFrame]
"""

import operator
from functools import reduce

from nlsn.nebula.backend_util import HAS_PANDAS, HAS_POLARS, HAS_SPARK

__all__ = ["GenericDataFrame", "get_dataframe_type"]

_df_types: list[type] = []

if HAS_PANDAS:
    import pandas

    _df_types.append(pandas.DataFrame)

if HAS_POLARS:
    import polars

    _df_types.append(polars.DataFrame)

if HAS_SPARK:
    import pyspark

    _df_types.append(pyspark.sql.DataFrame)

GenericDataFrame = reduce(operator.or_, _df_types)


def get_dataframe_type(df) -> str:
    """Determines the type of dataframe (pandas, polars, or spark).

    Args:
        df: The input dataframe (any supported backend).

    Returns:
        str: The type of dataframe ('pandas', 'polars', or 'spark').

    Raises:
        TypeError: If the dataframe type is unknown or unsupported.
    """
    df_module = type(df).__module__

    if "pandas" in df_module:
        return "pandas"
    elif "polars" in df_module:
        return "polars"
    elif "pyspark" in df_module:
        return "spark"

    # Fallback to isinstance checks (more reliable but requires imports)
    if HAS_PANDAS:
        from pandas import DataFrame as pandas_DF
        if isinstance(df, pandas_DF):
            return "pandas"

    if HAS_POLARS:
        from polars import DataFrame as pl_DF
        if isinstance(df, pl_DF):
            return "polars"

    if HAS_SPARK:
        from pyspark.sql import DataFrame as ps_DF
        if isinstance(df, ps_DF):
            return "spark"

    # Provide helpful error with detected type info
    raise TypeError(
        f"Unknown or unsupported dataframe type: {type(df)}. "
        f"Supported types: "
        f"{'pandas.DataFrame, ' if HAS_PANDAS else ''}"
        f"{'polars.DataFrame, ' if HAS_POLARS else ''}"
        f"{'pyspark.sql.DataFrame' if HAS_SPARK else ''}"
    )
