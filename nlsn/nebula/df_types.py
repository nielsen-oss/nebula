"""Create the 'GenericDataFrame' type.

Dynamically create a Union type like:
Union[pyspark.sql.DataFrame, pandas.DataFrame, polars.DataFrame]
"""

import operator
from functools import reduce
from typing import Any

import narwhals as nw

from nlsn.nebula.backend_util import HAS_PANDAS, HAS_POLARS, HAS_SPARK

__all__ = [
    "GenericNativeDataFrame",
    "GenericDataFrame",
    "NwDataFrame",
    "get_dataframe_type",
    "is_natively_spark",
]

NwDataFrame = nw.DataFrame | nw.LazyFrame

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

# Handle case where no backends are installed
if _df_types:
    GenericNativeDataFrame = reduce(operator.or_, _df_types)
else:
    # Fallback type when no backends installed
    # This allows imports to work, but any usage will fail with helpful message
    GenericNativeDataFrame = Any

GenericDataFrame = NwDataFrame | GenericNativeDataFrame


def get_dataframe_type(df) -> str:
    """Determines the type of dataframe (pandas, polars, or spark).

    Args:
        df: The input dataframe (any supported backend).

    Returns:
        str: The type of dataframe ('pandas', 'polars', or 'spark').

    Raises:
        TypeError: If the dataframe type is unknown or unsupported.
    """
    # Fast path: check module name (avoids imports)
    df_module = type(df).__module__

    if "pandas" in df_module:
        return "pandas"
    elif "polars" in df_module:
        return "polars"
    elif "pyspark" in df_module:
        return "spark"

    # Fallback: isinstance checks (more reliable but requires imports)
    if HAS_PANDAS:
        from pandas import DataFrame as pandas_DF
        if isinstance(df, pandas_DF):
            return "pandas"

    if HAS_POLARS:
        import polars as pl
        if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
            return "polars"

    if HAS_SPARK:
        from pyspark.sql import DataFrame as ps_DF
        if isinstance(df, ps_DF):
            return "spark"

    # Build a helpful error message
    supported = []
    if HAS_PANDAS:
        supported.append("pandas.DataFrame")
    if HAS_POLARS:
        supported.append("polars.DataFrame")
    if HAS_SPARK:
        supported.append("pyspark.sql.DataFrame")

    if not supported:
        raise TypeError(
            f"Unknown dataframe type: {type(df)}. "
            "No supported backends are installed. "
            "Install at least one: pip install pandas|polars|pyspark"
        )

    raise TypeError(
        f"Unknown dataframe type: {type(df)}. "
        f"Supported types: {', '.join(supported)}"
    )


def is_natively_spark(df) -> bool:
    if HAS_SPARK:
        if isinstance(df, (nw.DataFrame, nw.LazyFrame)):
            df_native = nw.from_native(df)
        else:
            df_native = df

        return get_dataframe_type(df_native) == "spark"
    return False
