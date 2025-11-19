"""Create the 'GenericDataFrame' type.

Dynamically create a Union type like:
Union[pyspark.sql.dataframe.DataFrame, pandas.core.frame.DataFrame]
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

# # Use reduce for Python 3.9+, fallback to eval for earlier versions
# try:
#     GenericDataFrame = reduce(operator.or_, _df_types)
# except TypeError:
#     _eval_str = ", ".join([f"{i.__module__}.{i.__name__}" for i in _df_types])
#     GenericDataFrame = eval(f"Union[{_eval_str}]")

GenericDataFrame = reduce(operator.or_, _df_types)


def get_dataframe_type(df: GenericDataFrame) -> str:
    """Determines the type of dataframe (pandas, polars, or spark).

    Args:
        df (GenericDataFrame): The input dataframe.

    Returns:
        str: The type of dataframe (pandas, polars, or spark).

    Raises:
        TypeError: If the dataframe type is unknown.
    """
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

    raise TypeError("Unknown dataframe type")
