"""functions for TransformerPipeline."""

import narwhals as nw

from nlsn.nebula.df_types import GenericDataFrame, get_dataframe_type
from nlsn.nebula.nw_util import to_native_dataframes

__all__ = [
    "split_df",
    "to_schema",
]


def split_df(df, cfg: dict) -> tuple["GenericDataFrame", "GenericDataFrame"]:
    """Split a dataframe according to the given configuration."""
    df_type_name: str = get_dataframe_type(df)
    if df_type_name == "spark":
        from nlsn.nebula.pipelines._spark_split_functions import spark_split_function

        func = spark_split_function
    elif df_type_name == "pandas":
        from nlsn.nebula.pipelines._pandas_split_functions import pandas_split_function

        func = pandas_split_function
    elif df_type_name == "polars":
        from nlsn.nebula.pipelines._polars_split_functions import polars_split_function

        func = polars_split_function
    else:  # pragma: no cover
        raise ValueError(f"Unsupported dataframe type: {df_type_name}")

    return func(
        df,
        input_col=cfg.get("input_col"),
        operator=cfg.get("operator"),
        value=cfg.get("value"),
        compare_col=cfg.get("comparison_column"),
    )


def to_schema(li_df: list, schema) -> list["GenericDataFrame"]:
    """Cast a list of dataframes to a schema."""
    native_dataframes, native_backend, nw_found = to_native_dataframes(li_df)

    if native_backend == "pandas":
        ret = [_df.astype(schema) for _df in native_dataframes]

    elif native_backend == "polars":
        ret = [_df.cast(schema) for _df in native_dataframes]

    elif native_backend == "spark":
        from nlsn.nebula.spark_util import cast_to_schema

        ret = [cast_to_schema(i, schema) for i in native_dataframes]

    else:  # pragma: no cover
        raise ValueError(f"Unsupported dataframe type: {native_backend}")

    return [nw.from_native(i) for i in ret] if nw_found else ret
