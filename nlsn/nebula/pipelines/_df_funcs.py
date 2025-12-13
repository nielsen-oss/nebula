"""functions for TransformerPipeline."""

import narwhals as nw

from nlsn.nebula.df_types import GenericDataFrame, get_dataframe_type

__all__ = [
    "join_dfs",
    "split_df",
    "to_schema",
]

from nlsn.nebula.nw_util import to_native_dataframes


def join_dfs(
        df_left, df_right, on, how: str, *, broadcast: bool | None = None
) -> "GenericDataFrame":
    """Join two dataframes."""
    df_type_name: str = get_dataframe_type(df_left)
    if df_type_name == "spark":
        if broadcast:
            from pyspark.sql.functions import broadcast

            df_right = broadcast(df_right)
        return df_left.join(df_right, on=on, how=how)
    elif df_type_name == "pandas":
        return df_left.merge(df_right, on=on, how=how)
    elif df_type_name == "polars":
        return df_left.join(df_right, on=on, how=how)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported dataframe type: {df_type_name}")


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

    if native_backend == "spark":
        from nlsn.nebula.spark_util import cast_to_schema

        ret = [cast_to_schema(i, schema) for i in li_df]

    elif native_backend == "pandas":
        ret = [_df.astype(schema) for _df in li_df]

    elif native_backend == "polars":
        try:  # Old polars version hasn't 'cast' method
            ret = [_df.cast(schema) for _df in li_df]
        except AttributeError:
            import polars as pl

            ret = []
            for _df in li_df:
                _df_cast = _df.with_columns(
                    [pl.col(k).cast(v) for k, v in schema.items()]
                )
                ret.append(_df_cast)

    else:  # pragma: no cover
        raise ValueError(f"Unsupported dataframe type: {native_backend}")

    return [nw.from_native(i) for i in ret] if nw_found else ret
