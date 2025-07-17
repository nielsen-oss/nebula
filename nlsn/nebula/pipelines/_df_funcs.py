"""functions for TransformerPipeline."""

from functools import partial, reduce
from typing import List, Optional, Tuple

from nlsn.nebula.auxiliaries import get_symmetric_differences_in_sets
from nlsn.nebula.df_types import GenericDataFrame, get_dataframe_type

__all__ = [
    "append_df",
    "df_is_empty",
    "join_dfs",
    "split_df",
    "to_schema",
]


def _ensure_same_cols_in_dfs(li_df: List["GenericDataFrame"]) -> None:
    sets = []
    for df in li_df:
        sets.append(set(df.columns))

    diff = get_symmetric_differences_in_sets(*sets)
    if diff:
        raise ValueError(f"Different columns found: {diff}")


def append_df(
    li_df: List["GenericDataFrame"], allow_missing_cols: bool
) -> "GenericDataFrame":
    """Append a list of dataFrames."""
    df_type_name: str = get_dataframe_type(li_df[0])
    if df_type_name == "spark":
        from pyspark.sql import DataFrame

        func = partial(DataFrame.unionByName, allowMissingColumns=allow_missing_cols)
        return reduce(func, li_df)

    elif df_type_name == "pandas":
        import pandas as pd

        if not allow_missing_cols:
            _ensure_same_cols_in_dfs(li_df)
        return pd.concat(li_df, axis=0)

    elif df_type_name == "polars":
        import polars as pl

        if allow_missing_cols:
            return pl.concat(li_df, rechunk=True, how="diagonal")
        else:
            _ensure_same_cols_in_dfs(li_df)
            cols = li_df[0].columns
            li_df_consistent = [_df[cols] for _df in li_df]
            return pl.concat(li_df_consistent, rechunk=True)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported dataframe type: {df_type_name}")


def df_is_empty(df) -> bool:
    """Check whether a dataframe is empty."""
    df_type_name: str = get_dataframe_type(df)
    if df_type_name == "spark":
        return df.rdd.isEmpty()
    elif df_type_name == "pandas":
        return df.shape[0] == 0
    elif df_type_name == "polars":
        return df.shape[0] == 0
    else:  # pragma: no cover
        raise ValueError(f"Unsupported dataframe type: {df_type_name}")


def join_dfs(
    df_left, df_right, on, how: str, *, broadcast: Optional[bool] = None
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


def split_df(df, cfg: dict) -> Tuple["GenericDataFrame", "GenericDataFrame"]:
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


def to_schema(li_df: list, schema) -> List["GenericDataFrame"]:
    """Cast a list of dataframes to a schema."""
    df_type_name: str = get_dataframe_type(li_df[0])
    if df_type_name == "spark":
        from nlsn.nebula.spark_util import cast_to_schema

        return [cast_to_schema(i, schema) for i in li_df]
    elif df_type_name == "pandas":
        return [_df.astype(schema) for _df in li_df]
    elif df_type_name == "polars":
        try:  # Old polars version hasn't 'cast' method
            return [_df.cast(schema) for _df in li_df]
        except AttributeError:
            import polars as pl

            new_list = []
            for _df in li_df:
                _df_cast = _df.with_columns(
                    [pl.col(k).cast(v) for k, v in schema.items()]
                )
                new_list.append(_df_cast)
            return new_list

    else:  # pragma: no cover
        raise ValueError(f"Unsupported dataframe type: {df_type_name}")
