"""functions for TransformerPipeline."""

from nlsn.nebula.df_types import GenericDataFrame, get_dataframe_type

__all__ = [
    "split_df",
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
