"""Transformers for debugging."""

from typing import Iterable, List, Optional, Union

from nlsn.nebula.base import Transformer

__all__ = ["PrintSchema"]


class PrintSchema(Transformer):
    backends = {"pandas", "polars", "spark"}

    def __init__(
        self,
        *,
        columns: Optional[Union[str, List[str]]] = None,
        regex: Optional[str] = None,
        glob: Optional[str] = None,
        startswith: Optional[Union[str, Iterable[str]]] = None,
        endswith: Optional[Union[str, Iterable[str]]] = None,
    ):
        """Print out the data types of the dataframe.

        Args:
            columns (str | list(str) | None):
                A list of columns to select. Defaults to None.
            regex (str | None):
                Select the columns by using a regex pattern.
                Defaults to None.
            glob (str | None):
                Select the columns by using a bash-like pattern.
                Defaults to None.
            startswith (str | iterable(str) | None):
                Select all the columns whose names start with the provided
                string(s). Defaults to None.
            endswith (str | iterable(str) | None):
                Select all the columns whose names end with the provided
                string(s). Defaults to None.
        """
        super().__init__()
        self._set_columns_selections(
            columns=columns,
            regex=regex,
            glob=glob,
            startswith=startswith,
            endswith=endswith,
        )

    def _transform(self, df):
        return self._select_transform(df)

    def __print_pandas_polars_schema(self, df, t: str) -> None:
        selection: List[str] = self._get_selected_columns(df)
        if selection:
            df_show = df[selection]
        else:
            df_show = df

        if t == "pandas":
            schema = df_show.dtypes.to_dict()
        else:
            schema = df_show.schema

        print("Data Types:")
        for k, v in schema.items():
            print(f'"{k}": {v}')

    def _transform_pandas(self, df):
        self.__print_pandas_polars_schema(df, "pandas")
        return df

    def _transform_polars(self, df):
        self.__print_pandas_polars_schema(df, "polars")
        return df

    def _transform_spark(self, df):
        selection: List[str] = self._get_selected_columns(df)
        if selection:
            df_show = df.select(selection)
        else:
            df_show = df
        df_show.printSchema()
        return df
