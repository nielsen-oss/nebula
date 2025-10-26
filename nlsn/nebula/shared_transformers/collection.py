"""General Purposes Transformers."""

import itertools
from typing import Any, Optional

from nlsn.nebula import nebula_storage as ns
from nlsn.nebula.auxiliaries import assert_allowed
from nlsn.nebula.base import Transformer

__all__ = [
    "CreateDataFrame",
    "StoreColumnNames",
    "StoreColumnValues",
    "WithColumn",
]


class CreateDataFrame(Transformer):
    backends = {"pandas", "polars", "spark"}

    def __init__(
        self,
        *,
        data: Any,
        storage_key: Optional[str] = None,
        broadcast: bool = False,
        kwargs: Optional[dict] = None,
    ):
        """Create a DataFrame using the same backend as the input one with the provided data.

        Args:
            data (Any):
                The input data, which can be in any format compatible with the
                chosen backend method:
                - pandas.DataFrame
                - spark.createDataFrame
                - polars.DataFrame
            storage_key (str | None):
                If provided, stores the data with the given key and returns
                the original input DataFrame. If not provided, returns the
                newly created DataFrame and drops the input DataFrame.
                Defaults to None.
            broadcast (bool):
                If the backend is "spark", the DataFrame will be broadcast.
                Ignored for other backends. Defaults to False.
            kwargs (dict | None):
                Additional keyword arguments compatible with the chosen
                backend method:
                - pandas.DataFrame
                - spark.createDataFrame
                - polars.DataFrame
                Defaults to None.
        """
        super().__init__()

        data = data or []
        self._data: Any = data
        self._storage_key: Optional[str] = storage_key
        self._broadcast: bool = broadcast
        self._kwargs: dict = kwargs or {}

    @staticmethod
    def _from_dict_of_list_to_list_of_dicts(data: dict) -> list[dict]:
        lengths = {len(v) for v in data.values()}
        if len(lengths) != 1:
            raise ValueError(f"Found multiple lengths in the input data: {lengths}")

        keys = list(data.keys())
        n: int = list(lengths)[0]

        if n == 0:
            return []

        ret = []
        for i in range(n):
            ret.append({k: data[k][i] for k in keys})
        return ret

    def _transform(self, df):
        df_created = self._select_transform(df)
        if self._storage_key:
            ns.set(self._storage_key, df_created)
            return df
        return df_created

    def _transform_pandas(self, _df):
        import pandas as pd

        return pd.DataFrame(self._data, **self._kwargs)

    def _transform_polars(self, _df):
        import polars as pl

        return pl.DataFrame(self._data, **self._kwargs)

    def _transform_spark(self, df):
        from nlsn.nebula.spark_util import get_spark_session

        if isinstance(self._data, dict):
            values = list(self._data.values())[0]
            if isinstance(values, (list, tuple)):
                self._data = self._from_dict_of_list_to_list_of_dicts(self._data)

        ss = get_spark_session(df)
        df_created = ss.createDataFrame(self._data, **self._kwargs)
        if self._broadcast:
            from pyspark.sql.functions import broadcast

            df_created = broadcast(df_created)
        return df_created


class StoreColumnNames(Transformer):
    backends = {"pandas", "polars", "spark"}

    def __init__(self, *, key: str):
        """Store column names into nebula storage under a given key."""
        super().__init__()
        self._key: str = key

    def _transform(self, df):
        columns = list(df.columns)
        ns.set(self._key, columns)
        return df


class StoreColumnValues(Transformer):
    backends = {"pandas", "polars", "spark"}

    def __init__(
        self,
        *,
        column: str,
        key: str,
        as_type: str = "list",
        sort: bool = False,
    ):
        """Store a specified column's data into nebula storage under a given key.

        It extracts a column from the input DataFrame, converts it to a
        specified Python type (list, array, set, frozenset), and then
        stores it in the nebula storage.
        Optionally, the stored data can be sorted.
        The input dataframe returns unchanged.

        Args:
            column (str):
                The name of the column to be extracted and stored.
            key (str):
                The key under which the extracted data will be stored in the namespace.
            as_type (str, optional):
                The desired Python type for the stored data.
                Must be one of "list", "array" (for numpy array),
                "set", or "frozenset". Defaults to "list".
            sort (bool, optional):
                If True, the stored data will be sorted.
                Sorting is not supported for "set" or "frozenset" types.
                Defaults to False.

        Raises:
            ValueError: If `as_type` is not one of the allowed types.
            ValueError: If `sort` is True and `as_type` is "set" or "frozenset".
        """
        assert_allowed(as_type, {"array", "list", "set", "frozenset"}, "as_type")

        if sort and (as_type in {"set", "frozenset"}):
            raise ValueError("Cannot sort a set.")

        super().__init__()
        self._column: str = column
        self._key: str = key
        self._as_type: str = as_type
        self._sort: bool = sort

    def __store(self, s) -> None:
        if self._as_type == "array":
            store = s.to_numpy()
        else:
            store = s.to_list()
        if self._as_type == "array":
            if self._sort:
                import numpy as np

                store = np.sort(store)
        else:
            if self._sort:
                store = sorted(store)
            elif self._as_type == "set":
                store = set(store)
            elif self._as_type == "frozenset":
                store = frozenset(store)

        ns.set(self._key, store)

    def _transform(self, df):
        series = self._select_transform(df)
        self.__store(series)
        return df

    def _transform_pandas(self, df):
        return df[self._column]

    def _transform_polars(self, df):
        return df.select(self._column).to_series()

    def _transform_spark(self, df):
        return df.select(self._column).toPandas()[self._column]


class WithColumn(Transformer):
    backends = {"pandas", "polars", "spark"}

    def __init__(
        self,
        *,
        column_name: str,
        value: Optional[Any] = None,
        cast: Optional[Any] = None,
        copy: bool = True,
    ):
        """Add a column to the DataFrame.

        If the column already exists, it will be replaced.

        Args:
            column_name (str):
                Column name.
            value (any | None):
                Value to add, if not passed the default is null-value.
                Backend-specific type restrictions:
                - Pandas:
                    All types are allowed.
                - Polars:
                    Only scalar types are allowed.
                - Spark:
                    Can be a scalar value or a non-nested tuple,
                    list, or dictionary. Tuples and lists will be
                    converted to ArrayType, dictionaries to MapType.
                    The <set> Python type is not implemented.
                Defaults to None.
            cast (str | pyspark.sql.types.DataType | numpy.dtype | None):
                If the provided type is a Python object (e.g., numpy.int32,
                pyspark.sql.types.StringType(), or polars.Int64), any
                atomic type can be passed.
                If 'cast' is a string, refer to the following conventions:
                Pandas:
                    All the string conventions like "int32", "object",
                    "float64", etc. are allowed.
                Polars:
                    Polars cast as string are not supported yet, pass
                    directly the polars type (e.g., pl.Int64).
                Spark:
                    Any atomic type like
                        - "string"
                        - "int"
                        - "integer"
                        - "float"
                        - "double"
                        - "timestamp"
                        - ...
                    or a non-nested MapType / ArrayType:
                        - "map<string, int>"
                        - "array<float>"
                Defaults to None.
            copy (bool):
                Valid for Pandas dataframes only, ignored for Spark dataframes.
                If True, copy the DataFrame before adding the column.
                Defaults to True.
        """
        super().__init__()
        self._name: str = column_name
        self._value: Optional[Any] = value
        self._cast: Optional[Any] = cast
        self._copy: bool = copy

    @staticmethod
    def _check_spark_cast(cast):
        if cast is None:
            return
        cast = cast.strip()
        if cast.lower().startswith("set"):
            raise ValueError("<set> type ('cast') is not allowed.")
        if cast.endswith(">>"):
            msg = f"Nested complex ('cast') types are not allowed. {cast}"
            raise ValueError(msg)

    @staticmethod
    def _check_spark_value(value):
        if value is None:
            return
        if isinstance(value, set):
            raise ValueError("<set> type ('value') is not allowed.")
        not_allowed = (dict, list, tuple, set)
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return
            v = value[0]
            if isinstance(v, not_allowed):
                msg = f"Nested complex ('value') types are not allowed. {value}"
                raise ValueError(msg)
        elif isinstance(value, dict):
            values = list(value.values())
            if any(isinstance(i, not_allowed) for i in values):
                raise ValueError(f"Nested value(s) not allowed. {values}")

    def _transform(self, df):
        return self._select_transform(df)

    def _transform_pandas(self, df):
        ret = df.copy() if self._copy else df
        ret[self._name] = self._value
        if self._cast:
            ret[self._name] = ret[self._name].astype(self._cast)
        return ret

    def _transform_polars(self, df):
        import polars as pl

        value = pl.lit(self._value)
        if self._cast:
            value = value.cast(self._cast)
        return df.with_columns(value.alias(self._name))

    def _transform_spark(self, df):
        from pyspark.sql import functions as F

        self._check_spark_cast(self._cast)
        self._check_spark_value(self._value)

        spark_value: F.col
        if isinstance(self._value, (list, tuple)):  # handle arrays
            ar = [F.lit(x) for x in self._value]
            spark_value = F.array(*ar)

        elif isinstance(self._value, dict):  # handle maps
            mapping = list(itertools.chain(*self._value.items()))
            spark_value = F.create_map(*[F.lit(x) for x in mapping])

        else:  # scalar types
            spark_value = F.lit(self._value)

        if self._cast is not None:
            spark_value = spark_value.cast(self._cast)

        return df.withColumn(self._name, spark_value)
