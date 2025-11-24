"""General Purposes Transformers."""

import itertools

from nlsn.nebula.base import Transformer

__all__ = [
    "WithColumn",
]


class WithColumn(Transformer):
    backends = {"pandas", "polars", "spark"}

    def __init__(
            self,
            *,
            column_name: str,
            value=None,
            cast=None,
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
        self._value = value
        self._cast = cast
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
