"""Transformers for numerical operations."""

from typing import Optional, Union

from nlsn.nebula.auxiliaries import assert_is_integer, ensure_list
from nlsn.nebula.base import Transformer
from nlsn.nebula.deprecations import deprecate_transformer

__all__ = [
    "RoundDecimalValues",  # Deprecated; alias RoundValues
    "RoundValues",
]


class RoundValues(Transformer):
    backends = {"pandas", "polars", "spark"}

    def __init__(
        self,
        *,
        input_columns: Union[str, list[str]],
        precision: int = 2,
        output_column: Optional[str] = None,
    ):
        """Round to decimal values when precision <= 0, or at integral part otherwise.

        To Allow negative rounding in DecimalType set
        spark.sql.legacy.allowNegativeScaleOfDecimal -> true

        Args:
            input_columns (str | list(str)):
                Name of the input column containing decimal values.
                If it is a list with len > 1, the `output_column` must be None
                and the round occurs in place.
            precision (int):
                Number of decimal places to round to (default: 2).
            output_column (str | None):
                Name of the output column, if not provided, the output columns
                will be the input column (inplace rounding.)
                If `input_column` is a list with len > 1, the `output_column`
                must be None.

        Raises:
            ValueError: If `input_column` is not an integer.
        """
        assert_is_integer(precision, "precision")

        super().__init__()
        self._input_cols: list[str] = ensure_list(input_columns)
        self._output_col: Optional[str] = None
        self._is_multiple: bool = len(self._input_cols) > 1

        if self._is_multiple:
            if output_column is not None:
                msg = "If 2+ columns are passed as input 'output_column' must be None."
                raise AssertionError(msg)
            if len(self._input_cols) != len(set(self._input_cols)):
                raise AssertionError("Input columns not unique.")
        else:
            self._output_col = output_column or self._input_cols[0]
        self._scale: int = int(precision)

        self._allowed_dtypes: tuple

    def _transform(self, df):
        return self._select_transform(df)

    def _check_spark_dtype(self, df, columns: list[str]):
        msg_err = "Input column '{}' is '{}'. Must be numeric type."
        for col_name in columns:
            dtype = df.schema[col_name].dataType
            if not isinstance(dtype, self._allowed_dtypes):  # pragma: no cover
                raise TypeError(msg_err.format(col_name, dtype))

    def _transform_spark(self, df):
        import pyspark.sql.functions as F
        from pyspark.sql.types import (
            DecimalType,
            DoubleType,
            FloatType,
            IntegerType,
            LongType,
        )

        self._allowed_dtypes = (
            DecimalType,
            DoubleType,
            FloatType,
            IntegerType,
            LongType,
        )

        self._check_spark_dtype(df, self._input_cols)

        # Single column
        if not self._is_multiple:
            self._check_spark_dtype(df, self._input_cols)
            op = F.round(self._input_cols[0], self._scale)
            return df.withColumn(self._output_col, op)

        # Multiple columns: in this case, the 'output_column' is None, and the
        # column order is preserved.
        cols2round: set[str] = set(self._input_cols)
        vr: int = self._scale
        cols = [F.round(c, vr).alias(c) if c in cols2round else c for c in df.columns]

        return df.select(cols)

    def _transform_pandas(self, df):
        if not self._is_multiple:
            df[self._output_col] = df[self._input_cols[0]].round(self._scale)
            return df

        for c in self._input_cols:
            df[c] = df[c].round(self._scale)

        return df

    def _transform_polars(self, df):
        import polars as pl

        is_positive: bool = self._scale >= 0
        schema: dict = df.schema

        def _round_neg(_x):  # pragma: no cover
            return round(_x, self._scale)

        if not self._is_multiple:
            input_col: str = self._input_cols[0]
            if is_positive:
                op = pl.col(input_col).round(self._scale)
            else:
                op = pl.col(input_col).map_elements(_round_neg).cast(schema[input_col])
            return df.with_columns(op.alias(self._output_col))

        if is_positive:
            li_ops = [pl.col(i).round(self._scale) for i in self._input_cols]
        else:
            li_ops = [
                pl.col(i).map_elements(_round_neg).cast(schema[i]) for i in self._input_cols
            ]

        return df.with_columns(li_ops)


# ---------------------- DEPRECATED ----------------------
RoundDecimalValues = deprecate_transformer(RoundValues, "RoundDecimalValues")
