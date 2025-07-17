"""Custom Pandas / Polars extra transformers."""

import polars as pl

from nlsn.nebula.base import Transformer

__all__ = ["AddOne"]


class AddOne(Transformer):
    backends = {"pandas", "polars"}

    def __init__(self, *, column_name: str):
        """Add one to the specified column."""
        super().__init__()
        self._name: str = column_name

    def _transform(self, df):
        return self._select_transform(df)

    def _transform_pandas(self, df):
        ret = df.copy()
        ret[self._name] += 1
        return ret

    def _transform_polars(self, df):
        value = pl.col(self._name) + 1
        return df.with_columns(value.alias(self._name))
