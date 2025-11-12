"""Dataframe Schema Operations: Casting, Nullability, etc.

These transformers are not supposed to act directly on values, but
some operations (i.e., cast) can affect them.
"""

import pyspark.sql.functions as F

from nlsn.nebula.base import Transformer

__all__ = [
    "Cast",
]


class Cast(Transformer):
    def __init__(self, *, cast: dict[str, str]):
        """Cast fields to the requested data-types.

        Args:
            cast (dict(str, str)):
                Cast dictionary <column -> type>.
        """
        super().__init__()
        self._cast: dict[str, str] = cast

    def _transform(self, df):
        cols = [
            F.col(c).cast(self._cast[c]).alias(c) if c in self._cast else c
            for c in df.columns
        ]
        return df.select(*cols)
