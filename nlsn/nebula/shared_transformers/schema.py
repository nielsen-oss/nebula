"""Dataframe Schema Operations: Casting, Nullability, etc.

These transformers are not supposed to act directly on values, but
some operations (i.e., cast) can affect them.
"""

from nlsn.nebula.base import Transformer
from nlsn.nebula.df_types import get_dataframe_type

__all__ = [
    "SortColumnNames",
]


class SortColumnNames(Transformer):
    backends = {"pandas", "polars", "spark"}

    def __init__(self):
        """Sort the columns alphabetically."""
        super().__init__()

    def _transform(self, df):
        name: str = get_dataframe_type(df)
        if name == "pandas":
            return df[sorted(df.columns)]
        return df.select(*sorted(df.columns))
