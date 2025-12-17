"""Transformers for debugging."""

from nebula.base import Transformer

__all__ = [
    "PrintSchema",
]


class PrintSchema(Transformer):
    def __init__(self):
        """Print out the data types of the dataframe."""
        super().__init__()

    @staticmethod
    def _transform_nw(df_nw):
        for k, v in df_nw.schema.items():
            print(f"{k}: {v}")
        return df_nw
