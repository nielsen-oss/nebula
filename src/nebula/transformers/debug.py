"""Transformers for debugging."""

from nebula.base import Transformer
from nebula.logger import logger

__all__ = [
    "PrintSchema",
]


class PrintSchema(Transformer):
    """Print out the data types of the dataframe."""

    @staticmethod
    def _transform_nw(df_nw):
        for k, v in df_nw.schema.items():
            logger.info(f"{k}: {v}")
        return df_nw
