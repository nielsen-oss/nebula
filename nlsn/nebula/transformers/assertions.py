"""Transformers to assert certain conditions.

They don't manipulate the data but may trigger eager evaluation.
"""

from nlsn.nebula.auxiliaries import ensure_flat_list
from nlsn.nebula.base import Transformer
from nlsn.nebula.logger import logger

__all__ = [
    # "AssertCount",
    # "AssertNotEmpty",
    # "AssertRowsAreDistinct",
    "DataFrameContainsColumns",
]


class DataFrameContainsColumns(Transformer):
    def __init__(self, *, columns: str | list[str]):
        """Raise AssertionError if the dataframe does not contain the expected columns.

        Args:
            columns (str | list(str)):
                The column or list of columns to check for existence.

        Raises:
            AssertionError if the dataframe does not contain the expected columns.
        """
        super().__init__()
        self._cols: list[str] = ensure_flat_list(columns)

    def _transform(self, df):
        actual_cols = set(df.columns)
        missing_cols = [i for i in self._cols if i not in actual_cols]

        if missing_cols:
            logger.info(f"Missing columns: {missing_cols}")
            raise AssertionError("Some expected columns are missing")

        return df
