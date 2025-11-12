"""Transformers to assert certain conditions.

They don't manipulate the data but may trigger eager evaluation.
"""

from nlsn.nebula.base import Transformer

__all__ = [
    "AssertNotEmpty",
]


class AssertNotEmpty(Transformer):
    def __init__(self, *, df_name: str = "Dataframe"):
        """Raise AssertionError if the input dataframe is empty.

        Args:
            df_name (str):
                DataFrame name to display if the assertion fails.
                The error message will be `df_name` + " is empty"
                Defaults to "Dataframe".

        Raises:
            AssertionError if the dataframe is empty.
        """
        super().__init__()
        self._df_name: str = df_name

    def _transform(self, df):
        if df.rdd.isEmpty():
            raise AssertionError(f"{self._df_name} is empty")
        return df
