"""Row Filtering Operations."""

from nlsn.nebula.base import Transformer

__all__ = [
    "Distinct",
]


class Distinct(Transformer):
    backends = {"pandas", "polars", "spark"}

    def __init__(self, *, maintain_order: bool = False):
        """Return only distinct rows.

        Args:
            maintain_order (bool):
                This parameter is used only for Polars dataframes.
                For Pandas and Spark ones are ignored.
                Keep the same order as the original DataFrame.
                This is more expensive to compute. Settings this to True
                blocks the possibility to run on the streaming engine.
                Defaults to False.
        """
        super().__init__()
        self._maintain_order: bool = maintain_order

    def _transform(self, df):
        return self._select_transform(df)

    @staticmethod
    def _transform_spark(df):
        return df.distinct()

    @staticmethod
    def _transform_pandas(df):
        return df.drop_duplicates()

    def _transform_polars(self, df):
        return df.unique(maintain_order=self._maintain_order)
