"""Transformers to assert certain conditions.

They don't manipulate the data but may trigger eager evaluation.
"""

from nlsn.nebula.auxiliaries import ensure_flat_list
from nlsn.nebula.base import Transformer
from nlsn.nebula.logger import logger

__all__ = [
    "AssertCount",
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

class AssertCount(Transformer):
    def __init__(
            self,
            *,
            expected: Optional[int] = None,
            min_count: Optional[int] = None,
            max_count: Optional[int] = None,
            operator: str = "eq",  # eq, ne, ge, gt, le, lt
            persist_spark: bool = False,  # NEW: explicit control
            collect_polars: bool = True,  # NEW: explicit control
    ):
        """
        Args:
            persist_spark: If True, cache Spark DF before counting
            collect_polars: If True, collect Polars LazyFrame before counting
        """
        self._expected = expected
        self._min = min_count
        self._max = max_count
        self._operator = operator
        self._persist_spark = persist_spark
        self._collect_polars = collect_polars

    def _transform(self, df):
        backend = self._detect_backend(df)

        # Handle Spark specially
        if backend == "spark":
            if self._persist_spark and not df.is_cached:
                df = df.cache()
            count = df.count()  # Spark action
            # df is still the same object (cached or not)

        # Handle Polars LazyFrame
        elif backend == "polars" and hasattr(df, 'collect'):
            if self._collect_polars:
                # Collect and count, but return original LazyFrame
                count = len(df.collect())
                # df is still LazyFrame
            else:
                # Use estimated count or other strategy
                count = df.select(pl.count()).collect().item()

        # Pandas or materialized Polars
        else:
            count = len(df)

        # Validate
        self._validate_count(count)

        # Return original df (unchanged)
        return df
