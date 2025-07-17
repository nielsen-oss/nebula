"""Logging and Monitoring transformers."""

from typing import Hashable, Optional

from nlsn.nebula.base import Transformer
from nlsn.nebula.logger import logger
from nlsn.nebula.storage import assert_is_hashable
from nlsn.nebula.storage import nebula_storage as ns

__all__ = [
    "Count",
]


class Count(Transformer):
    backends = {"pandas", "polars", "spark"}

    def __init__(self, *, persist: bool = False, store_key: Optional[Hashable] = None):
        """Count and log the number of rows in the DataFrame.

        Args:
            persist (bool):
                Cache the dataframe before the count. This parameter is taken
                into consideration only for Spark dataframes.
                Defaults to False.
            store_key (hashable | None):
                If provided, store the value (int) in the Nebula Cache.
                It must be a hashable value. Default to None.
        """
        if store_key is not None:
            assert_is_hashable(store_key)

        super().__init__()
        self._persist: bool = persist
        self._store: Optional[Hashable] = store_key

    def _transform(self, df):
        return self._select_transform(df)

    def __log_and_store(self, n_rows: int) -> None:
        logger.info(f"Number of rows: {n_rows}")
        if self._store:
            ns.set(self._store, n_rows)

    def _transform_pandas(self, df):
        n_rows: int = df.shape[0]
        self.__log_and_store(n_rows)
        return df

    def _transform_polars(self, df):
        n_rows: int = df.shape[0]
        self.__log_and_store(n_rows)
        return df

    def _transform_spark(self, df):
        from nlsn.nebula.spark_util import cache_if_needed

        df = cache_if_needed(df, self._persist)
        n_rows: int = df.count()
        self.__log_and_store(n_rows)
        return df
