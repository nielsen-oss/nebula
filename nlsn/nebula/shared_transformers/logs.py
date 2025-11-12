"""Logging and Monitoring transformers."""

from typing import Hashable, Optional

import narwhals as nw

from nlsn.nebula.base import Transformer
from nlsn.nebula.logger import logger
from nlsn.nebula.storage import assert_is_hashable
from nlsn.nebula.storage import nebula_storage as ns

__all__ = [
    "Count",
]


class Count(Transformer):

    def __init__(self, *, store_key: Optional[Hashable] = None, persist: bool = False):
        """Count and log the number of rows in the DataFrame.

        Args:
            store_key (hashable | None):
                If provided, store the value (int) in the Nebula Cache.
                It must be a hashable value. Default to None.
            persist (bool):
                Cache the dataframe before the count. This parameter is taken
                into consideration only for Spark dataframes.
                Defaults to False.
        """
        if store_key is not None:
            assert_is_hashable(store_key)

        super().__init__()
        self._persist: bool = persist
        self._store: Optional[Hashable] = store_key

    def __log_and_store(self, n_rows: int) -> None:
        logger.info(f"Number of rows: {n_rows}")
        if self._store:
            ns.set(self._store, n_rows)

    def _transform_nw(self, nw_df):
        if self._persist and is_spark_native(nw_df):
            nw_df = nw_df.persist()


        n_rows: int = nw_df.shape[0]
        self.__log_and_store(n_rows)
        return nw_df

    def _transform_spark(self, df):
        from nlsn.nebula.spark_util import cache_if_needed

        df = cache_if_needed(df, self._persist)
        n_rows: int = df.count()
        self.__log_and_store(n_rows)
        return df
