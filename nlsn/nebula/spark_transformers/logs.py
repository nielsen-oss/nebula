"""Logging and Monitoring transformers.

They don't manipulate the data but may trigger eager evaluation.
"""

from typing import Iterable, List, Optional, Union

from nlsn.nebula.auxiliaries import (
    assert_at_least_one_non_null,
)
from nlsn.nebula.base import Transformer
from nlsn.nebula.logger import logger
from nlsn.nebula.spark_util import (
    cache_if_needed,
    get_data_skew,
)
from nlsn.nebula.storage import nebula_storage as ns

__all__ = [
    "GroupByCountRows",
]


class GroupByCountRows(Transformer):
    def __init__(
            self,
            *,
            columns: Optional[Union[str, List[str]]] = None,
            regex: Optional[str] = None,
            glob: Optional[str] = None,
            startswith: Optional[Union[str, Iterable[str]]] = None,
            endswith: Optional[Union[str, Iterable[str]]] = None,
            allow_excess_columns: bool = True,
            store_key: Optional[str] = None,
            persist: bool = False,
    ):
        """Group by and count the number of rows.

        It returns the input dataframe unmodified.

        Args:
            columns (str | list(str) | None):
                A list of columns to groupby. Defaults to None.
            regex (str | None):
                Select the columns to groupby by using a regex pattern.
                Defaults to None.
            glob (str | None):
                Select the columns to groupby by using a bash-like pattern.
                Defaults to None.
            startswith (str | iterable(str) | None):
                Select all the columns whose names start with the provided
                string(s). Defaults to None.
            endswith (str | iterable(str) | None):
                Select all the columns whose names end with the provided
                string(s). Defaults to None.
            allow_excess_columns (bool):
                Whether to allow 'columns' argument to list columns that are
                not present in the dataframe. Default True.
                If 'columns' contains columns that are not present in the
                DataFrame and 'allow_excess_columns' is set to False, raise
                an AssertionError.
            store_key (str | None):
                If provided, store the result as dictionary like:
                group -> number of rows
                as type: <tuple<str>> -> <int>.
            persist (bool):
                If True, cache the DataFrame before collecting the data.
                Default to False.

        Raises:
            AssertionError: If no columns are provided.
            AssertionError: If `allow_excess_columns` is False, and the column
            list contains columns that are not present in the DataFrame.
        """
        assert_at_least_one_non_null(columns, regex, glob)

        super().__init__()
        self._set_columns_selections(
            columns=columns,
            regex=regex,
            glob=glob,
            startswith=startswith,
            endswith=endswith,
            allow_excess_columns=allow_excess_columns,
        )
        self._store_key: Optional[str] = store_key
        self._persist: bool = persist

    def _transform(self, df):
        df = cache_if_needed(df, self._persist)

        selection = self._get_selected_columns(df)

        rows = df.groupBy(selection).count().collect()
        groups: List[tuple] = [i[:-1] for i in rows]
        counts: List[int] = [i[-1] for i in rows]

        d_counts = dict(zip(groups, counts))

        logger.info(f"Row count grouping {selection}:")
        for k, v in d_counts.items():
            logger.info(f"{k} -> {v}")

        if self._store_key:
            ns.set(self._store_key, d_counts)

        return df
