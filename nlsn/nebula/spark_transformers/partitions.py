"""Transformers that act on whole datasets, whole tables or spark internals."""

from typing import List, Optional, Set, Union

from nlsn.nebula.auxiliaries import assert_is_bool, ensure_flat_list
from nlsn.nebula.base import Transformer
from nlsn.nebula.logger import logger
from nlsn.nebula.spark_util import get_default_spark_partitions, get_spark_session

__all__ = [
    "Cache",  # alias Persist
    "CoalescePartitions",
    "Persist",  # alias Cache
    "Repartition",
    "UnPersist",
    "ClearCache",
]


def _assert_valid_partitions(value: int, name: str):
    if (not isinstance(value, int)) or (value < 1):
        msg = f'If "{name}" is provided, must be an integer > 1'
        raise AssertionError(msg)


class _Partitions(Transformer):
    def __init__(
        self,
        *,
        num_partitions: Optional[int] = None,
        to_default: bool = False,
        rows_per_partition: Optional[int] = None,
    ):
        assert_is_bool(to_default, "to_default")

        n_p = bool(num_partitions) + bool(to_default) + bool(rows_per_partition)
        if n_p == 0:
            raise AssertionError("No partition strategy provided")
        if n_p > 1:
            msg = 'Only one among "num_partitions", "to_partitions" '
            msg += 'and "rows_per_partition" can be provided'
            raise AssertionError(msg)

        if num_partitions is not None:
            _assert_valid_partitions(num_partitions, "num_partitions")

        if rows_per_partition is not None:
            _assert_valid_partitions(rows_per_partition, "rows_per_partition")

        super().__init__()

        self._num_part: Optional[int] = num_partitions
        self._to_default: bool = to_default  # not used
        self._rows_per_part: Optional[int] = rows_per_partition

    def _get_requested_partitions(self, df, op: str) -> int:
        if self._rows_per_part:
            n_rows: int = df.count()
            n_part = max(n_rows // self._rows_per_part, 1)
            logger.info(f"{op} to {n_part} partitions ({self._rows_per_part} per row)")

        elif self._num_part:
            n_part = self._num_part
            logger.info(f"{op} to {n_part} partitions")

        else:  # to default partition
            n_part = get_default_spark_partitions(df)
            logger.info(f"{op} to default partitions ({n_part})")

        return n_part

    def _transform(self, df):
        raise NotImplementedError


class Persist(Transformer):
    def __init__(self):
        """Cache dataframe if not already cached."""
        super().__init__()

    def _transform(self, df):
        if df.is_cached:
            logger.info("DataFrame was already cached, no need to persist.")
            return df
        logger.info("Caching dataframe")
        return df.cache()


class CoalescePartitions(_Partitions):
    def __init__(
        self,
        *,
        num_partitions: Optional[int] = None,
        to_default: bool = False,
        rows_per_partition: Optional[int] = None,
    ):
        """Coalesce a dataframe according to the inputs.

        Only one among "num_partitions", "to_partitions" and
        "rows_per_partition" can be provided.

        Args:
            num_partitions (int | None):
                Specify the target number of partitions.
            to_default (bool):
                Coalesce to the default number of spark shuffle partitions.
            rows_per_partition (int | None):
                Coalesce the dataframe based on the desired
                number of rows per partition.
        """
        super().__init__(
            num_partitions=num_partitions,
            to_default=to_default,
            rows_per_partition=rows_per_partition,
        )

    def _transform(self, df):
        n_part: int = self._get_requested_partitions(df, "Coalesce")
        return df.coalesce(n_part)


class Repartition(_Partitions):
    def __init__(
        self,
        *,
        num_partitions: Optional[int] = None,
        to_default: bool = False,
        rows_per_partition: Optional[int] = None,
        columns: Optional[Union[str, List[str]]] = None,
    ):
        """Return a new DataFrame partitioned by the given partitioning expressions.

        Only one among "num_partitions", "to_partitions" and
        "rows_per_partition" can be provided.

        The resulting DataFrame is hash partitioned.

        Args:
            num_partitions (int | None):
                Specify the target number of partitions.
            to_default (bool):
                Coalesce to the default number of spark shuffle partitions.
            rows_per_partition (int | None):
                Coalesce the dataframe based on the desired
                number of rows per partition.
            columns (str | list(str) | None):
                Partitioning columns.
        """
        super().__init__(
            num_partitions=num_partitions,
            to_default=to_default,
            rows_per_partition=rows_per_partition,
        )

        self._columns: Optional[List[str]] = None

        msg = '"columns" must be <str> or <list> of <str>'
        if columns is not None:
            if not isinstance(columns, (list, str)):
                raise AssertionError(msg)
            if isinstance(columns, list):
                if not all(isinstance(i, str) for i in columns):
                    raise AssertionError(msg)
                if len(columns) != len(set(columns)):
                    raise AssertionError("duplicated columns")

            self._columns = ensure_flat_list(columns)

    def _transform(self, df):
        n_part: int = self._get_requested_partitions(df, "Repartition")
        args = [n_part]

        if self._columns:
            set_cols: Set[str] = set(self._columns)
            diff = set_cols.difference(df.columns)
            if diff:
                raise AssertionError(f"{diff} not in columns")

            args += self._columns

        return df.repartition(*args)


class UnPersist(Transformer):
    def __init__(self, *, blocking: bool = False):
        """Unpersist a Spark DataFrame.

        Args:
            blocking (bool):
                If set to True, the 'unpersist' operation will block your
                computation pipeline at the moment it reaches that instruction
                until it has finished removing the contents of the dataframe.
                Otherwise, it will just put a mark on the dataframe which
                tells spark it can safely delete it whenever it needs to.
                Defaults to False
        """
        super().__init__()
        self._blocking: bool = blocking

    def _transform(self, df):
        logger.info("Un-persist the dataframe")
        return df.unpersist(blocking=self._blocking)


class ClearCache(Transformer):
    def __init__(self):
        """Remove all cached tables from the in-memory cache."""
        super().__init__()

    def _transform(self, df):
        logger.info("Removing all cached tables from the in-memory cache.")
        spark_session = get_spark_session(df)
        spark_session.catalog.clearCache()
        return df


# ---------------------- ALIASES ----------------------
Cache = Persist
