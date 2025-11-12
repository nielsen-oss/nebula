"""Spark transformers related to the partitioning."""

from typing import Hashable

from nlsn.nebula.auxiliaries import assert_is_bool, ensure_flat_list
from nlsn.nebula.base import Transformer
from nlsn.nebula.logger import logger
from nlsn.nebula.spark_util import get_default_spark_partitions, get_spark_session, cache_if_needed, get_data_skew
from nlsn.nebula.storage import assert_is_hashable
from nlsn.nebula.storage import nebula_storage as ns

__all__ = [
    "Cache",  # alias Persist
    "ClearCache",
    "CoalescePartitions",
    "GetNumPartitions",
    "LogDataSkew",
    "Persist",  # alias Cache
    "Repartition",
    "UnPersist",
]


def _assert_valid_partitions(value: int, name: str):
    if (not isinstance(value, int)) or (value < 1):
        msg = f'If "{name}" is provided, must be an integer > 1'
        raise AssertionError(msg)


class _Partitions(Transformer):
    def __init__(
            self,
            *,
            num_partitions: int | None = None,
            to_default: bool = False,
            rows_per_partition: int | None = None,
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

        self._num_part: int | None = num_partitions
        self._to_default: bool = to_default  # not used
        self._rows_per_part: int | None = rows_per_partition

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

    def _transform_spark(self, df):
        raise NotImplementedError


class ClearCache(Transformer):
    def __init__(self):
        """Remove all cached tables from the in-memory cache."""
        super().__init__()

    def _transform_spark(self, df):
        logger.info("Removing all cached tables from the in-memory cache.")
        spark_session = get_spark_session(df)
        spark_session.catalog.clearCache()
        return df


class CoalescePartitions(_Partitions):
    def __init__(
            self,
            *,
            num_partitions: int | None = None,
            to_default: bool = False,
            rows_per_partition: int | None = None,
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

    def _transform_spark(self, df):
        n_part: int = self._get_requested_partitions(df, "Coalesce")
        return df.coalesce(n_part)


class GetNumPartitions(Transformer):
    """Spark transformer."""

    def __init__(self, *, store_key: Hashable | None = None):
        """Count and log the number of partitions in RDD.

        Args:
            store_key (hashable | None):
                If provided, store the value (int) in the Nebula Cache.
                It must be a hashable value. Default to None.
        """
        if store_key is not None:
            assert_is_hashable(store_key)

        super().__init__()
        self._store: Hashable | None = store_key

    def _transform_spark(self, df):
        n: int = df.rdd.getNumPartitions()
        logger.info(f"Number of partitions: {n}")
        if self._store:
            ns.set(self._store, n)
        return df



class LogDataSkew(Transformer):
    """Spark transformer."""
    def __init__(self, *, persist: bool = False):
        """Describe the partition distribution of the dataframe.

        - Number of partitions.
        - Distribution: mean | std | min | 25% | 50% | 75% | max

        The input dataframe is not modified.

        Args:
            persist (bool):
                Persist the dataframe if not already cached.
        """
        try:  # pragma: no cover
            import pandas  # noqa: F401
        except ImportError:  # pragma: no cover
            msg = "'pandas' optional package not installed. \n"
            msg += "Run 'pip install pandas' or 'install nebula[pandas]'"
            raise ImportError(msg)

        super().__init__()
        self._persist: bool = persist

    def _transform_spark(self, df):
        df = cache_if_needed(df, self._persist)

        dict_skew = get_data_skew(df, as_dict=True)
        n_part: int = dict_skew["partitions"]
        desc: str = dict_skew["skew"]

        logger.info(f"Number of partitions: {n_part}")
        logger.info("Rows distribution in partitions:")
        logger.info(desc)

        return df

class Persist(Transformer):
    def __init__(self):
        """Cache dataframe if not already cached."""
        super().__init__()

    def _transform_spark(self, df):
        if df.is_cached:
            logger.info("DataFrame was already cached, no need to persist.")
            return df
        logger.info("Caching dataframe")
        return df.cache()


class Repartition(_Partitions):
    def __init__(
            self,
            *,
            num_partitions: int | None = None,
            to_default: bool = False,
            rows_per_partition: int | None = None,
            columns: str | list[str] | None = None,
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

        self._columns: list[str] | None = None

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

    def _transform_spark(self, df):
        n_part: int = self._get_requested_partitions(df, "Repartition")
        args = [n_part]

        if self._columns:
            set_cols: set[str] = set(self._columns)
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

    def _transform_spark(self, df):
        logger.info("Un-persist the dataframe")
        return df.unpersist(blocking=self._blocking)


# ---------------------- ALIASES ----------------------
Cache = Persist
