"""Logging and Monitoring transformers.

They don't manipulate the data but may trigger eager evaluation.
"""

import socket
from typing import Dict, Hashable, Iterable, List, Optional, Union

from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from nlsn.nebula.auxiliaries import (
    assert_at_least_one_non_null,
    assert_is_bool,
    assert_only_one_non_none,
    ensure_flat_list,
)
from nlsn.nebula.base import Transformer
from nlsn.nebula.logger import logger
from nlsn.nebula.spark_util import (
    assert_col_type,
    cache_if_needed,
    get_data_skew,
    get_spark_session,
)
from nlsn.nebula.storage import assert_is_hashable
from nlsn.nebula.storage import nebula_storage as ns

__all__ = [
    "CountDistinctRows",
    "CountWhereBool",
    "CpuInfo",
    "GetNumPartitions",
    "GroupByCountRows",
    "LogDataSkew",
]


class CountDistinctRows(Transformer):
    def __init__(
        self,
        *,
        columns: Optional[Union[str, List[str]]] = None,
        store_key: Optional[str] = None,
    ):
        """Count distinct rows of a dataframe.

        Args:
            columns (str | list(str) | None):
                A list of columns to select for the count distinct operation.
                If None, all columns are selected.
                Defaults to None.
            store_key (bool | str):
                If provided, store the count value in nebula storage with
                the given key.
                Defaults to None.
        """
        super().__init__()
        self._columns: List[str] = ensure_flat_list(columns)
        self._store_key: Optional[str] = store_key

    def _transform(self, df):
        cols: List[str]
        if self._columns:
            cols = self._columns
            msg = f"selected columns ({cols})"
        else:
            cols = df.columns
            msg = "all columns"

        n: int = df.select(cols).distinct().count()

        logger.info(f"Count distinct rows on {msg}: {n}")
        if self._store_key:
            ns.set(self._store_key, n)

        return df


class CountWhereBool(Transformer):
    def __init__(
        self,
        *,
        count_type: bool = True,
        columns: Optional[Union[str, List[str]]] = None,
        regex: Optional[str] = None,
        glob: Optional[str] = None,
        startswith: Optional[Union[str, Iterable[str]]] = None,
        endswith: Optional[Union[str, Iterable[str]]] = None,
        allow_excess_columns: bool = True,
        store_key: Optional[str] = None,
        persist: bool = False,
    ):
        """Count the number of 'True' / 'False' in boolean columns.

        Null values are not counted.
        It returns the input dataframe unmodified.

        Args:
            count_type (bool):
                If <True> count the number of True values, if False count the
                number of False values.
                Defaults to True.
            columns (str | list(str) | None):
                List of columns to use. Defaults to None.
            regex (str | None):
                Select the columns by using a regex pattern.
                Defaults to None.
            glob (str | None):
                Select the columns by using a bash-like pattern.
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
                column -> count of True / False
                as type: <str> -> <int>.
            persist (bool):
                If True, cache the DataFrame before collecting the data.
                Default to False.

        Raises:
            AssertionError: If no columns are provided.
            AssertionError: If `allow_excess_columns` is False, and the column
            list contains columns that are not present in the DataFrame.
        """
        assert_is_bool(count_type, "count_type")
        assert_only_one_non_none(columns, regex, glob)

        super().__init__()
        self._counts: bool = count_type
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

        for c in selection:
            assert_col_type(df, c, "boolean")

        if self._counts:
            agg = [F.sum(F.col(i).cast("int")).alias(i) for i in selection]
        else:
            agg = [F.sum((~F.col(i)).cast("int")).alias(i) for i in selection]

        counts: Dict[str, int]
        counts = df.agg(*agg).rdd.map(lambda x: x.asDict()).collect()[0]
        counts = {k: v if v else 0 for k, v in counts.items()}

        if self._store_key:
            ns.set(self._store_key, counts)

        for c in selection:
            v = counts[c]
            v = v if v else 0
            logger.info(f"'{self._counts}' in {c}: {v}")

        return df


class CpuInfo(Transformer):
    def __init__(self, *, n_partitions: int = 100):
        """Show the CPU name of the spark workers.

        The main dataframe remains untouched.

        E.g.:
        +----------------------------------------------+---------------------+
        |cpu                                           |host                 |
        +----------------------------------------------+---------------------+
        |AMD EPYC 7571                                 |spark-worker-3ge[...]|
        |Intel(R) Xeon(R) Platinum 8259CL CPU @ 2.50GHz|spark-worker-099[...]|
        |Intel(R) Xeon(R) Platinum 8259CL CPU @ 2.50GHz|spark-worker-099[...]|
        +----------------------------------------------+---------------------+

        Args:
            n_partitions (int):
                The CPU name is retrieved through a UDF that runs on a mock
                dataframe, not the provided one as input.
                The parameter 'n_partitions' defines the number of rows and
                partitions of this dummy dataframe that will be created.
                A rule of thumb is to have at least 10 times the number of
                clusters and rows and partitions to ensure that each node
                processes the data.
                Default 100.
        """
        super().__init__()

        try:  # pragma: no cover
            import cpuinfo  #   # noqa: F401
        except ImportError:  # pragma: no cover
            msg = "'cpuinfo' optional package not installed. \n"
            msg += "Run 'pip install py-cpuinfo' or 'install nebula[cpu-info]'"
            raise ImportError(msg)

        self._n: int = n_partitions

    def _transform(self, df):
        import cpuinfo  # noqa: F401

        def _func() -> list:  # pragma: no cover
            cpu: str = cpuinfo.get_cpu_info()["brand_raw"]
            name: str = socket.gethostname()
            return [cpu, name]

        data = [[i] for i in range(self._n)]
        schema = StructType([StructField("_none_", IntegerType(), True)])
        ss = get_spark_session(df)

        cpu_info_udf = F.udf(_func, ArrayType(StringType()))

        (
            ss.createDataFrame(data, schema=schema)
            .repartition(self._n)
            .withColumn("_arr_", cpu_info_udf())
            .withColumn("cpu", F.col("_arr_").getItem(0))
            .withColumn("host", F.col("_arr_").getItem(1))
            .select("cpu", "host")
            .distinct()
            .sort("cpu")
            .show(100, truncate=False)
        )

        return df


class GetNumPartitions(Transformer):
    def __init__(self, *, store_key: Optional[Hashable] = None):
        """Count and log the number of partitions in RDD.

        Args:
            store_key (hashable | None):
                If provided, store the value (int) in the Nebula Cache.
                It must be a hashable value. Default to None.
        """
        if store_key is not None:
            assert_is_hashable(store_key)

        super().__init__()
        self._store: Optional[Hashable] = store_key

    def _transform(self, df):
        n: int = df.rdd.getNumPartitions()
        logger.info(f"Number of partitions: {n}")
        if self._store:
            ns.set(self._store, n)

        return df


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


class LogDataSkew(Transformer):
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

    def _transform(self, df):
        df = cache_if_needed(df, self._persist)

        dict_skew = get_data_skew(df, as_dict=True)
        n_part: int = dict_skew["partitions"]
        desc: str = dict_skew["skew"]

        logger.info(f"Number of partitions: {n_part}")
        logger.info("Rows distribution in partitions:")
        logger.info(desc)

        return df
