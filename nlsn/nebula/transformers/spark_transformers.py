"""Spark transformers related to the partitioning."""
import socket
from typing import Any

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, StringType

from nlsn.nebula.auxiliaries import ensure_flat_list, assert_allowed
from nlsn.nebula.base import Transformer
from nlsn.nebula.logger import logger
from nlsn.nebula.spark_util import (
    get_default_spark_partitions,
    cache_if_needed,
    get_data_skew,
    get_spark_session
)

__all__ = [
    "Cache",  # alias Persist
    "CoalescePartitions",
    "ColumnMethod",
    "CpuInfo",
    "LogDataSkew",
    "Persist",  # alias Cache
    "Repartition",
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
            rows_per_partition: int | None = None,
    ):
        n_p = bool(num_partitions) + bool(rows_per_partition)
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
        self._rows_per_part: int | None = rows_per_partition

    def _get_requested_partitions(self, df, op: str) -> int:
        if self._rows_per_part:
            n_rows: int = df.count()
            n_part = max(n_rows // self._rows_per_part, 1)
            # print(f"{op} to {n_part} partitions ({self._rows_per_part} per row)")

        elif self._num_part:
            n_part = self._num_part
            # print(f"{op} to {n_part} partitions")

        else:  # to default partition
            n_part = get_default_spark_partitions(df)
            # print(f"{op} to default partitions ({n_part})")

        return n_part

    def _transform_spark(self, df):
        raise NotImplementedError


class CoalescePartitions(_Partitions):
    def __init__(
            self,
            *,
            num_partitions: int | None = None,
            rows_per_partition: int | None = None,
    ):
        """Coalesce a dataframe according to the inputs.

        Only one among "num_partitions", and "rows_per_partition"
        can be provided. If both are None, the partitions will be coalesced
        to the default value.

        Args:
            num_partitions (int | None):
                Specify the target number of partitions.
            rows_per_partition (int | None):
                Coalesce the dataframe based on the desired
                number of rows per partition.
        """
        super().__init__(
            num_partitions=num_partitions,
            rows_per_partition=rows_per_partition,
        )

    def _transform_spark(self, df):
        n_part: int = self._get_requested_partitions(df, "Coalesce")
        return df.coalesce(n_part)


class ColumnMethod(Transformer):
    def __init__(
            self,
            *,
            input_column: str,
            output_column: str | None = None,
            method: str,
            args: list[Any] = None,
            kwargs: dict[str, Any] | None = None,
    ):
        """Call a pyspark.sql.Column method with the provided args/kwargs.

        Args:
            input_column (str):
                Name of the input column.
            output_column (str):
                Name of the column where the result of the function is stored.
                If not provided, the input column will be used.
                Defaults to None.
            method (str):
                Name of the pyspark.sql.Column method to call.
            args (list(any) | None):
                Positional arguments of pyspark.sql.Column method.
                Defaults to None.
            kwargs (dict(str, any) | None):
                Keyword arguments of pyspark.sql.Column method.
                Defaults to None.
        """
        super().__init__()
        self._input_col: str = input_column
        self._output_col: str = output_column if output_column else input_column
        self._meth: str = method
        self._args: list = args if args else []
        self._kwargs: dict[str, Any] = kwargs if kwargs else {}

        # Attempt to retrieve any errors during initialization.
        # Use a try-except block because Spark may not be running at this
        # point, making it impossible to guarantee the availability of the
        # requested method.
        self._assert_col_meth(False)

    def _assert_col_meth(self, raise_err: bool):
        try:
            all_meths = dir(F.col(self._input_col))
        except AttributeError as e:  # pragma: no cover
            if raise_err:
                raise e
            return

        valid_meths = {
            i for i in all_meths if (not i.startswith("_")) and (not i[0].isupper())
        }
        if self._meth not in valid_meths:
            raise ValueError(f"'method' must be one of {sorted(valid_meths)}")

    def _transform_spark(self, df):
        self._assert_col_meth(True)
        func = getattr(F.col(self._input_col), self._meth)(*self._args, **self._kwargs)
        return df.withColumn(self._output_col, func)


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
            import cpuinfo  # noqa: F401
        except ImportError:  # pragma: no cover
            msg = "'cpuinfo' optional package not installed. \n"
            msg += "Run 'pip install py-cpuinfo' or 'install nebula[cpu-info]'"
            raise ImportError(msg)

        self._n: int = n_partitions

    def _transform_spark(self, df):
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
            .show(1000, truncate=False)
        )
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

    @staticmethod
    def _transform_spark(df):
        if df.is_cached:
            return df
        return df.cache()


class SqlFunction(Transformer):
    def __init__(
            self,
            *,
            column: str,
            function: str,
            args: list[Any] | None = None,
            kwargs: dict[str, Any] | None = None,
    ):
        """Call a pyspark.sql.function with the provided args/kwargs.

        Args:
            column (str):
                Name of the column where the result of the function is stored.
            function (str):
                Name of the pyspark.sql.function to call.
            args (list(any) | None):
                Positional arguments of pyspark.sql.function. Defaults to None.
            kwargs (dict(str, any) | None):
                Keyword arguments of pyspark.sql.function. Defaults to None.
        """
        valid_funcs = {
            i for i in dir(F) if (not i.startswith("_")) and (not i[0].isupper())
        }
        assert_allowed(function, valid_funcs, "function")

        super().__init__()
        self._output_col: str = column
        self._func_name: str = function
        self._args: list = args if args else []
        self._kwargs: dict[str, Any] = kwargs if kwargs else {}

    def _transform_spark(self, df):
        func = getattr(F, self._func_name)(*self._args, **self._kwargs)
        return df.withColumn(self._output_col, func)


class Repartition(_Partitions):
    def __init__(
            self,
            *,
            num_partitions: int | None = None,
            rows_per_partition: int | None = None,
            columns: str | list[str] | None = None,
    ):
        """Return a new DataFrame partitioned by the given partitioning.

        Only one among "num_partitions", and "rows_per_partition"
        can be provided. If both are None, the dataframe will be repartitioned
        to the default value.

        The resulting DataFrame is hash partitioned.

        Args:
            num_partitions (int | None):
                Specify the target number of partitions.
            rows_per_partition (int | None):
                Coalesce the dataframe based on the desired
                number of rows per partition.
            columns (str | list(str) | None):
                Partitioning columns.
        """
        super().__init__(
            num_partitions=num_partitions,
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


# ---------------------- ALIASES ----------------------
Cache = Persist
