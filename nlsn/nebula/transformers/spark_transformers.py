"""Spark transformers related to the partitioning."""

import socket
from typing import Any, Iterable

from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    DataType,
    IntegerType,
    MapType,
    StringType,
    StructField,
    StructType,
)

from nlsn.nebula.auxiliaries import ensure_flat_list, assert_allowed
from nlsn.nebula.base import Transformer
from nlsn.nebula.logger import logger
from nlsn.nebula.spark_util import (
    cache_if_needed,
    drop_duplicates_no_randomness,
    get_data_skew,
    get_default_spark_partitions,
    get_spark_session,
)

__all__ = [
    "Cache",  # alias Persist
    "CoalescePartitions",
    "CpuInfo",
    "LogDataSkew",
    "Persist",  # alias Cache
    "Repartition",
    "SparkColumnMethod",
    "SparkDropDuplicates",
    "SparkExplode",
    "SparkSqlFunction",
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
        return df if df.is_cached else df.cache()


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


class SparkColumnMethod(Transformer):
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


class SparkDropDuplicates(Transformer):
    def __init__(
            self,
            *,
            columns: str | list[str] | None = None,
            regex: str | None = None,
            glob: str | None = None,
            startswith: str | Iterable[str] | None = None,
            endswith: str | Iterable[str] | None = None,
    ):
        """Perform spark `drop_duplicates` operation.

        Input parameters are eventually used to select a subset of the columns.
        In such cases, the 'drop_duplicates_no_randomness' function is used
        to minimize randomness; otherwise, a bare 'drop_duplicates()' or
        '.distinct()' method is used.

        Args:
            columns (str | list(str) | None):
                List of the subset columns. Defaults to None.
            regex (str | None):
                Select the subset columns to select by using a regex pattern.
                Defaults to None.
            glob (str | None):
                Select the subset columns by using a bash-like pattern.
                Defaults to None.
            startswith (str | iterable(str) | None):
                Select all the subset columns whose names start with the provided
                string(s). Defaults to None.
            endswith (str | iterable(str) | None):
                Select all the subset columns whose names end with the provided
                string(s). Defaults to None.
        """
        super().__init__()
        self._set_columns_selections(
            columns=columns,
            regex=regex,
            glob=glob,
            startswith=startswith,
            endswith=endswith,
        )

    def _transform_spark(self, df):
        subset: list[str] = self._get_selected_columns(df)
        if subset and (set(subset) != set(list(df.columns))):
            return drop_duplicates_no_randomness(df, subset)
        return df.drop_duplicates()

class SparkExplode(Transformer):
    def __init__(
            self,
            *,
            input_col: str,
            output_cols: str | list[str] | None = None,
            outer: bool = True,
            drop_after: bool = False,
    ):
        """Explode an array column into multiple rows.

        Args:
            input_col (str):
                Column to explode.
            output_cols (str | None):
                Where to store the values.
                If the Column to explode is an <ArrayType>, 'output_cols'
                can be null and the exploded values inside the input column.
                Otherwise, if the Column to explode is a <MapType>,
                'output_cols' must be a 2-element <list> or <tuple> of string,
                representing the key and the value respectively.
            outer (bool):
                Whether to perform an outer-explode (null values are preserved).
                If the Column to explode is an <ArrayType>, it will preserve
                empty arrays and produce a null value as output.
                If the Column to explode is an <MapType>, it will preserve empty
                dictionaries and produce a null values as key and value output.
                Defaults to True.
            drop_after (bool):
                If to drop input_column after the F.explode.
        """
        if isinstance(output_cols, (list, tuple)):
            n = len(output_cols)
            msg = "If 'output_cols' is an iterable it must "
            msg += "be a 2-element <list> or <tuple> of string."
            if n != 2:
                raise AssertionError(msg)
            if not all(isinstance(i, str) for i in output_cols):
                raise AssertionError(msg)

        super().__init__()
        self._input_col: str = input_col
        self._output_cols: list[str] | str = output_cols or input_col
        self._outer: bool = outer
        self._drop_after: bool = drop_after

    def _transform_spark(self, df):
        explode_method = F.explode_outer if self._outer else F.explode

        input_type: DataType = df.select(self._input_col).schema[0].dataType

        if isinstance(input_type, ArrayType):
            if not isinstance(self._output_cols, str):  # pragma: no cover
                msg = "If the column to explode is <ArrayType> the 'output_col' "
                msg += "parameter must be a <str>."
                raise AssertionError(msg)
            func = explode_method(self._input_col)
            ret = df.withColumn(self._output_cols, func)

        elif isinstance(input_type, MapType):
            if not isinstance(self._output_cols, (list, tuple)):
                msg = "If the column to explode is <MapType> the 'output_cols' "
                msg += "parameter must be a 2 element <list>/<tuple> of <str>."
                raise AssertionError(msg)
            not_exploded = [i for i in df.columns if i not in self._output_cols]
            exploded = explode_method(self._input_col).alias(*self._output_cols)
            ret = df.select(*not_exploded, exploded)

        else:
            msg = "Input type not understood. Accepted <ArrayType> and <MapType>"
            raise AssertionError(msg)

        # Only if input col is different from output col
        if self._drop_after and self._input_col != self._output_cols:
            ret = ret.drop(self._input_col)

        return ret


class SparkSqlFunction(Transformer):
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


# ---------------------- ALIASES ----------------------
Cache = Persist
