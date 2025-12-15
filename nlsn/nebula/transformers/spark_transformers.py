"""Spark transformers related to the partitioning."""

import socket
from itertools import chain
from typing import Any, Iterable

from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    DataType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from pyspark.sql.types import MapType

from nlsn.nebula.auxiliaries import (
    assert_allowed,
    assert_at_most_one_args,
    ensure_flat_list,
    ensure_nested_length,
    is_list_uniform, validate_aggregations,
)
from nlsn.nebula.base import Transformer
from nlsn.nebula.logger import logger
from nlsn.nebula.spark_util import (
    drop_duplicates_no_randomness,
    get_data_skew,
    get_default_spark_partitions,
    get_spark_session,
)

__all__ = [
    "AggregateOverWindow",
    "Cache",  # alias Persist
    "CoalescePartitions",
    "ColumnsToMap",
    "CpuInfo",
    "LagOverWindow",
    "LogDataSkew",
    "MapToColumns",
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
        raise ValueError(msg)


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
            raise ValueError(msg)

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


class ColumnsToMap(Transformer):
    def __init__(
            self,
            *,
            output_column: str,
            columns: str | list[str] | None = None,
            regex: str | None = None,
            glob: str | None = None,
            startswith: str | Iterable[str] | None = None,
            endswith: str | Iterable[str] | None = None,
            exclude_columns: str | Iterable[str] | None = None,
            cast_values: str | None = None,
            drop_input_columns: bool = False,
    ):
        """Create a MapType field using the provided columns.

        Args:
            output_column (str):
                Name of the output column.
            columns (str | list(str) | None):
                A list of columns to select. Defaults to None.
            regex (str | None):
                Take the columns to select by using a regex pattern.
                Defaults to None.
            glob (str | None):
                Take the columns to select by using a bash-like pattern.
                Defaults to None.
            startswith (str | iterable(str) | None):
                Select all the columns whose names start with the provided
                string(s). Defaults to None.
            endswith (str | iterable(str) | None):
                Select all the columns whose names end with the provided
                string(s). Defaults to None.
            exclude_columns (str | iterable(str) | None):
                List of columns that will not be selected. Defaults to None.
            cast_values (str | DataType | None):
                If provided, cast the values to the specified type.
                Defaults to None.
            drop_input_columns (bool)
                If True, drop the input columns. Defaults to False.
        """
        super().__init__()
        self._output_column = output_column
        self._set_columns_selections(
            columns=columns,
            regex=regex,
            glob=glob,
            startswith=startswith,
            endswith=endswith,
        )
        self._excl_cols: set[str] = set(ensure_flat_list(exclude_columns))
        self._cast_value: str | None = cast_values
        self._drop: bool = drop_input_columns

    def _transform_spark(self, df):
        input_columns: list[str] = self._get_selected_columns(df)
        if self._excl_cols:
            input_columns = [i for i in input_columns if i not in self._excl_cols]

        pairs = []
        for name in input_columns:
            key = F.lit(name)
            value = F.col(name)
            if self._cast_value is not None:
                value = value.cast(self._cast_value)
            pairs.append((key, value))

        out = F.create_map(*chain.from_iterable(pairs))
        df = df.withColumn(self._output_column, out)

        if self._drop:
            return df.drop(*input_columns)
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
                The CPU name is retrieved through an UDF that runs on a mock
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
        if self._persist and (not df.is_cached):
            df = df.cache()

        dict_skew = get_data_skew(df, as_dict=True)
        n_part: int = dict_skew["partitions"]
        desc: str = dict_skew["skew"]

        logger.info(f"Number of partitions: {n_part}")
        logger.info("Rows distribution in partitions:")
        logger.info(desc)

        return df


class MapToColumns(Transformer):
    def __init__(
            self,
            *,
            input_column: str,
            output_columns: list[str] | list[list[str]] | dict[Any, str],
    ):
        """Extract keys from a MapType column and create new columns.

        Args:
            input_column (str):
                The name of the MapType column to extract keys from.
            output_columns (str, | None):
                Keys to extract to create new columns. It can be a:
                - list of strings
                - Nested list of 2-element list, where the
                    - 1st value represents the key to extract.
                    - 2nd value represents the alias for the new column.
                        It must be a string.
                - Dictionary where:
                    - the key represents the MapType key to extract
                    - the value represents the alias for the new column.
                        It must be a string.

        Raises (ValueError):
            AssertionError: If 'output_columns' is an empty list / dictionary.
            TypeError: If the input column is not a MapType.
            TypeError: If the 'output_columns' is not specified within a
                valid list or dictionary format.
        """
        if not output_columns:
            raise AssertionError("'output_columns' cannot be empty")

        super().__init__()
        self._input_col: str = input_column
        self.cols: list[tuple[str, str]]

        values: list  # A list created for checks only

        if isinstance(output_columns, (list, tuple)):
            if is_list_uniform(output_columns, str):
                self.cols = [(i, i) for i in output_columns]
                # Do not perform further checks
                return
            else:
                # Try to convert to dict to check if the list is made of
                # 2-element iterables and ensure the first element is hashable
                if not ensure_nested_length(output_columns, 2):
                    msg = "If 'output_columns' is provided as nested list "
                    msg += "all the sub-lists must have length equal to 2."
                    raise TypeError(msg)
                d = dict(output_columns)
                # Extract values to perform some checks later
                values = list(d.values())
                self.cols = output_columns

        elif isinstance(output_columns, dict):
            values = list(output_columns.values())
            self.cols = list(output_columns.items())

        else:
            msg = "'output columns' must be a (list(str)) or a "
            msg += "(list(tuple(str, str))) or a (dict(str, str))"
            raise TypeError(msg)

        if not is_list_uniform(values, str):
            msg = "All values provided for column aliases must be <string>"
            raise TypeError(msg)

        if len(set(values)) != len(output_columns):
            msg = "All values provided for column aliases must not contain duplicates"
            raise TypeError(msg)

    def _transform_spark(self, df):
        col_data_type = df.schema[self._input_col].dataType
        if not isinstance(col_data_type, MapType):
            raise TypeError(f"Column '{self._input_col}' is not of MapType.")

        items = [F.col(self._input_col).getItem(i).alias(j) for i, j in self.cols]
        return df.select(*df.columns, *items)


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


# ----------------------- WINDOW TRANSFORMERS -----------------------

_DOCSTRING_ARGS_WINDOW = """
            ascending (bool | list(bool)):
                It can be a <bool>, in this case the order will be same for
                all the column. Or it can be a list of <bool>, specifying
                which order to follow for each column. Defaults to True.
            rows_between (((str | int), (str | int)) | None):
                Creates a WindowSpec with the frame boundaries defined,
                from start (inclusive) to end (inclusive).
                Both start and end are relative positions from the current row.
                For example, “0” means “current row”, while “-1” means the row
                before the current row, and “5” means the fifth row after the
                current row.
                A row based boundary is based on the position of the row within
                the partition. An offset indicates the number of rows above
                or below the current row, the frame for the current row starts
                or ends. For instance, given a row based sliding frame with a
                lower bound offset of -1 and an upper bound offset of +2.
                The frame for row with index 5 would range from index 4 to
                index 7.
                Defaults to None.
            range_between (((str | int), (str | int)) | None):
                Creates a WindowSpec with the frame boundaries defined, from
                start (inclusive) to end (inclusive).
                Both start and end are relative from the current row. For
                example, “0” means “current row”, while “-1” means one off
                before the current row, and “5” means the five off after
                the current row.
                A range-based boundary is based on the actual value of the
                ORDER BY expression(s). An offset is used to alter the value
                of the ORDER BY expression, for instance if the current
                ORDER BY expression has a value of 10 and the lower bound
                offset is -3, the resulting lower bound for the current row
                will be 10 - 3 = 7. This however puts a number of constraints
                on the ORDER BY expressions: there can be only one expression
                and this expression must have a numerical data type. An
                exception can be made when the offset is unbounded, because
                no value modification is needed, in this case multiple and
                non-numeric ORDER BY expression are allowed.
                If provided, the parameter 'order_cols' must be passed as well.
                Defaults to None.

        Notes for 'rows_between' and 'range_between':
            To use the very first boundary for left boundary, the user can
            provide the string "start" instead of an integer. "start"
            represents the 'Window.unboundedPreceding' default value.
            To use the very last boundary for the right boundary, the user
            can provide the string "end" instead of an integer.
            "end" represents the 'Window.unboundedFollowing' default value.
"""

_ALLOWED_WINDOW_AGG: set[str] = {
    "avg",
    "collect_list",
    "collect_set",
    "countDistinct",
    "first",
    "last",
    "max",
    "mean",
    "min",
    "stddev",
    "sum",
    "variance",
}


def validate_window_frame_boundaries(start, end) -> tuple[int, int]:
    """Validate the window frame boundaries."""
    if (start is None) or (end is None):
        raise ValueError("'start' and 'end' cannot be None")
    if isinstance(start, str):
        if start != "start":
            raise ValueError("if 'start' is <str> must be 'start'")
        start = Window.unboundedPreceding
    if isinstance(end, str):
        if end != "end":
            raise ValueError("if 'end' is <str> must be 'end'")
        end = Window.unboundedFollowing
    return start, end


def _expand_ascending_windowing_cols(ascending, order_cols) -> list[str]:
    """Expand ascending to fill the missing orders."""
    n_ascending = len(ascending)
    n_order_cols = len(order_cols)

    if n_ascending == 1:
        return ascending * n_order_cols

    if n_order_cols != n_ascending:
        msg = f"Length of order columns: {n_order_cols} does not "
        msg += f"match length of sort criteria: {n_ascending}."
        raise ValueError(msg)
    return ascending


class _Window(Transformer):
    def __init__(
            self,
            *,
            partition_cols: str | list[str] | None,
            order_cols: str | list[str] | None,
            ascending: bool | list[bool],
            rows_between: tuple[str | int, str | int],
            range_between: tuple[str | int, str | int],
    ):
        if range_between and not order_cols:
            msg = "If 'range_between' is provided 'order_cols' must be set as well."
            raise ValueError(msg)

        super().__init__()
        self._partition_cols: list[str] = ensure_flat_list(partition_cols)

        self._list_order_cols: list[str] = ensure_flat_list(order_cols)

        self._ascending: list[str] = _expand_ascending_windowing_cols(
            ensure_flat_list(ascending), self._list_order_cols
        )

        self._order_cols: list[F.col] = [
            F.col(i).asc() if j else F.col(i).desc()
            for i, j in zip(self._list_order_cols, self._ascending)
        ]

        self._rows_between: tuple[int, int] | None = None
        if rows_between:
            self._rows_between = validate_window_frame_boundaries(*rows_between)

        self._range_between: tuple[int, int] | None = None
        if range_between:
            self._range_between = validate_window_frame_boundaries(*range_between)

    @property
    def _get_window(self):
        window = Window
        if self._partition_cols:
            window = window.partitionBy(self._partition_cols)

        if self._order_cols:
            window = window.orderBy(self._order_cols)

        if self._rows_between:
            window = window.rowsBetween(*self._rows_between)

        if self._range_between:
            window = window.rangeBetween(*self._range_between)

        return window


class AggregateOverWindow(_Window):
    def __init__(
            self,
            *,
            partition_cols: str | list[str] | None = None,
            aggregations: list[dict[str, str]] | dict[str, str],
            order_cols: str | list[str] | None = None,
            ascending: bool | list[bool] = True,
            rows_between: tuple[str | int, str | int] = None,
            range_between: tuple[str | int, str | int] = None,
    ):  # noqa: D208, D209
        """Aggregate over a window.

        It returns the original dataframe with attached new columns defined
        in aggregations.

        Args:
            partition_cols (str | list(str) | None):
                Columns to partition on. If set to None, the entire DataFrame is
                considered as a single partition. Defaults to None.
            aggregations (list(dict(str, str)) | dict(str, str)):
                A list of aggregation dictionaries to be applied.
                If a single aggregation is provided (equivalent to a list of
                length=1), it can be a flat dictionary.
                Each aggregation is defined with the following fields:
                'col': (the column to aggregate)
                'agg': (the aggregation operation)
                'alias': (the alias for the aggregated column)
                Eg:
                [
                    {"agg": "avg", "col": "dollars", "alias": "mean_dollars"},
                    {"agg": "sum", "col": "dollars", "alias": "tot_dollars"},
                ]
                "alias" is a necessary key, and to prevent unexpected behavior
                it cannot be a column used in neither 'partition_col' nor in
                'order_cols', even though it can a column used in
                'aggregations', but keep in mind that aggregations are computed
                sequentially, not in parallel.
            order_cols (str | list(str) | None):
                Columns to order the partition by. If provided, the partition
                will be ordered based on these columns. Defaults to None."""
        assert_at_most_one_args(
            rows_between=rows_between, range_between=range_between
        )

        if isinstance(aggregations, dict):
            aggregations = [aggregations]

        validate_aggregations(
            aggregations, _ALLOWED_WINDOW_AGG, exact_keys={"agg", "col", "alias"}
        )

        super().__init__(
            partition_cols=partition_cols,
            order_cols=order_cols,
            ascending=ascending,
            rows_between=rows_between,
            range_between=range_between,
        )

        self._aggregations: list[dict[str, str]] = aggregations
        self._aliases: set[str] = {i["alias"] for i in self._aggregations}
        self._check_alias_override(self._partition_cols, "partition_cols")
        self._check_alias_override(self._list_order_cols, "order_cols")

    def _check_alias_override(self, cols: list[str], name: str):
        ints = self._aliases.intersection(cols)
        if ints:
            raise AssertionError(f'Some aliased override "{name}": {ints}')

    def _transform_spark(self, df):
        list_agg: list[tuple[F.col, str]] = []
        for el in self._aggregations:
            agg: F.col = getattr(F, el["agg"])(el["col"])
            alias: str = el["alias"]
            list_agg.append((agg, alias))

        # Check duplicate names for final cols, in case we keep only aliases
        return_cols = [i for i in df.columns if i not in self._aliases]

        ints = self._aliases & set(df.columns)
        if ints:
            logger.warning(f"Overlapping column names: {ints} - keeping only the alias")

        win = self._get_window
        windowed_cols = [c.over(win).alias(alias) for c, alias in list_agg]

        return df.select(*return_cols, *windowed_cols)


class LagOverWindow(_Window):
    def __init__(
            self,
            *,
            partition_cols: str | list[str] | None = None,
            order_cols: str | list[str] | None = None,
            lag_col: str,
            lag: int,
            output_col: str,
            ascending: bool | list[bool] = True,
            rows_between: tuple[str | int, str | int] = None,
            range_between: tuple[str | int, str | int] = None,
    ):  # noqa: D208, D209
        """Aggregate over a window.

        It returns the original dataframe with attached new columns defined
        in aggregations.

        Args:
            partition_cols (str | list(str) | None):
                Columns to partition on. If set to None, the entire DataFrame is
                considered as a single partition. Defaults to None.
            order_cols (str | list(str) | None):
                Columns to order the partition by. If provided, the partition
                will be ordered based on these columns. Defaults to None.
            lag_col (str):
                Column to be windowed by the lag defined in the 'lag' parameter.
            output_col (str):
                Name of the output column containing the windowed result.
        """
        assert_at_most_one_args(
            rows_between=rows_between, range_between=range_between
        )

        super().__init__(
            partition_cols=partition_cols,
            order_cols=order_cols,
            ascending=ascending,
            rows_between=rows_between,
            range_between=range_between,
        )

        self._lag_col: str = lag_col
        self._output_col: str = output_col
        self._lag: int = lag

    def _transform_spark(self, df):
        win = self._get_window
        return df.withColumn(
            self._output_col, F.lag(self._lag_col, self._lag).over(win)
        )


AggregateOverWindow.__init__.__doc__ = (
        AggregateOverWindow.__init__.__doc__ + _DOCSTRING_ARGS_WINDOW
)
LagOverWindow.__init__.__doc__ = LagOverWindow.__init__.__doc__ + _DOCSTRING_ARGS_WINDOW

# ----------------------------- ALIASES -----------------------------
Cache = Persist
