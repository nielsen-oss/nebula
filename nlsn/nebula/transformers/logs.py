"""Logging and Monitoring transformers.

They don't manipulate the data but may trigger eager evaluation.
"""

import socket

from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from nlsn.nebula.base import Transformer
from nlsn.nebula.spark_util import get_spark_session

__all__ = [
    "CpuInfo",
    # "LogDataSkew",
]


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
