"""Spark UDFs."""

import importlib
from types import ModuleType
from typing import Optional

import pyspark.sql
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, StringType, StructField, StructType

__all__ = [
    "lib_in_spark_workers",
    "lib_version_in_spark_workers",
    "udf_ensure_package_in_spark_workers",
    "udf_lib_version_in_spark_workers",
]

_psql = pyspark.sql  # keep it for linting

__CACHE: dict = {}


def __get_import_df(spark):
    """Create a mock df for the 'lib_in_spark_workers' function."""
    df = __CACHE.get("import_df")
    if df is not None:
        return df

    field = [StructField("c1", StringType(), True)]
    data = [["a"], ["b"]]
    df = spark.createDataFrame(data, schema=StructType(field)).cache()
    __CACHE["import_df"] = df
    return df


@F.udf(returnType=BooleanType())
def udf_ensure_package_in_spark_workers(s: str) -> bool:  # pragma: no cover
    """Return True if a package is installed in workers, False otherwise."""
    try:
        importlib.import_module(s)
        return True
    except (ImportError, ModuleNotFoundError):
        return False


@F.udf(returnType=StringType())
def udf_lib_version_in_spark_workers(s: str) -> str:  # pragma: no cover
    """Try to return the version of the requested package in spark workers."""

    def _look_for_version(
            _module: ModuleType, _attr: str
    ) -> Optional[str]:  # pragma: no cover
        try:
            version = getattr(_module, _attr)
        except AttributeError:
            return None

        if isinstance(version, str):
            return version

        if callable(version):
            try:
                version_called = version()
            except TypeError:
                return None

            if isinstance(version_called, str):
                return version_called

        return None

    try:
        module = importlib.import_module(s)
    except (ImportError, ModuleNotFoundError):
        return f"Module {s} is not installed on spark workers"

    for attr in ["__version__", "version"]:
        response = _look_for_version(module, attr)
        if response is not None:
            return response

    # Maybe it has the "version" module

    if hasattr(module, "version"):
        sub = getattr(module, "version")
        if isinstance(sub, ModuleType):
            for attr in ["__version__", "version"]:
                response = _look_for_version(sub, attr)
                if response is not None:
                    return response

    return "Unable to determine the package version"


def lib_in_spark_workers(spark: "pyspark.sql.SparkSession", s: str) -> bool:
    """Check if a package is installed in spark workers.

    Args:
        spark (pyspark.sql.SparkSession): Spark session.
        s (str): Name of the library to verify.

    Returns (bool):
        True if the requested package is installed, False otherwise.
    """
    df = __get_import_df(spark)

    func = udf_ensure_package_in_spark_workers(F.lit(s))

    rows = df.withColumn("response", func).select("response").collect()

    ans = {i[0] for i in rows}

    if ans == {True}:
        return True
    elif ans == {False}:
        return False

    raise RuntimeError(f"Unknown response: {ans}")  # pragma: no cover


def lib_version_in_spark_workers(spark: "pyspark.sql.SparkSession", s: str) -> str:
    """Retrieve the version of a specific package in Spark workers.

    Args:
        spark (pyspark.sql.SparkSession): Spark session.
        s (str): Name of the package to check for its version.

    Returns:
        str: The version of the specified package on Spark workers.
    """
    df = __get_import_df(spark)

    func = udf_lib_version_in_spark_workers(F.lit(s))

    rows = df.withColumn("response", func).select("response").distinct().collect()

    versions = {i[0] for i in rows}

    n_versions = len(versions)
    if not n_versions:  # pragma: no cover
        return "No version found"

    if n_versions == 1:
        return list(versions)[0]
    return f"Multiple versions found: {versions}"  # pragma: no cover
