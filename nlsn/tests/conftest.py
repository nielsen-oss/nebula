"""Test fixtures."""

import os
import sys

import pytest


@pytest.fixture(scope="session", name="data_path")
def data_path() -> str:
    """Fixture to provide a base data path."""
    return os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="session", name="spark")
def start_spark():
    """Init Spark if ee requested.

    If 'TESTS_NO_SPARK' exists as an env variable, skip this fixture and
    do not execute any test related to spark.
    """
    if os.environ.get("TESTS_NO_SPARK"):
        return

    venv_python = sys.executable
    os.environ['PYSPARK_PYTHON'] = venv_python
    os.environ['PYSPARK_DRIVER_PYTHON'] = venv_python

    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.config("spark.executor.memory", "4G")
        .config("spark.executor.cores", "2")
        .config("spark.driver.memory", "4G")
        .config("spark.sql.shuffle.partitions", 10)
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    yield spark
    spark.stop()
