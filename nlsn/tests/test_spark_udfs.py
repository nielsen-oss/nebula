"""Unit-tests for 'spark_udfs' module."""

# pylint: disable=unused-wildcard-import

from nlsn.nebula.spark_udfs import *


class TestLibInSparkWorkers:
    """Test 'lib_in_spark_workers' function."""

    @staticmethod
    def test_lib_in_spark_workers(spark):
        """Test 'lib_in_spark_workers' function."""
        assert lib_in_spark_workers(spark, "pandas")
        assert not lib_in_spark_workers(spark, "not_installed_library")

    @staticmethod
    def test_lib_version_in_spark_workers(spark):
        """Test 'lib_version_in_spark_workers' function."""
        ans = lib_version_in_spark_workers(spark, "pandas")
        assert isinstance(ans, str)

        wrong_ans = lib_version_in_spark_workers(spark, "not_installed_library")
        assert isinstance(wrong_ans, str)
