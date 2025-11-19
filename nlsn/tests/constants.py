"""Constants for Unit-tests."""

import os

__all__ = ["SPARK_VERSION", "TEST_BACKENDS"]

SPARK_VERSION: str
TEST_BACKENDS = ["pandas", "polars"]

if os.environ.get("LOCAL_TESTS_NO_SPARK"):
    # if LOCAL_TESTS_NO_SPARK exists as env variable skip this fixture and
    # do not execute any test related to spark.
    SPARK_VERSION = "0.0.0"
else:
    import pyspark

    SPARK_VERSION = pyspark.__version__
    TEST_BACKENDS.append("spark")
