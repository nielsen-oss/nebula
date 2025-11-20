"""Constants for Unit-tests."""

import os

__all__ = ["TESTS_NO_SPARK", "TEST_BACKENDS"]

TESTS_NO_SPARK: bool
TEST_BACKENDS = ["pandas", "polars"]

if os.environ.get("TESTS_NO_SPARK"):
    # if TESTS_NO_SPARK exists as env variable skip this fixture and
    # do not execute any test related to spark.
    TESTS_NO_SPARK = False
else:

    TESTS_NO_SPARK = True
    TEST_BACKENDS.append("spark")
