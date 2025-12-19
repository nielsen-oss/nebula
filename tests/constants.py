"""Constants for Unit-tests."""

import os

__all__ = ["TEST_BACKENDS"]

TEST_BACKENDS = ["pandas", "polars"]

if not os.environ.get("TESTS_NO_SPARK"):
    TEST_BACKENDS.append("spark")
