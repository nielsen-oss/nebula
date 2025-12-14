"""Run the unit-tests that do not require spark.

When the env var TESTS_NO_SPARK is set to "true" conftest.py does not init spark.
"""

import os
from pathlib import Path

import pytest

if __name__ == "__main__":
    os.environ["TESTS_NO_SPARK"] = "true"
    path_test = Path("..") / "tests"

    paths_tests_no_spark = [
        i for i in path_test.rglob("test_*.py") if "spark" not in i.stem
    ]

    local_tests_str = sorted([*map(str, paths_tests_no_spark)])
    pytest.main(local_tests_str)
