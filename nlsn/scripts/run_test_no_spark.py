"""Run some unit-tests that do not require spark.

When the env var TESTS_NO_SPARK is set, conftest.py does not init spark.
"""

import os
from pathlib import Path

import pytest

if __name__ == "__main__":
    # FIXME: just /test and rglob

    os.environ["TESTS_NO_SPARK"] = "true"

    path_test = Path("..") / "tests"
    # path_pipe = path_test / "test_pipelines"
    path_trf = path_test / "test_transformers"

    paths_tests_no_spark = [
        *[i for i in path_test.glob("test_*.py") if "spark" not in i.stem],
        *[i for i in path_trf.glob("test_*.py") if "spark" not in i.stem]
    ]
    local_tests_str = sorted([*map(str, paths_tests_no_spark)])
    pytest.main(local_tests_str)
