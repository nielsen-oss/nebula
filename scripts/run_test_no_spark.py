"""Run the unit-tests that do not require spark.

When the env var TESTS_NO_SPARK is set to "true" conftest.py does not init spark.
"""

import os
import subprocess
from pathlib import Path

if __name__ == "__main__":
    os.environ["TESTS_NO_SPARK"] = "true"
    cwd = Path(__file__).parent.resolve().absolute()
    print(f"{cwd=}")

    path_test = cwd.parent / "tests"
    print(f"{path_test=}")

    paths_tests_no_spark = [i for i in path_test.rglob("test_*.py") if "spark" not in i.stem]

    assert paths_tests_no_spark

    local_tests_str = sorted([*map(str, paths_tests_no_spark)])

    pytest_args = [
        "pytest",
        *list(map(str, local_tests_str)),
        "--cov=nebula",
        "--cov-report=term-missing",
    ]

    result = subprocess.run(
        pytest_args,
        cwd=cwd,
        # Set to True to capture output as string, False to print directly
        capture_output=False,
        # Don't raise an exception for non-zero exit codes (let pytest
        # report failures)
        check=False,
    )
