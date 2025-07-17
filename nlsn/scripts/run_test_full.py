"""Run all the unit-tests."""

import subprocess
from pathlib import Path

# To run all the spark tests:
# import os
# os.environ["full_nebula_test"] = "true"

if __name__ == "__main__":
    cwd = Path(__file__).parent.parent.parent.resolve()
    print(cwd)

    path_test = cwd / "nlsn" / "tests"
    list_tests = sorted(path_test.rglob("test_*.py"))

    assert list_tests

    pytest_args = [
        "pytest",
        *list(map(str, list_tests)),
        "--cov=nlsn/nebula",
        "--cov-report=term-missing",
    ]

    result = subprocess.run(
        pytest_args,
        cwd=cwd,
        capture_output=False,  # Set to True to capture output as string, False to print directly
        check=False,  # Don't raise an exception for non-zero exit codes (let pytest report failures)
    )
