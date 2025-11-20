"""Run some unit-tests that do not require spark.

When the env var TESTS_NO_SPARK is set, conftest.py does not init spark.
"""

import os
from pathlib import Path

import pytest

if __name__ == "__main__":
    os.environ["TESTS_NO_SPARK"] = "true"

    path_test = Path("..") / "tests"
    path_pipe = path_test / "test_pipelines"
    path_trf = path_test / "test_transformers"

    list_tests_pipe_no_spark = sorted(path_pipe.glob("*.py"))
    assert list_tests_pipe_no_spark

    list_test_trf_pandas = sorted(path_trf.rglob("test_pandas_*.*py"))

    local_tests = [
        # transformer syntax
        path_test / "test_transformer_syntax.py",
        # utils and auxiliaries
        path_test / "test_auxiliaries.py",
        path_test / "test_base.py",
        path_test / "test_helpers.py",
        path_test / "test_metaclasses.py",
        path_test / "test_nebula_storage.py",
        # pandas transformers
        *list_test_trf_pandas,
        # pipeline w/o spark
        path_pipe / "test_pandas_polars_pipelines",
        *list_tests_pipe_no_spark,
        path_trf / "aggregations" / "test_validations.py",
    ]

    local_tests_str = [*map(str, local_tests)]

    pytest.main(local_tests_str)
