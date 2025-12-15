"""Unit-Test 'df_types' module."""

import os

import narwhals as nw
import pandas as pd
import pytest

from nlsn.nebula.df_types import *
from nlsn.tests.auxiliaries import from_pandas
from nlsn.tests.constants import TEST_BACKENDS


class TestGetDataframeType:
    """Unit-Test 'get_dataframe_type' function."""

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    def test_valid(self, spark, backend):
        """Test dataframe type detection for all backends."""
        df_pd = pd.DataFrame({"a": [1], "b": [2.0]})
        df = from_pandas(df_pd, backend, to_nw=False, spark=spark)
        chk = get_dataframe_type(df)
        assert chk == backend

    @staticmethod
    def test_invalid():
        """Test that invalid input raises TypeError."""
        with pytest.raises(TypeError):
            get_dataframe_type([])


@pytest.mark.parametrize("to_nw", [True, False])
class TestIsNativelySpark:
    """Unit-Test 'is_natively_spark' function."""

    @staticmethod
    @pytest.fixture(scope="class")
    def df_input(spark):
        df_pd = pd.DataFrame({"a": [1], "b": [2.0]})
        return from_pandas(df_pd, "spark", to_nw=False, spark=spark)

    @pytest.mark.skipif(os.environ.get("TESTS_NO_SPARK") == "true", reason="no spark")
    def test_is_spark(self, df_input, to_nw):
        df_input = nw.from_native(df_input) if to_nw else df_input
        assert is_natively_spark(df_input)

    @staticmethod
    def test_is_not_spark(to_nw):
        df_input = pd.DataFrame({"a": [1], "b": [2.0]})
        df_input = nw.from_native(df_input) if to_nw else df_input
        assert not is_natively_spark(df_input)
