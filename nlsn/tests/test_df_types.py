"""Unit-Test 'df_types' module."""

import pandas as pd
import pytest

from nlsn.nebula.df_types import get_dataframe_type
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
