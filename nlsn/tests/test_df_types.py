"""Unit-Test 'df_types' module."""

import pandas as pd
import pytest

from nlsn.nebula.df_types import get_dataframe_type
from nlsn.tests.auxiliaries import from_pandas


class TestGetDataframeType:
    """Unit-Test 'get_dataframe_type' function."""

    @pytest.mark.parametrize("backend", ["pandas", "polars", "spark"])
    def test_valid(self, spark, backend):
        df_pd = pd.DataFrame({"a": [1], "b": [2.0]})
        df = from_pandas(df_pd, backend, to_nw=False, spark=spark)
        chk = get_dataframe_type(df)
        assert chk == backend

    @staticmethod
    def test_invalid():
        with pytest.raises(TypeError):
            get_dataframe_type([])
