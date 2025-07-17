"""Unit-test for Pandas / Polars CreateDataFrame."""

import pandas as pd
import polars as pl
import pytest

from nlsn.nebula import nebula_storage as ns
from nlsn.nebula.shared_transformers import CreateDataFrame


class TestFromDictOfListToListOfDicts:
    """Test static method '_from_dict_of_list_to_list_of_dicts'."""

    def test_valid_input(self):
        """Valid data."""
        data = {"a": [1, 2, 3], "b": ["x", "y", "z"]}
        exp = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}]
        assert exp == CreateDataFrame._from_dict_of_list_to_list_of_dicts(data)

    def test_empty_input(self):
        """Empty data."""
        data = {"a": [], "b": []}
        # pylint: disable=use-implicit-booleaness-not-comparison)
        assert [] == CreateDataFrame._from_dict_of_list_to_list_of_dicts(data)

    def test_mismatched_lengths(self):
        """Wrong lengths."""
        data = {"a": [1, 2], "b": ["x", "y", "z"]}
        with pytest.raises(ValueError):
            CreateDataFrame._from_dict_of_list_to_list_of_dicts(data)


@pytest.fixture(scope="module", name="df_input")
def _get_df_input():
    return pd.DataFrame([["a"], ["b"]], columns=["col_1"])


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("kwargs", [None, {"columns": ["A", "B"]}])
@pytest.mark.parametrize("key", [None, "k1"])
def test_create_dataframe_pandas_polars(df_input, backend, kwargs, key):
    """Test 'CreateDataFrame' transformer using pandas / polars as backend."""
    if backend == "polars":
        if kwargs:
            # Polars does nto support 'columns' in the dataframe initialization
            return
        df_input = pl.from_pandas(df_input)

    data = {"col1": [1, 2], "col2": [3, 4]}
    t = CreateDataFrame(data=data, storage_key=key, kwargs=kwargs)
    ns.clear()
    try:
        df_out = t.transform(df_input)
        if backend == "polars":
            df_out = df_out.to_pandas()
        kwargs = kwargs or {}
        df_exp = pd.DataFrame(data, **kwargs)
        if key:
            df_stored = ns.get(key)
            if backend == "polars":
                df_input = df_input.to_pandas()
                df_stored = df_stored.to_pandas()

            pd.testing.assert_frame_equal(df_out, df_input)
            pd.testing.assert_frame_equal(df_stored, df_exp)
        else:
            pd.testing.assert_frame_equal(df_out, df_exp)
            assert not ns.isin(key)
    finally:
        ns.clear()
