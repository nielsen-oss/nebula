"""Unit-test for Pandas / Polars CreateDataFrame."""
import os

import pandas as pd
import pytest
from chispa import assert_df_equality

from nlsn.nebula import nebula_storage as ns
from nlsn.nebula.spark_util import is_broadcast
from nlsn.nebula.transformers.collections import InjectData
from nlsn.tests.auxiliaries import from_pandas, to_pandas


class TestInjectData:  # FIXME: move to keyword
    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    @pytest.mark.parametrize("to_nw", [True, False])
    def test_pandas_polars(self, backend: str, to_nw: bool):
        data = {"c1": [1, 2], "c2": [3, 4]}
        df_exp = pd.DataFrame(data)

        # Create a random df just to define the backend
        df_input_pd = pd.DataFrame([["a"], ["b"]], columns=["col"])
        df_input = from_pandas(df_input_pd, backend, to_nw=to_nw)
        t = InjectData(data=data, storage_key="k1")
        ns.clear()
        try:
            df_out = t.transform(df_input)
            df_out_pd = to_pandas(df_out)

            # input and output dataframes must be unchanged
            pd.testing.assert_frame_equal(df_input_pd, df_out_pd)

            # Verify the stored DF
            df_stored = ns.get("k1")
            # if to_nw:  # FIXME: restore when is keyword
            #     assert isinstance(df_stored, (nw.DataFrame, nw.LazyFrame)), type(df_stored)
            # else:
            #     if backend == "pandas":
            #         assert isinstance(df_stored, pd.DataFrame)
            #     else:
            #         assert isinstance(df_stored, pl.DataFrame)
            df_stored_pd = to_pandas(df_stored)
            pd.testing.assert_frame_equal(df_stored_pd, df_exp)
        finally:
            ns.clear()

    @pytest.mark.skipif(os.environ.get("TESTS_NO_SPARK") == "true", reason="no spark")
    @pytest.mark.parametrize("broadcast", [True, False])
    def test_spark(self, spark, broadcast):
        data = {"c1": [1, 2], "c2": [3, 4]}
        df_input_pd = pd.DataFrame(data)
        df_input = spark.createDataFrame(df_input_pd)  # FIXME: annoying ..
        df_exp = spark.createDataFrame(pd.DataFrame(data))

        t = InjectData(data=data, storage_key="k1", broadcast=broadcast)
        ns.clear()
        try:
            df_out = t.transform(df_input)

            df_stored = ns.get("k1")
            if broadcast:
                assert is_broadcast(df_stored)
            assert_df_equality(df_stored, df_exp, ignore_row_order=True)

            assert_df_equality(df_out, df_input, ignore_row_order=True)
        finally:
            ns.clear()
