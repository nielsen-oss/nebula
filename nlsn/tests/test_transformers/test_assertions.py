"""Unit-test for AssertNotEmpty."""

import pandas as pd
import polars as pl
import pytest

from nlsn.nebula.transformers import *
from nlsn.tests.auxiliaries import from_pandas
from nlsn.tests.constants import TEST_BACKENDS


class TestAssertContainsColumns:

    @pytest.mark.parametrize(
        "cols, error",
        [
            ([], False),
            ("a", False),
            (["a"], False),
            (["a", "b"], False),
            (["a", "b", "c"], True),
            ("c", True),
            (["c"], True),
        ],
    )
    def test_dataframe_contains_columns(self, cols, error: bool):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        t = AssertContainsColumns(columns=cols)
        if error:
            with pytest.raises(AssertionError):
                t.transform(df)
        else:
            t.transform(df)


class TestAssertNotEmpty:
    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    def test_not_empty(self, spark, backend: str):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df = from_pandas(df, backend, to_nw=False, spark=spark)
        t = AssertNotEmpty()
        t.transform(df)

    @pytest.mark.parametrize("lazy", [True, False])
    def test_not_empty_polars_lazy(self, lazy: bool):
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        if lazy:
            df = df.lazy()
        t = AssertNotEmpty()
        t.transform(df)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_empty(self, backend: str):
        df = pd.DataFrame({"a": [], "b": []}).astype("int32")
        df = from_pandas(df, backend, to_nw=False)
        t = AssertNotEmpty()
        with pytest.raises(AssertionError):
            t.transform(df)

    def test_spark_empty(self, spark):
        df = spark.createDataFrame([], schema="a: int, b: int")
        t = AssertNotEmpty()
        with pytest.raises(AssertionError):
            t.transform(df)

    @pytest.mark.parametrize("lazy", [True, False])
    def test_not_empty_polars_lazy(self, lazy: bool):
        df = pl.DataFrame({})
        if lazy:
            df = df.lazy()
        t = AssertNotEmpty()
        with pytest.raises(AssertionError):
            t.transform(df)
