"""Unit-test for StoreColumnNames."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from nlsn.nebula.shared_transformers import StoreColumnNames
from nlsn.nebula.storage import nebula_storage as ns


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_pandas_polars_store_column_names(backend):
    """Test StoreColumnNames transformer."""
    ns.clear()
    input_cols = ["a", "c", "b"]
    df_input = pd.DataFrame(np.random.random((2, 3)), columns=input_cols)
    try:
        if backend == "polars":
            df_input = pl.from_pandas(df_input)

        StoreColumnNames(key="test").transform(df_input)
        assert ns.get("test") == input_cols
    finally:
        ns.clear()
