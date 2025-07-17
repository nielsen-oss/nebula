"""Unit-test for SortColumnNames."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from nlsn.nebula.shared_transformers import SortColumnNames


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_pandas_polars_sort_column_names(backend):
    """Test 'SortColumnNames' transformer."""
    input_cols = ["a", "c", "b"]
    df_input = pd.DataFrame(np.random.random((2, 3)), columns=input_cols)

    columns_exp = sorted(input_cols)

    if backend == "polars":
        df_input = pl.from_pandas(df_input)

    assert columns_exp != list(input_cols)

    df_chk = SortColumnNames().transform(df_input)
    columns_chk = sorted(df_chk.columns)

    assert columns_chk == columns_exp
