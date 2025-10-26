"""Unit-test for StoreColumnValues in pandas and polars."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from nlsn.nebula.shared_transformers import StoreColumnValues
from nlsn.nebula.storage import nebula_storage as ns


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("as_type", ["array", "list", "set", "frozenset"])
@pytest.mark.parametrize("sort", [True, False])
def test_pandas_polars_store_column_values(backend, as_type: str, sort: bool):
    """Test StoreColumnValues transformer."""
    kws = {"as_type": as_type, "sort": sort}

    # Test errors
    if (as_type in {"set", "frozenset"}) and sort:
        with pytest.raises(ValueError):
            StoreColumnValues(key="test", column="b", **kws)
        return

    ns.clear()
    input_cols = ["a", "b"]
    data = np.random.randint(0, 10, (20, len(input_cols)))
    df_input = pd.DataFrame(data, columns=input_cols)
    input_s = df_input["b"]
    try:
        if backend == "polars":
            df_input = pl.from_pandas(df_input)

        StoreColumnValues(key="test", column="b", **kws).transform(df_input)
        chk = ns.get("test")
        if as_type == "array":
            ar_exp = input_s.to_numpy()
            if sort:
                ar_exp = np.sort(ar_exp)
            np.testing.assert_array_equal(chk, ar_exp)
            return
        list_exp = input_s.to_list()
        if sort:
            list_exp = sorted(list_exp)
        if as_type == "set":
            assert chk == set(list_exp)
        elif as_type == "frozenset":
            assert chk == frozenset(list_exp)
        else:
            assert chk == list_exp

    finally:
        ns.clear()
