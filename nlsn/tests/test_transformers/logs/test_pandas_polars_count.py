"""Unit-test for pandas / polars Count transformer."""

import numpy as np
import pandas as pd
import pytest

from nlsn.nebula.pandas_polars_transformers import Count
from nlsn.nebula.storage import nebula_storage as ns
from nlsn.tests.auxiliaries import assert_pandas_polars_frame_equal, pandas_to_polars


def test_count_wrong_store_key():
    """Test Count transformer again a non-hashable store_key."""
    with pytest.raises(AssertionError):
        Count(store_key={})


@pytest.mark.parametrize("df_type", ["pandas", "polars"])
def test_count(df_type: str):
    """Test Count transformer."""
    ns.clear()
    ns.allow_overwriting()

    store_key = "test_count"

    n_rows = 10
    df = pd.DataFrame(np.random.randint(0, 10, (n_rows, 2)), columns=["a", "b"])
    df = pandas_to_polars(df_type, df)

    t = Count(store_key=store_key)
    df_out = t.transform(df)

    assert_pandas_polars_frame_equal(df_type, df, df_out)

    count_chk = ns.get(store_key)
    ns.clear()
    assert count_chk == n_rows
