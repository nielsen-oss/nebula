"""Unit-test for Pandas / Polars DropColumns."""

from typing import List, Optional

import pandas as pd
import pytest

from nlsn.nebula.pandas_polars_transformers import DropColumns
from nlsn.tests.auxiliaries import (
    assert_pandas_polars_frame_equal,
    get_expected_columns,
    pandas_to_polars,
)

from ._shared import SharedDropColumns


@pytest.fixture(scope="module", name="df_input")
def _get_df_input():
    """Create initial DataFrame."""
    columns = SharedDropColumns.columns
    data = SharedDropColumns.generate_data()
    return pd.DataFrame(data, columns=columns)


@pytest.mark.parametrize("df_type", ["pandas", "polars"])
@pytest.mark.parametrize(*SharedDropColumns.columns_params)
@pytest.mark.parametrize(*SharedDropColumns.regex_params)
@pytest.mark.parametrize(*SharedDropColumns.glob_params)
def test_drop_columns(
    df_input,
    df_type: str,
    columns: Optional[List[str]],
    regex: Optional[str],
    glob: Optional[str],
):
    """Test DropColumns transformer."""
    if columns is None and regex is None and glob is None:
        return
    df_input = pandas_to_polars(df_type, df_input)

    input_columns = list(df_input.columns)  # pandas would use .tolist()

    cols2drop = get_expected_columns(df_input, columns, regex, glob)

    exp_cols = [i for i in input_columns if i not in cols2drop]
    df_exp = df_input[exp_cols]

    t = DropColumns(columns=columns, regex=regex, glob=glob)
    df_chk = t.transform(df_input)

    chk_cols = list(i for i in df_chk.columns)
    assert chk_cols == exp_cols

    assert_pandas_polars_frame_equal(df_type, df_exp, df_chk)


def test_drop_columns_not_present(df_input):
    """Check that DropColumns allows not present columns by default."""
    t = DropColumns(columns=["not_exists"])
    df_chk = t.transform(df_input)
    assert list(df_chk.columns) == list(df_input.columns)
