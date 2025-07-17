"""Unit-test for Pandas / Polars SelectColumns."""

from typing import List, Optional

import pandas as pd
import polars as pl
import pytest

from nlsn.nebula.pandas_polars_transformers import SelectColumns
from nlsn.tests.auxiliaries import get_expected_columns
from nlsn.tests.test_transformers.test_columns._shared import SharedSelectColumns


@pytest.mark.parametrize("df_type", ["pandas", "polars"])
@pytest.mark.parametrize(*SharedSelectColumns.columns_params)
@pytest.mark.parametrize(*SharedSelectColumns.regex_params)
@pytest.mark.parametrize(*SharedSelectColumns.glob_params)
def test_select_columns(
    df_type: str,
    columns: Optional[List[str]],
    regex: Optional[str],
    glob: Optional[str],
):
    """Test SelectColumns transformer."""
    if columns is None and regex is None and glob is None:
        return

    data = SharedSelectColumns.generate_data()
    if df_type == "pandas":
        df_input = pd.DataFrame(data, columns=SharedSelectColumns.columns)
    else:
        df_input = pl.DataFrame(data, schema=SharedSelectColumns.columns)

    t = SelectColumns(
        columns=columns,
        regex=regex,
        glob=glob,
    )
    df_out = t.transform(df_input)

    cols_chk = list(df_out.columns)  # pandas would use .tolist()

    cols_exp = get_expected_columns(df_input, columns, regex, glob)
    assert cols_exp == cols_chk
