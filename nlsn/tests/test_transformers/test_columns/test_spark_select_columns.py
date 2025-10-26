"""Unit-test for spark SelectColumns."""

from typing import List, Optional

import pytest

from nlsn.nebula.spark_transformers import SelectColumns
from nlsn.tests.auxiliaries import get_expected_columns
from nlsn.tests.test_transformers.test_columns._shared import SharedSelectColumns


@pytest.mark.parametrize(*SharedSelectColumns.columns_params)
@pytest.mark.parametrize(*SharedSelectColumns.regex_params)
@pytest.mark.parametrize(*SharedSelectColumns.glob_params)
def test_select_columns(
    spark,
    columns: Optional[List[str]],
    regex: Optional[str],
    glob: Optional[str],
):
    """Test SelectColumns transformer."""
    if columns is None and regex is None and glob is None:
        return
    data = SharedSelectColumns.generate_data()
    df_input = spark.createDataFrame(data, SharedSelectColumns.columns)

    t = SelectColumns(
        columns=columns,
        regex=regex,
        glob=glob,
    )
    df_out = t.transform(df_input)
    cols_chk = df_out.columns

    cols_exp = get_expected_columns(df_input, columns, regex, glob)
    assert cols_exp == cols_chk
