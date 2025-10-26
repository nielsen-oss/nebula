"""Unit-test for Pandas / Polars RenameColumns."""

import pandas as pd
import pytest

from nlsn.nebula.pandas_polars_transformers import RenameColumns
from nlsn.tests.auxiliaries import pandas_to_polars
from nlsn.tests.test_transformers.test_columns._shared import SharedRenameColumns


@pytest.fixture(scope="module", name="df_input")
def _get_df_input():
    return pd.DataFrame(SharedRenameColumns.data, columns=SharedRenameColumns.columns)


def test_error_init():
    """Test if errors in RenameColumns are properly raised during the initialization."""
    with pytest.raises(AssertionError):
        RenameColumns(columns=["a"], columns_renamed=["b", "c"])

    with pytest.raises(AssertionError):
        RenameColumns()

    with pytest.raises(AssertionError):
        RenameColumns(columns=["a"])

    with pytest.raises(AssertionError):
        RenameColumns(columns_renamed=["a"])

    with pytest.raises(AssertionError):
        RenameColumns(regex_pattern=r"last")

    with pytest.raises(AssertionError):
        RenameColumns(regex_replacement="new")

    with pytest.raises(AssertionError):
        RenameColumns(columns=["a"], columns_renamed=["b", "c"], mapping={"a": "b"})


def test_error_missing_columns(df_input):
    """Test if errors in RenameColumns are properly raised during the transformation."""
    t = RenameColumns(columns=["c1", "c1a"], columns_renamed=["xx", "yy"])
    with pytest.raises(AssertionError):
        t.transform(df_input)

    t = RenameColumns(mapping={"c1": "xx", "c1a": "yy"})
    with pytest.raises(AssertionError):
        t.transform(df_input)


@pytest.mark.parametrize("df_type", ["pandas", "polars"])
@pytest.mark.parametrize(*SharedRenameColumns.params)
def test_rename_columns(
    df_input,
    df_type: str,
    mapping,
    columns,
    columns_renamed,
    regex_pattern,
    regex_replacement,
    exp,
):
    """Test RenameColumns transformer."""
    t = RenameColumns(
        mapping=mapping,
        columns=columns,
        columns_renamed=columns_renamed,
        regex_pattern=regex_pattern,
        regex_replacement=regex_replacement,
    )
    df_input = pandas_to_polars(df_type, df_input)

    df_out = t.transform(df_input)
    assert list(df_out.columns) == exp

    chk_str = set(df_out[exp[0]])
    chk_flt = set(df_out[exp[1]])

    assert chk_str == SharedRenameColumns.set_str
    assert chk_flt == SharedRenameColumns.set_flt
