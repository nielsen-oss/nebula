"""Unit-test for 'Assertions' transformers."""

import pandas as pd
import pytest

from nlsn.nebula.transformers.assertions import *


class TestDataFrameContainsColumns:
    """Test DataFrameContainsColumns transformer."""

    @staticmethod
    @pytest.fixture(scope="class", name="df_input")
    def _get_input_df():
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        return df

    @pytest.mark.parametrize(
        "columns, error",
        [
            ([], False),
            ("col1", False),
            (["col1"], False),
            (["col1", "col2"], False),
            (["col1", "col2", "col3"], True),
            ("col3", True),
            (["col3"], True),
        ],
    )
    def test_dataframe_contains_columns(self, df_input, columns, error: bool):
        t = DataFrameContainsColumns(columns=columns)
        if error:
            with pytest.raises(AssertionError):
                t.transform(df_input)
        else:
            t.transform(df_input)
