"""Unit-Test for debug transformers."""

import pandas as pd
import pytest

from nlsn.nebula.transformers.debug import PrintSchema
from nlsn.tests.auxiliaries import from_pandas


class TestPrintSchema:
    @pytest.mark.parametrize("to_nw", [True, False])
    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test(self, to_nw: bool, backend: str):
        """Test PrintSchema transformer."""
        df_pd = pd.DataFrame(
            [
                [1, {"1": "a", "2": "b"}, [10, 11]],
                [3, {"3": "c"}, [12, 13]]
            ], columns=["a", "b", "c"]
        )

        df = from_pandas(df_pd, backend, to_nw)

        t = PrintSchema()
        df_out = t.transform(df)
        assert df is df_out  # Assert equivalence
