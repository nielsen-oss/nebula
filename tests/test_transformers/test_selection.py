"""Unit-tests for 'selection' transformers."""

from itertools import product

import numpy as np
import pandas as pd
import pytest

from nebula.transformers import *
from ..auxiliaries import get_expected_columns, from_pandas

_columns_product = [f"{a}{b}" for a, b in product(["c", "d"], range(5))]


def _get_random_int_data(n_rows: int, n_cols: int) -> list[list[int]]:
    shape: tuple[int, int] = (n_rows, n_cols)  # rows x cols
    return np.random.randint(0, 100, shape).tolist()


class TestDropColumns:
    """Test DropColumns transformer."""

    @pytest.mark.parametrize(
        "backend, is_nw, kws",
        [
            ("pandas", True, {"columns": ["c1", "c3"]}),
            ("polars", True, {"glob": "c*"}),
            ("polars", False, {"regex": "c[1-5]"}),
        ],
    )
    def test_drop_columns(self, backend: str, is_nw: bool, kws):
        """Test several combinations."""
        data = _get_random_int_data(5, len(_columns_product))
        df_input = pd.DataFrame(data, columns=_columns_product)
        input_columns: list[str] = df_input.columns.tolist()

        df_input = from_pandas(df_input, backend, is_nw)

        cols2drop = get_expected_columns(input_columns, **kws)
        exp_cols = [i for i in input_columns if i not in cols2drop]

        t = DropColumns(**kws)
        df_out = t.transform(df_input)

        chk_cols = list(df_out.columns)
        assert chk_cols == exp_cols

    def test_drop_columns_not_present(self):
        """Ensure DropColumns allows not existent columns by default."""
        df_input = pd.DataFrame(
            [
                [
                    1,
                    2,
                ],
                [3, 4],
            ],
            columns=["a", "b"],
        )
        t = DropColumns(columns=["not_exists"])
        df_chk = t.transform(df_input)
        assert list(df_chk.columns) == df_input.columns.tolist()


class TestRenameColumns:
    """Test RenameColumns transformer."""

    @pytest.mark.parametrize(
        "backend, is_nw, kws, expected",
        [
            (
                    "pandas",
                    False,
                    {"mapping": {"c1": "new_c1", "d2": "new_d2"}},
                    ["c0", "new_c1", "c2", "c3", "c4", "d0", "d1", "new_d2", "d3", "d4"],
            ),
            (
                    "polars",
                    True,
                    {"columns": ["c0", "d0"], "columns_renamed": ["col_c0", "col_d0"]},
                    ["col_c0", "c1", "c2", "c3", "c4", "col_d0", "d1", "d2", "d3", "d4"],
            ),
            (
                    "polars",
                    True,
                    {"regex_pattern": "^c", "regex_replacement": "column_"},
                    [
                        "column_0",
                        "column_1",
                        "column_2",
                        "column_3",
                        "column_4",
                        "d0",
                        "d1",
                        "d2",
                        "d3",
                        "d4",
                    ],
            ),
        ],
    )
    def test_rename_columns(self, backend, is_nw, kws, expected):
        """Test several combinations."""
        data = _get_random_int_data(5, len(_columns_product))
        df_input = pd.DataFrame(data, columns=_columns_product)

        df_input = from_pandas(df_input, backend, is_nw)

        t = RenameColumns(**kws)
        df_out = t.transform(df_input)

        chk_cols = list(df_out.columns)
        assert chk_cols == expected


class TestSelectColumns:
    """Test SelectColumns transformer."""

    @pytest.mark.parametrize(
        "backend, is_nw, kws",
        [
            ("pandas", True, {"columns": "c3"}),
            ("polars", False, {"glob": "*"}),
        ],
    )
    def test_select_columns(self, backend: str, is_nw: bool, kws):
        data = _get_random_int_data(5, len(_columns_product))
        df_input = pd.DataFrame(data, columns=_columns_product)
        input_columns: list[str] = df_input.columns.tolist()

        df_input = from_pandas(df_input, backend, is_nw)

        exp_cols = get_expected_columns(input_columns, **kws)

        t = SelectColumns(**kws)
        df_out = t.transform(df_input)

        chk_cols = list(df_out.columns)
        assert chk_cols == exp_cols
