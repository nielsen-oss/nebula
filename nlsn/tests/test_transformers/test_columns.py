"""Unit-test for spark 'columns' transformers."""

from itertools import product

import numpy as np
import pandas as pd
import pytest
import narwhals as nw
import pyspark.sql.functions as F
from nlsn.nebula.transformers import *
from nlsn.tests.auxiliaries import get_expected_columns, from_pandas

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
            ("spark", False, {"regex": "c[1-5]"}),
        ],
    )
    def test_drop_columns(self, spark, backend: str, is_nw: bool, kws):
        """Test several combinations."""
        data = _get_random_int_data(5, len(_columns_product))
        df_input = pd.DataFrame(data, columns=_columns_product)
        input_columns: list[str] = df_input.columns.tolist()

        df_input = from_pandas(df_input, backend, is_nw, spark=spark)

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
                    "spark",
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
    def test_rename_columns(self, spark, backend, is_nw, kws, expected):
        """Test several combinations."""
        data = _get_random_int_data(5, len(_columns_product))
        df_input = pd.DataFrame(data, columns=_columns_product)

        df_input = from_pandas(df_input, backend, is_nw, spark=spark)

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
            ("spark", True, {"regex": "c[23]"}),
        ],
    )
    def test_select_columns(self, spark, backend: str, is_nw: bool, kws):
        """Test several combinations."""
        data = _get_random_int_data(5, len(_columns_product))
        df_input = pd.DataFrame(data, columns=_columns_product)
        input_columns: list[str] = df_input.columns.tolist()

        df_input = from_pandas(df_input, backend, is_nw, spark=spark)

        exp_cols = get_expected_columns(input_columns, **kws)

        t = SelectColumns(**kws)
        df_out = t.transform(df_input)

        chk_cols = list(df_out.columns)
        assert chk_cols == exp_cols


class TestAddPrefixSuffixToColumnNames:
    """Test AddPrefixSuffixToColumnNames transformer."""

    @pytest.mark.parametrize("columns", [None, [], "a_1", ["a_1", "a_2"]])
    @pytest.mark.parametrize("prefix", [None, "pre_"])
    @pytest.mark.parametrize("suffix", [None, "_post"])
    @pytest.mark.parametrize("regex", [None, "^a", "^z"])
    @pytest.mark.parametrize("glob", [None, "*", "", "a*"])
    def test(self, prefix, suffix, columns, regex, glob):
        """Test adding prefix and suffix to specific columns."""
        if not prefix and not suffix:
            with pytest.raises(AssertionError):
                AddPrefixSuffixToColumnNames(
                    columns=columns, regex=regex, glob=glob, prefix=prefix, suffix=suffix
                )
            return

        input_columns = ["a_1", "a_2", "ab_1", "ab_2"]
        data = _get_random_int_data(5, len(input_columns))
        df_input = pd.DataFrame(data, columns=input_columns)

        t = AddPrefixSuffixToColumnNames(
            columns=columns, regex=regex, glob=glob, prefix=prefix, suffix=suffix
        )
        df_out = t.transform(df_input)
        chk_cols: list[str] = list(df_out.columns)

        prefix = prefix if prefix else ""
        suffix = suffix if suffix else ""

        cols2rename = get_expected_columns(input_columns, columns=columns, regex=regex, glob=glob)
        exp_cols = [
            f"{prefix}{c}{suffix}" if c in cols2rename else c for c in df_input.columns
        ]

        assert chk_cols == exp_cols


class TestAddTypedColumns:
    """Test AddTypedColumns transformer."""

    @staticmethod
    def test_invalid():
        """Test AddTypedColumns with invalid input."""
        # Not allowed type.
        with pytest.raises(AssertionError):
            AddTypedColumns(columns="string")

        # Not pairs
        with pytest.raises(AssertionError):
            AddTypedColumns(columns=[("c1", "t1"), ("c2", "t2", "x"), ("c3", "t3")])

        # Not allowed dictionary
        with pytest.raises(AssertionError):
            AddTypedColumns(columns={1: "string"})

        # Not allowed nested dictionary
        with pytest.raises(AssertionError):
            AddTypedColumns(columns={"c1": {"type": "string", "wrong": 1}})

    @staticmethod
    @pytest.mark.parametrize("cols", [None, {}, []])
    def test_empty_input(cols):
        """Test AddTypedColumns with empty input."""
        df_input = pd.DataFrame({"a": [1, 2]})
        AddTypedColumns(columns=cols).transform(df_input)

    def test_dict_format(self):
        """Test adding typed columns with default values."""

        df_input = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        nw_df = nw.from_native(df_input)

        cols = {
            "new_str": {"type": "string", "value": "X"},
            "new_int": {"type": "int64", "value": 1},
            "new_null": "float64"
        }

        t = AddTypedColumns(columns=cols)
        df_out = t.transform(nw_df)
        df_out_native = nw.to_native(df_out)

        expected_cols = ["a", "b", "new_str", "new_int", "new_null"]
        assert sorted(df_out_native.columns) == sorted(expected_cols)

        # Check values
        assert df_out_native["new_str"].iloc[0] == "X"
        assert df_out_native["new_int"].iloc[0] == 1
        assert pd.isna(df_out_native["new_null"].iloc[0])

    @pytest.mark.parametrize("to_nw", [True, False])
    def test_list_format(self, to_nw: bool):
        """Test adding typed columns using list format."""
        df_input = pd.DataFrame({"a": [1, 2]})

        df_input = from_pandas(df_input, "pandas", to_nw)

        t = AddTypedColumns(columns=[("new_col", "string"), ("another_col", "int64")])
        df_out = t.transform(df_input)

        expected_cols = ["a", "new_col", "another_col"]
        assert sorted(df_out.columns) == sorted(expected_cols)

    @pytest.fixture(scope="class", name="df_input_spark")
    def _get_df_input_spark(self, spark):
        data = [
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
        ]
        df = pd.DataFrame(data, columns=["a", "b", "c"])
        return spark.createDataFrame(df).persist()

    @staticmethod
    def _assert_spark_fields(df_input, df_out):
        fields_chk = df_out.schema.fields
        fields_input = df_input.schema.fields
        n_input = len(fields_input)

        assert fields_input == fields_chk[:n_input]

        new_fields = fields_chk[n_input:]
        assert len(new_fields) == (len(fields_chk) - n_input)

        # Assert input column are untouched
        for c in df_input.columns:
            n_null = df_out.filter(F.col(c).isNull()).count()
            assert n_null == 0

        return new_fields

    def test_add_typed_columns_null_values(self, df_input_spark):
        """Test AddTypedColumns with valid input and null values."""
        input_columns = [
            ("c", "string"),
            ("d", "integer"),  # "integer", not "int"
        ]

        t = AddTypedColumns(columns=input_columns)
        df_out = t.transform(df_input_spark)

        new_fields = self._assert_spark_fields(df_input_spark, df_out)

        n_rows: int = df_input_spark.count()
        dict_types = dict(input_columns)

        # Assert new columns are properly cast and contains only null
        for field in new_fields:
            column_name: str = field.name
            # typeName returns 'integer' not 'int' as IntegerType name.
            type_name: str = field.dataType.typeName()

            assert dict_types[column_name] == type_name

            n_null = df_out.filter(F.col(column_name).isNull()).count()
            assert n_null == n_rows

    def test_add_typed_columns_non_null_values(self, df_input_spark):
        """Test AddTypedColumns with valid input and default values."""
        input_columns = {
            "b": {"type": "string", "value": "X"},
            "d": {"type": "integer", "value": 1},  # "integer", not "int"
            "e": "float",
        }

        t = AddTypedColumns(columns=input_columns)
        df_out = t.transform(df_input_spark).persist()

        new_fields = self._assert_spark_fields(df_input_spark, df_out)

        n_rows: int = df_input_spark.count()

        # Assert new columns are properly cast and contains only null
        for field in new_fields:
            column_name: str = field.name
            # typeName returns 'integer' not 'int' as IntegerType name.
            type_name: str = field.dataType.typeName()

            nd_input = input_columns[column_name]
            if isinstance(nd_input, dict):
                type_exp = nd_input["type"]
                default_value = input_columns[column_name]["value"]
                cond = F.col(column_name) == default_value
            else:
                type_exp = nd_input
                cond = F.col(column_name).isNull()

            assert type_exp == type_name

            n_null = df_out.filter(cond).count()
            assert n_null == n_rows
