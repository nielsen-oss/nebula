"""Unit-tests for pipeline utils."""

import os

import narwhals as nw
import pandas as pd
import polars as pl
import pytest
from pyspark.sql.types import FloatType, LongType, StringType, StructField, StructType

from nebula.pipelines.util import *
from nebula.transformers import *
from ...auxiliaries import from_pandas, to_pandas


@pytest.mark.parametrize("add_params", [True, False])
@pytest.mark.parametrize("max_len", [-1, 0, 100])
@pytest.mark.parametrize("wrap_text", [True, False])
@pytest.mark.parametrize("as_list", [True, False])
def test_get_transformer_name(add_params, max_len, wrap_text, as_list):
    """Test 'get_transformer_name' function."""
    cols_select: list[str] = ["this_column_is_23_chars"] * 100
    param_len_full: int = len("".join(cols_select))
    t = SelectColumns(columns=cols_select)
    kwargs = {
        "add_params": add_params,
        "max_len": max_len,
        "wrap_text": wrap_text,
        "as_list": as_list,
    }
    if add_params and wrap_text and as_list:
        with pytest.raises(ValueError):
            get_transformer_name(t, **kwargs)
        return

    chk = get_transformer_name(t, **kwargs)

    base_len = len(f"{t.__class__.__name__} -> PARAMS: ") * 1.1  # keep a margin

    if as_list:
        assert isinstance(chk, list)
        if not add_params:
            return
        n_chk = sum(len(i) for i in chk)
        if max_len <= 0:
            assert n_chk > param_len_full, chk
        else:
            assert n_chk <= (base_len + max_len), chk
    else:
        assert isinstance(chk, str)
        if not add_params:
            return
        n_chk = len(chk)
        if max_len <= 0:
            assert n_chk > param_len_full, chk
        else:
            assert n_chk <= (base_len + max_len), chk


def _func1():
    pass


def _func2():
    pass


def _func3():
    pass


_expected_dict_extra_func = {"_func1": _func1, "_func2": _func2, "_func3": _func3}


@pytest.mark.parametrize(
    "funcs, expected",
    [
        ({}, {}),
        ([], {}),
        (_func2, {"_func2": _func2}),
        (_expected_dict_extra_func, _expected_dict_extra_func),
        ([_func1, _func2, _func3], _expected_dict_extra_func),
    ],
)
def test_create_dict_extra_functions(funcs, expected):
    """Test 'create_dict_extra_functions' function."""
    chk = create_dict_extra_functions(funcs)
    assert chk == expected


@pytest.mark.parametrize(
    "o",
    [
        "wrong",
        [_func1, _func2, _func2],
        [_func1, _func2, 1],
        {"_func1": _func1, "_func2": "_func2"},
    ],
)
def test_create_dict_extra_functions_error(o):
    """Test 'create_dict_extra_functions' function with wrong arguments."""
    with pytest.raises(AssertionError):
        create_dict_extra_functions(o)


class TestSplitDf:
    """Test the split_df function with Polars."""

    @pytest.fixture(scope="class", name="df_input")
    def _get_df_input(self):
        return pl.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "status": ["active", "inactive", "active", "pending", "active"],
            "score": [85, 42, 91, None, 73],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        })

    def test(self, df_input):
        cfg = {"input_col": "score", "operator": "gt", "value": 80}
        df_1_chk_nw, df_2_chk_nw = split_df(df_input, cfg)
        df_1_chk = nw.to_native(df_1_chk_nw)
        df_2_chk = nw.to_native(df_2_chk_nw)

        # Scores > 80: 85, 91 (2 rows)
        df_1_exp = df_input.filter(pl.col("score") > 80)
        pl.testing.assert_frame_equal(df_1_chk, df_1_exp)

        # Scores <= 80 or null: 42, None, 73 (3 rows)
        df_2_exp = df_input.filter((pl.col("score") <= 80) | pl.col("score").is_null())
        pl.testing.assert_frame_equal(df_2_chk, df_2_exp)


class TestToSchema:
    """Test suite for to_schema function."""

    @pytest.fixture(scope="class", name="list_dfs")
    def _get_list_dfs(self) -> list[pd.DataFrame]:
        df1 = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10.5, 20.5, 30.5],
            "name": ["a", "b", "c"]
        })

        df2 = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10, 20, 30],
            "name": ["a", "b", "c"]
        })

        df3 = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [1.2, float("nan"), 3.1],
            "name": ["a", "b", "c"]
        })

        return [df1, df2, df3]

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    @pytest.mark.parametrize(
        "n, to_nw",
        [
            (1, None),
            (1, "all"),
            (None, 1),
            (None, None),
            (None, "all"),
        ]
    )
    def test_pandas(self, backend: str, list_dfs, n, to_nw):
        dtypes = {"id": "int64", "value": "float32"}
        dataframes = list_dfs[:n]
        expected = [i.astype(dtypes) for i in dataframes]

        dataframes = [from_pandas(i, backend, to_nw=False, spark=None) for i in dataframes]

        if to_nw == 1:
            dataframes[1] = nw.from_native(dataframes[1])
        elif to_nw == "all":
            dataframes = [nw.from_native(i) for i in dataframes]

        pl_schema = {"id": pl.Int64, "value": pl.Float32}
        result = to_schema(dataframes, dtypes if backend == "pandas" else pl_schema)
        if to_nw is not None:
            result = [nw.to_native(i) for i in result]

        result = [to_pandas(i) for i in result]

        for df_chk, df_exp in zip(result, expected):
            pd.testing.assert_frame_equal(df_chk, df_exp, check_dtype=True)

    @pytest.mark.skipif(os.environ.get("TESTS_NO_SPARK") == "true", reason="no spark")
    @pytest.mark.parametrize("to_nw", [True, False])
    def test_spark(self, spark, list_dfs, to_nw):
        df_input_pd = list_dfs[0]
        df_input = from_pandas(df_input_pd, "spark", to_nw=to_nw, spark=spark)

        spark_schema = [
            StructField("id", LongType(), True),
            StructField("value", FloatType(), True),
            StructField("name", StringType(), True),
        ]

        df_chk = to_schema([df_input], StructType(spark_schema))[0]
        if to_nw:
            df_chk = nw.to_native(df_chk)

        assert df_chk.schema[1].dataType.typeName() == "float"  # in spark is 32bit
        df_exp = df_input_pd.copy().astype({"value": "float32"})
        pd.testing.assert_frame_equal(df_chk.toPandas(), df_exp, check_dtype=True)
