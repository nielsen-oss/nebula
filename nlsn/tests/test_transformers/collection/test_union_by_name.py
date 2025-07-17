"""Unit-test for UnionByName."""

from typing import Dict

import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType
from pyspark.sql.utils import AnalysisException

from nlsn.nebula import nebula_storage as ns
from nlsn.nebula.spark_transformers import UnionByName
from nlsn.tests.constants import SPARK_VERSION

_DATA = [["a"], ["b"]]

_TABLE_1 = "df_unionByName"
_TABLE_2 = "df_unionByName_2"
_TABLE_3 = "df_unionByName_3"

_N_INPUT = len(_DATA)


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    """Define test data."""
    fields = [StructField("col_1", StringType())]

    # Dataframe columns:
    # df: "col_1"
    # df2: "col_1", "col_2"
    # df3: "col_2"
    df = spark.createDataFrame(_DATA, schema=StructType(fields)).persist()
    df2 = df.withColumn("col_2", F.lit(1))
    df3 = df2.drop("col_1")
    ns.set(_TABLE_1, df)
    ns.set(_TABLE_2, df2)
    ns.set(_TABLE_3, df3)
    df.createOrReplaceTempView(_TABLE_1)
    return df


# --------------------- Test exceptions ---------------------


@pytest.mark.parametrize("temp_view, store_key", [(None, None), ("", ""), ("x", "x")])
def test_union_by_name_with_wrong_input(temp_view, store_key):
    """Test UnionByName transformer providing 0 or 2 input tables."""
    with pytest.raises(ValueError):
        UnionByName(temp_view=temp_view, store_key=store_key)


def test_union_by_name_with_wrong_drop_parameters():
    """Test UnionByName transformer providing both drop parameters."""
    with pytest.raises(AssertionError):
        UnionByName(temp_view="x", drop_excess_columns=True, allow_missing_columns=True)


@pytest.mark.parametrize("drop", [True, False])
def test_union_by_name_drop_excess_columns(df_input, drop: bool):
    """Test UnionByName transformer setting 'drop_excess_columns'."""
    t = UnionByName(store_key=_TABLE_2, drop_excess_columns=drop)
    if drop:
        df_chk = t.transform(df_input)
        assert df_input.count() == _N_INPUT
        assert df_chk.count() == (_N_INPUT * 2)
    else:
        with pytest.raises(AnalysisException):
            t.transform(df_input)


def test_union_by_name_select_wrong_columns(df_input):
    """Test UnionByName transformer selecting wrong columns."""
    # "col_2" does not exist in _TABLE_1
    t = UnionByName(store_key=_TABLE_1, select_before_union="col_2")
    with pytest.raises(AnalysisException):
        t.transform(df_input)


# --------------------- Valid cases ---------------------


def _assert(df_input, t):
    df_chk = t.transform(df_input)
    assert df_input.count() == _N_INPUT
    assert df_chk.count() == (_N_INPUT * 2)


@pytest.mark.parametrize("src", [{"temp_view": _TABLE_1}, {"store_key": _TABLE_1}])
def test_union_by_name(df_input, src: Dict[str, str]):
    """Test UnionByName transformer w/ spark temporary views and nebula storage."""
    t = UnionByName(**src)
    _assert(df_input, t)


@pytest.mark.parametrize("select", ["col_1", ["col_1"]])
def test_union_by_name_select_columns(df_input, select):
    """Test UnionByName transformer selecting columns."""
    t = UnionByName(store_key=_TABLE_2, select_before_union=select)
    _assert(df_input, t)


@pytest.mark.parametrize("drop", ["col_2", ["col_2"]])
def test_union_by_name_drop_before_union(df_input, drop):
    """Test UnionByName transformer setting 'drop_before_union'."""
    t = UnionByName(store_key=_TABLE_2, drop_before_union=drop)
    _assert(df_input, t)


@pytest.mark.skipif(SPARK_VERSION < "3.1.0", reason="requires pyspark 3.1.0 or higher")
@pytest.mark.parametrize("store_key", [_TABLE_2, _TABLE_3])
def test_union_by_name_allow_missing_columns(df_input, store_key: str):
    """Test UnionByName transformer setting 'allow_missing_columns'."""
    t = UnionByName(store_key=store_key, allow_missing_columns=True)
    _assert(df_input, t)

    # Assert that all the missing columns have been properly created
    columns_1 = set(df_input.columns)
    columns_2 = set(ns.get(store_key).columns)
    columns_exp = columns_1.union(columns_2)
    columns_chk = set(t.transform(df_input).columns)
    assert columns_chk == columns_exp
