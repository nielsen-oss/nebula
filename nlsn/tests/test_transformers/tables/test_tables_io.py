"""Unit-test for DropTable, GetTable, and RegisterTable."""

import pytest
from chispa import assert_df_equality
from pyspark.sql.types import FloatType, StringType, StructField, StructType

from nlsn.nebula.spark_transformers import (
    DropTable,
    GetTable,
    RegisterTable,
    SparkTableToStorage,
)
from nlsn.nebula.spark_util import get_spark_session, table_is_registered
from nlsn.nebula.storage import nebula_storage as ns


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    """Set the input dataframe."""
    fields = [
        StructField("c1", StringType(), True),
        StructField("c2", FloatType(), True),
    ]

    data = [
        ["a", 0.0],
        ["b", 1.0],
        ["c", 2.0],
        ["d", None],
    ]
    schema = StructType(fields)
    return spark.createDataFrame(data, schema=schema).persist()


@pytest.mark.parametrize("ignore_error", [True, False])
@pytest.mark.parametrize("nebula_cache", [True, False])
def test_drop_table_errors(df_input, ignore_error, nebula_cache):
    """Test DropTable passing a non existent table."""
    t = DropTable(table="t1", ignore_error=ignore_error, nebula_cache=nebula_cache)
    if ignore_error:
        t.transform(df_input)
    else:
        with pytest.raises(AssertionError):
            t.transform(df_input)


@pytest.mark.parametrize("nebula_cache", [True, False])
def test_get_table_errors(df_input, nebula_cache):
    """Test GetTable passing a non existent table."""
    t = GetTable(table="t1", nebula_cache=nebula_cache)
    with pytest.raises(AssertionError):
        t.transform(df_input)


@pytest.mark.parametrize("nebula_cache", [True, False])
def test_register_table_errors(df_input, nebula_cache):
    """Test RegisterTable passing a non existent table."""
    table = "t1"
    if nebula_cache:
        ns.set(table, df_input)
    else:
        df_input.createOrReplaceTempView(table)

    t = RegisterTable(table=table, mode="error", nebula_cache=nebula_cache)
    with pytest.raises(AssertionError):
        t.transform(df_input)


@pytest.mark.parametrize("nebula_cache", [True, False])
def test_drop_table(df_input, nebula_cache):
    """Test DropTable."""
    table = "t1"
    t = DropTable(table=table, ignore_error=False, nebula_cache=nebula_cache)

    if nebula_cache:
        # Set it and override eventually
        ns.set(table, df_input)
        t.transform(df_input)
        assert not ns.isin(table)
        return

    # Set it and override eventually
    df_input.createOrReplaceTempView(table)
    t.transform(df_input)
    ss = get_spark_session(df_input)
    assert not table_is_registered(table, ss)


@pytest.mark.parametrize("nebula_cache", [True, False])
def test_get_table(df_input, nebula_cache):
    """Test GetTable."""
    table = "t1"
    t = GetTable(table=table, nebula_cache=nebula_cache)

    # Set it and override eventually
    if nebula_cache:
        ns.set(table, df_input)
        df_chk = t.transform(df_input)
        assert df_chk is df_input  # Identity should be enough
    else:
        df_input.createOrReplaceTempView(table)
        df_chk = t.transform(df_input)
        assert_df_equality(df_input, df_chk)


@pytest.mark.parametrize("nebula_cache", [True, False])
def test_register_table(df_input, nebula_cache):
    """Test RegisterTable."""
    table = "t1"
    t = RegisterTable(table=table, mode="overwrite", nebula_cache=nebula_cache)
    t.transform(df_input)

    if nebula_cache:
        df_chk = ns.get(table)
        assert df_chk is df_input
    else:
        df_input.createOrReplaceTempView(table)
        ss = get_spark_session(df_input)
        df_chk = ss.table(table)
        assert_df_equality(df_input, df_chk)


def test_spark_table_to_storage(df_input):
    """Test SparkTableToStorage."""
    ns.clear()
    df_input.createOrReplaceTempView("t_SparkTableToStorage")
    t = SparkTableToStorage(input_table="t_SparkTableToStorage", store_key="neb")
    try:
        df_out = t.transform(df_input)
        assert df_out is df_input
        assert_df_equality(df_input, ns.get("neb"))
    finally:
        ns.clear()
