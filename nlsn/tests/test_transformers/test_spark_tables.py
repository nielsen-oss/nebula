"""Unit-test for DropTable, GetTable, and RegisterTable."""

import pytest
from chispa import assert_df_equality
from pyspark.sql.types import FloatType, StringType, StructField, StructType

from nlsn.nebula.spark_util import get_spark_session
from nlsn.nebula.storage import nebula_storage as ns
from nlsn.nebula.transformers.spark_tables import (
    GetTable,
    RegisterTable,
)


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
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


class TestGetTable:

    @pytest.mark.parametrize("nebula_cache", [True, False])
    def test_error(self, df_input, nebula_cache):
        """Pass a non-existent table."""
        t = GetTable(table="t1", nebula_cache=nebula_cache)
        with pytest.raises(AssertionError):
            t.transform(df_input)

    @pytest.mark.parametrize("nebula_cache", [True, False])
    def test_valid(self, df_input, nebula_cache):
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


class TestRegisterTable:
    """Test RegisterTable."""

    @pytest.mark.parametrize("nebula_cache", [True, False])
    def test_error(self, df_input, nebula_cache):
        """Pass a non-existent table."""
        table = "t1"
        if nebula_cache:
            ns.set(table, df_input)
        else:
            df_input.createOrReplaceTempView(table)

        t = RegisterTable(table=table, mode="error", nebula_cache=nebula_cache)
        with pytest.raises(AssertionError):
            t.transform(df_input)

    @pytest.mark.parametrize("nebula_cache", [True, False])
    def test_valid(self, df_input, nebula_cache):
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
