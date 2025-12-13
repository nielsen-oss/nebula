"""Unit-tests pipeline exceptions handling."""

import os

import pytest
from chispa import assert_df_equality
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, StringType, StructField, StructType

from nlsn.nebula import nebula_storage as ns
from nlsn.nebula.pipelines.pipelines import (
    _FAIL_CACHE,
    _PREFIX_FAIL_CACHE,
    TransformerPipeline,
    pipeline_config,
    raise_pipeline_error,
)
from nlsn.nebula.shared_transformers import WithColumn
from nlsn.nebula.spark_transformers import ChangeFieldsNullability, DropColumns, Limit
from nlsn.nebula.spark_util import null_cond_to_false

_MSG = "this custom message"


class TestExceptions:
    @staticmethod
    @pytest.mark.skipif(os.environ.get("TESTS_NO_SPARK") == "true", reason="no spark")
    def test_low_level_spark_exception(spark):
        """Check '_raise_pipeline_error' with high-level spark exception."""
        fields = [  # Two nullable fields
            StructField("with_null", StringType(), True),
            StructField("without_null", FloatType(), True),
        ]
        schema = StructType(fields)
        df = spark.createDataFrame([["a", 0.0], [None, 1.0]], schema=schema)

        t = ChangeFieldsNullability(
            nullable=False,
            columns="with_null",
            assert_non_nullable=True,
            persist=True,
        )

        try:
            try:
                # It raises Py4JJavaError
                t.transform(df.withColumn("non_nullable", F.lit("placeholder")))
            except Exception as e_inner:
                raise_pipeline_error(e_inner, _MSG)
        except Exception as e_outer:
            assert _MSG in str(e_outer)

    @staticmethod
    @pytest.mark.skipif(os.environ.get("TESTS_NO_SPARK") == "true", reason="no spark")
    def test_high_level_spark_exception(spark):
        """Check '_raise_pipeline_error' with high-level spark exception."""
        data = [
            [0.1234, "a", "b"],
            [0.1234, "a", "b"],
            [0.1234, "a", "b"],
            [1.1234, "a", "  b"],
            [2.1234, "  a  ", "  b  "],
            [3.1234, "", ""],
            [4.1234, "   ", "   "],
        ]

        fields = [
            StructField("c1", FloatType(), True),
            StructField("c2", StringType(), True),
            StructField("c3", StringType(), True),
        ]

        df1 = spark.createDataFrame(data, schema=StructType(fields))

        df2 = df1.withColumnRenamed("c3", "c4")

        try:
            try:
                # It raises AnalysisException
                df1.unionByName(df2)
            except Exception as e_inner:
                raise_pipeline_error(e_inner, _MSG)
        except Exception as e_outer:
            assert _MSG in str(e_outer)

    @staticmethod
    def test_python_exception():
        """Check '_raise_pipeline_error' with native python exception."""
        try:
            try:
                # It raises TypeError
                _ = "a" + 1
            except Exception as e_inner:
                raise_pipeline_error(e_inner, _MSG)
        except Exception as e_outer:
            assert _MSG in str(e_outer)


class ThisTransformerIsBroken:
    @staticmethod
    def transform(df):
        """Public transform method w/o parent class."""
        return df.select("wrong")


class TestCacheToNebulaStorage:
    @staticmethod
    @pytest.fixture(scope="class", name="df_input")
    def _get_df_input(spark):
        """Get input dataframe."""
        fields = [
            StructField("c1", FloatType(), True),
            StructField("c2", StringType(), True),
            StructField("c3", StringType(), True),
        ]

        data = [
            [0.1234, "a", "b"],
            [0.1234, "a", "b"],
            [0.1234, "a", "b"],
            [1.1234, "a", "  b"],
            [2.1234, "  a  ", "  b  "],
            [3.1234, "", ""],
            [4.1234, "   ", "   "],
            [5.1234, None, None],
            [6.1234, " ", None],
            [7.1234, "", None],
            [8.1234, "a", None],
            [9.1234, "a", ""],
            [10.1234, "   ", "b"],
            [11.1234, "a", None],
            [12.1234, None, "b"],
            [13.1234, None, "b"],
            [14.1234, None, None],
        ]

        return spark.createDataFrame(data, schema=StructType(fields)).persist()

    @staticmethod
    def test_flat_pipeline(df_input):
        """Unit-tests for 'cache-to-nebula-storage' in a flat pipeline."""
        ns.clear()
        _FAIL_CACHE.clear()

        pipe = TransformerPipeline([ThisTransformerIsBroken(), Limit(n=2)])

        with pytest.raises(Exception):
            pipe.run(df_input)

        name = _PREFIX_FAIL_CACHE + "ThisTransformerIsBroken"
        df_chk = ns.get(name)
        assert_df_equality(
            df_chk, df_input, ignore_row_order=True, ignore_nullable=True
        )

        ns.clear()
        _FAIL_CACHE.clear()

    @staticmethod
    def _split_function(df):
        cond = F.col("c1") < 10
        return {
            "low": df.filter(cond),
            "hi": df.filter(~null_cond_to_false(cond)),
        }

    @pytest.mark.parametrize(
        "data",
        (
            [ThisTransformerIsBroken(), Limit(n=2)],
            {"low": WithColumn(column_name="new", value="x"), "hi": Limit(n=2)},
        ),
    )
    def test_failure_cache_deactivated(self, df_input, data):
        """Unit-tests when the failure cache is not active."""
        ns.clear()
        _FAIL_CACHE.clear()
        pipeline_config["activate_failure_cache"] = False
        try:
            if isinstance(data, dict):
                split_func = self._split_function
            else:
                split_func = None
            pipe = TransformerPipeline(
                data, split_function=split_func, allow_missing_columns=False
            )
            with pytest.raises(Exception):
                pipe.run(df_input)
            assert not ns.list_keys()
        finally:
            pipeline_config["activate_failure_cache"] = True
            ns.clear()
            _FAIL_CACHE.clear()

    def test_split_pipeline(self, df_input):
        """Unit-tests for 'cache-to-nebula-storage' in a split pipeline."""
        ns.clear()
        _FAIL_CACHE.clear()

        dict_transf = {
            "low": [DropColumns(columns="c2")],
            "hi": [DropColumns(columns="c3")],
        }

        pipe = TransformerPipeline(dict_transf, split_function=self._split_function)

        with pytest.raises(Exception):
            _ = pipe.run(df_input)

        dict_df_exp = self._split_function(df_input)

        for key, df_input_exp in dict_df_exp.items():
            col2drop = "c2" if key == "low" else "c3"
            name = _PREFIX_FAIL_CACHE + key
            df_chk = ns.get(name)
            df_exp = df_input_exp.drop(col2drop)
            assert_df_equality(
                df_chk, df_exp, ignore_row_order=True, ignore_nullable=True
            )

        ns.clear()
        _FAIL_CACHE.clear()
