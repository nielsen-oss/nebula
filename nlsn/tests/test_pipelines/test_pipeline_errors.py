"""Unit-tests pipeline exceptions handling."""

import os

import polars as pl
import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, StringType, StructField, StructType

from nlsn.nebula import nebula_storage as ns
from nlsn.nebula.base import Transformer
from nlsn.nebula.pipelines.exceptions import raise_pipeline_error
from nlsn.nebula.pipelines.pipelines import (
    _FAIL_CACHE,
    _PREFIX_FAIL_CACHE,
    TransformerPipeline,
    pipeline_config,
)
from nlsn.nebula.spark_util import get_spark_session
from nlsn.nebula.transformers import *
from nlsn.tests.test_pipelines.auxiliaries import pl_assert_equal

_MSG = "this custom message"


class ChangeFieldsNullability(Transformer):
    """Just to raise a low level Py4JJavaError."""

    def __init__(
            self,
            *,
            nullable: bool,
            columns: list[str]
    ):
        super().__init__()
        self._nullable: bool = nullable
        self._columns = columns

    def _transform_spark(self, df):

        field: StructField
        new_fields: list[StructField] = []

        for field in df.schema:
            name: str = field.name
            if name in self._columns:
                new_field = StructField(name, field.dataType, self._nullable)
                new_fields.append(new_field)
            else:
                new_fields.append(field)

        ss = get_spark_session(df)
        df_ret = ss.createDataFrame(df.rdd, StructType(new_fields))

        df_ret.count()  # trigger it
        return df_ret


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

        t = ChangeFieldsNullability(nullable=False, columns=["with_null"])

        with pytest.raises(Exception) as exc_info:
            try:
                t.transform(df.withColumn("non_nullable", F.lit("placeholder")))
            except Exception as e:
                raise_pipeline_error(e, _MSG)

        assert _MSG in str(exc_info.value)

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
    def transform(_df):
        raise ValueError("Broken transformer")


class TestCacheToNebulaStorage:
    @staticmethod
    @pytest.fixture(scope="class", name="df_input")
    def _get_df_input():
        return pl.DataFrame({
            "c1": [1, 2],
            "c2": [3, 4],
            "c3": [5, 6],
        })

    @staticmethod
    def test_flat_pipeline(df_input):
        """Unit-tests for 'cache-to-nebula-storage' in a flat pipeline."""
        ns.clear()
        _FAIL_CACHE.clear()

        pipe = TransformerPipeline([ThisTransformerIsBroken(), AssertNotEmpty()])

        with pytest.raises(Exception):
            pipe.run(df_input)

        name = _PREFIX_FAIL_CACHE + "ThisTransformerIsBroken"
        df_chk = ns.get(name)
        pl_assert_equal(df_chk, df_input)
        ns.clear()
        _FAIL_CACHE.clear()

    @staticmethod
    def _split_function(df):
        cond = pl.col("c1") < 2
        return {
            "low": df.filter(cond),
            "hi": df.filter(~cond),
        }

    @pytest.mark.parametrize("data", (
            [ThisTransformerIsBroken()],
            {"low": DropColumns(columns="c2"), "hi": DropColumns(columns="c3")},
    ))
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
        data = {
            "low": [DropColumns(columns="c2")],
            "hi": [DropColumns(columns="c3")],
        }
        pipe = TransformerPipeline(data, split_function=self._split_function)
        with pytest.raises(Exception):
            pipe.run(df_input)

        dict_df_exp = self._split_function(df_input)

        for key, df_input_exp in dict_df_exp.items():
            col2drop = "c2" if key == "low" else "c3"
            name = _PREFIX_FAIL_CACHE + key
            df_chk = ns.get(name)
            df_exp = df_input_exp.drop(col2drop)
            pl_assert_equal(df_chk, df_exp)

        ns.clear()
        _FAIL_CACHE.clear()
