"""Test the failure cache."""

import os

import polars as pl
import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, StringType, StructField, StructType

from nebula import TransformerPipeline
from nebula import nebula_storage as ns
from nebula.base import Transformer
from nebula.pipelines.exceptions import raise_pipeline_error
from nebula.spark_util import get_spark_session
from nebula.transformers import *
from .auxiliaries import *
from ..auxiliaries import pl_assert_equal

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
        data = [["a", 0.0], [None, 1.0]]
        df = spark.createDataFrame(data, schema=StructType(fields))

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
        """Unit-tests for 'cache-to-nebula-storage'."""
        ns.clear()

        pipe = TransformerPipeline([ThisTransformerIsBroken(), AssertNotEmpty()])

        with pytest.raises(Exception):
            pipe.run(df_input)

        df_chk = ns.get('FAIL_DF_transformer:ThisTransformerIsBroken')
        pl_assert_equal(df_chk, df_input)
        ns.clear()

    @staticmethod
    def _invalid_split_function(df):
        cond = pl.col("c4") < 2  # c4 does not exist
        return {
            "low": df.filter(cond),
            "hi": df.filter(~cond),
        }

    def test_split_pipeline_before_splitting(self, df_input):
        """Retrieve the failed DFs before appending."""
        ns.clear()
        data = {
            "low": [DropColumns(columns="c2")],
            "hi": [DropColumns(columns="c3")],
        }
        pipe = TransformerPipeline(data, split_function=self._invalid_split_function)
        with pytest.raises(Exception):
            pipe.run(df_input)

        pl_assert_equal(
            ns.get('FAIL_DF_fork:split'),
            df_input
        )
        ns.clear()

    @staticmethod
    def _split_function(df):
        cond = pl.col("c1") < 2
        return {
            "low": df.filter(cond),
            "hi": df.filter(~cond),
        }

    def test_split_pipeline_before_appending(self, df_input):
        """Retrieve the failed DFs before appending."""
        ns.clear()
        data = {
            "low": [DropColumns(columns="c2")],
            "hi": [DropColumns(columns="c3")],
        }
        pipe = TransformerPipeline(data, split_function=self._split_function)
        with pytest.raises(Exception):
            pipe.run(df_input)

        dict_df_exp = self._split_function(df_input)

        pl_assert_equal(
            ns.get('FAIL_DF_low-df-before-appending:append'),
            dict_df_exp["low"].drop("c2")
        )
        pl_assert_equal(
            ns.get('FAIL_DF_hi-df-before-appending:append'),
            dict_df_exp["hi"].drop("c3")
        )
        ns.clear()

    def test_interleaved(self, df_input):
        ns.clear()
        pipe = TransformerPipeline(
            [CallMe(), SelectColumns(glob="*"), CallMe()],
            interleaved=[ThisTransformerIsBroken()]
        )
        with pytest.raises(ValueError):
            pipe.run(df_input)

        assert ns.get("_call_me_") == 1
        df_cached = ns.get("FAIL_DF_transformer:ThisTransformerIsBroken")
        pl_assert_equal(df_input, df_cached)
        ns.clear()
