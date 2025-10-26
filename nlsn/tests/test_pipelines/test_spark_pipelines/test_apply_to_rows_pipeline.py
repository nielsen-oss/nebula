"""Test 'apply_to_rows' pipeline functionalities."""

import os
import random

import pytest
from chispa import assert_df_equality
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType
from pyspark.sql.utils import AnalysisException

from nlsn.nebula.pipelines._spark_split_functions import spark_split_function
from nlsn.nebula.pipelines.pipeline_loader import load_pipeline
from nlsn.nebula.storage import nebula_storage as ns
from nlsn.tests.test_pipelines._shared import DICT_APPLY_TO_ROWS_PIPELINES
from nlsn.tests.test_pipelines.pipeline_yaml.auxiliaries import load_yaml

# Available pipelines:
# - apply_to_rows_is_null_dead_end: tested with pandas & polars
# - apply_to_rows_gt: tested with pandas & polars
# - apply_to_rows_comparison_col: tested with pandas and polars
# - apply_to_rows_error
# - apply_to_rows_otherwise: tested with pandas and polars


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    """Get input dataframe."""
    fields = [
        StructField("idx", IntegerType(), True),
        StructField("c1", StringType(), True),
        StructField("c2", StringType(), True),
    ]

    data = [
        [0, "a", "b"],
        [0, "a", "b"],
        [0, "a", "b"],
        [1, "a", "a"],
        [2, "a", "a"],
        [3, "", ""],
        [4, "", ""],
        [5, None, None],
        [6, " ", None],
        [7, "", None],
        [8, "a", None],
        [9, "a", ""],
        [10, "", "b"],
        [11, "a", None],
        [12, None, "b"],
    ]

    return spark.createDataFrame(data, schema=StructType(fields)).persist()


def _assert_dfs(df_chk, df_exp):
    assert_df_equality(df_chk, df_exp, ignore_row_order=True, ignore_nullable=True)


_SOURCES = ["py", "yaml"]


def _get_pipe(name: str, source: str):
    if source == "yaml":
        yaml_date = load_yaml("apply_to_rows.yml")
        return load_pipeline(yaml_date[name])
    elif source == "py":
        return DICT_APPLY_TO_ROWS_PIPELINES[name]()
    else:
        raise RuntimeError


def _apply_to_rows_gt(df_input, pipe_name, source):
    pipeline = _get_pipe(pipe_name, source)
    pipeline.show_pipeline()

    df_out = pipeline.run(df_input)

    df_apply, df_untouched = spark_split_function(
        df_input,
        input_col="idx",
        operator="gt",
        value=5,
        compare_col=None,
    )
    df_fork = df_apply.withColumn("c1", F.lit("x"))
    return df_out, df_fork, df_untouched


@pytest.mark.skipif(os.getenv("full_nebula_test") != "true", reason="tested in pandas")
def test_apply_to_rows_gt(df_input):
    """Test 'apply_to_rows'."""
    source: str = random.choice(_SOURCES)  # Randomly take one of them
    ns.clear()

    n_exp = df_input.rdd.getNumPartitions()

    pipe_name = "apply_to_rows_gt"
    df_out, df_fork, df_untouched = _apply_to_rows_gt(df_input, pipe_name, source)
    df_exp = df_untouched.unionByName(df_fork)

    _assert_dfs(df_out, df_exp)

    n_chk = df_out.rdd.getNumPartitions()
    assert n_chk == n_exp
    ns.clear()


def test_apply_to_rows_error(df_input):
    """Test wrong 'apply_to_rows' without 'allow_missing_columns'."""
    source: str = random.choice(_SOURCES)  # Randomly take one of them
    ns.clear()

    pipe_name = "apply_to_rows_error"
    pipeline = _get_pipe(pipe_name, source)
    pipeline.show_pipeline()

    with pytest.raises(AnalysisException):
        pipeline.run(df_input)

    ns.clear()


@pytest.mark.skipif(os.getenv("full_nebula_test") != "true", reason="tested in pandas")
def test_apply_to_rows_gt_otherwise(df_input):
    """Test 'apply_to_rows' with 'otherwise' pipeline."""
    source: str = random.choice(_SOURCES)  # Randomly take one of them
    ns.clear()

    pipe_name = "apply_to_rows_otherwise"
    df_out, df_fork, df_untouched = _apply_to_rows_gt(df_input, pipe_name, source)
    df_otherwise = df_untouched.withColumn("c1", F.lit("other"))
    df_exp = df_otherwise.unionByName(df_fork)

    _assert_dfs(df_out, df_exp)
    ns.clear()
