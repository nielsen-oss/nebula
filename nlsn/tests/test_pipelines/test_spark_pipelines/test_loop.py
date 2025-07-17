"""Test spark loop pipelines."""

import os

import pytest
from chispa import assert_df_equality
from pyspark.sql import functions as F

from nlsn.nebula.pipelines.pipeline_loader import load_pipeline
from nlsn.tests.test_pipelines.pipeline_yaml.auxiliaries import load_yaml


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    data = [
        ["A"],
        ["B"],
        ["C"],
        ["C"],
    ]
    return spark.createDataFrame(data, schema="join_col: string").persist()


@pytest.mark.skipif(os.getenv("full_nebula_test") != "true", reason="tested in pandas")
def test_loop_pipeline(df_input):
    """Test a nested for-loop in spark."""
    yaml_data = load_yaml("loop.yml")
    pipe = load_pipeline(yaml_data)
    pipe.show_pipeline(add_transformer_params=True)

    df_exp = (
        df_input.distinct()
        .withColumn("name_a", F.lit(None))
        .withColumn("ALGO_algo_X_20", F.lit(20))
        .withColumn("ALGO_algo_X_30", F.lit(30))
        .withColumn("name_b", F.lit("my_string"))
        .withColumn("ALGO_algo_Y_20", F.lit(20))
        .withColumn("ALGO_algo_Y_30", F.lit(30))
    )
    df_chk = pipe.run(df_input)
    assert_df_equality(df_exp, df_chk, ignore_row_order=True, ignore_nullable=True)
