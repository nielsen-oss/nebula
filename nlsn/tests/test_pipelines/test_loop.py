"""Test loop pipelines."""

from dataclasses import dataclass

import polars as pl
import pytest

from nlsn.nebula.pipelines.pipeline_loader import load_pipeline
from nlsn.tests.test_pipelines.auxiliaries import pl_assert_equal, Distinct
from nlsn.tests.test_pipelines.pipeline_yaml.auxiliaries import load_yaml


@dataclass
class ExtraTransformers:
    Distinct = Distinct


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    data = {"join_col": ["A", "B", "C", "C"]}
    return pl.DataFrame(data)


def test_loop_pipeline(df_input):
    """Test a nested for-loop in spark."""
    yaml_data = load_yaml("loop.yml")
    pipe = load_pipeline(yaml_data, extra_transformers=[ExtraTransformers])
    pipe.show_pipeline(add_transformer_params=True)

    df_exp = (
        df_input
        .unique()
        .with_columns(
            pl.lit(None).alias("name_a"),
            pl.lit(20).alias("ALGO_algo_X_20"),
            pl.lit(30).alias("ALGO_algo_X_30"),
            pl.lit("my_string").alias("name_b"),
            pl.lit(20).alias("ALGO_algo_Y_20"),
            pl.lit(30).alias("ALGO_algo_Y_30"),
        )
    )
    df_chk = pipe.run(df_input)
    pl_assert_equal(df_chk.sort("join_col"), df_exp.sort("join_col"))
