"""Test loop pipelines."""

import polars as pl
import pytest

from nlsn.nebula.pipelines.pipeline_loader import load_pipeline
from .auxiliaries import load_yaml
from .auxiliaries import *


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


@pytest.mark.parametrize("evaluate_loops", [True, False])
def test_invalid_split_named_loop(evaluate_loops):
    """Test a split-pipeline with for loops and a split named 'loop'."""

    def _split_func(_df):
        return {"loop": ..., "hi_values": ...}

    pipe_cfg = {
        "split_function": "split_func",
        "pipeline": {
            "loop": {"transformer": "AssertNotEmpty"},
            "hi_values": [{"transformer": "AssertNotEmpty"}],
        },
    }

    dict_split_functions = {"split_func": _split_func}

    with pytest.raises(AssertionError, match="loop"):
        load_pipeline(
            pipe_cfg,
            extra_functions=dict_split_functions,
            evaluate_loops=evaluate_loops,
        )
