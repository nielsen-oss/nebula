"""Test pandas loop pipelines."""

import pandas as pd
import pytest

from nlsn.nebula.pipelines.pipeline_loader import load_pipeline
from nlsn.tests.test_pipelines.pipeline_yaml.auxiliaries import load_yaml


@pytest.fixture(scope="module", name="df_input")
def _get_df_input():
    data = [
        ["A"],
        ["B"],
        ["C"],
        ["C"],
    ]

    return pd.DataFrame(data, columns=["join_col"])


def test_pandas_loop_pipeline(df_input):
    """Test a nested for-loop in pandas."""
    yaml_data = load_yaml("loop.yml")
    pipe = load_pipeline(yaml_data)
    pipe.show_pipeline(add_transformer_params=True)

    df_exp = df_input.copy().drop_duplicates()
    df_exp = df_exp.assign(
        name_a=None,
        ALGO_algo_X_20=20,
        ALGO_algo_X_30=30,
        name_b="my_string",
        ALGO_algo_Y_20=20,
        ALGO_algo_Y_30=30,
    )
    df_chk = pipe.run(df_input)
    pd.testing.assert_frame_equal(df_exp, df_chk)
