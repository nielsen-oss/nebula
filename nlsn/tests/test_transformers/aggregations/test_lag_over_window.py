"""Unit-test for LagOverWindow."""

import pandas as pd
import pytest

from nlsn.nebula.spark_transformers import LagOverWindow


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    data = [
        ["group_1", 2, 10000],
        ["group_1", 1, 2000],
        ["group_1", 3, 50],
        ["group_1", 4, 40],
        ["group_2", 7, 16000],
        ["group_2", 5, 12000],
        ["group_2", 2, 11000],
        ["group_3", 12, 900],
        ["group_3", 13, 600],
    ]
    schema_str = "group: string, index: int, value: int"
    return spark.createDataFrame(data, schema=schema_str).persist()


@pytest.mark.parametrize("lag", [2, -1, -2])
def test_lag_over_window(df_input, lag):
    """Test Lag transformer."""
    partition_cols = ["group"]
    order_cols = ["index"]
    lag_col = "value"
    output_col = "value_lag"

    t = LagOverWindow(
        partition_cols=partition_cols,
        order_cols=order_cols,
        lag_col=lag_col,
        lag=lag,
        output_col=output_col,
    )

    df_chk = t.transform(df_input).toPandas()

    # Assert that the number of nulls in the lagged column is equal to the number of groups
    n_nulls = df_chk[output_col].isna().sum()
    expected_nulls = int(df_chk[partition_cols].nunique()) * abs(lag)
    assert n_nulls == expected_nulls

    # Perform the same operation in Pandas
    df_exp = df_input.toPandas().sort_values(by=order_cols)
    df_exp[output_col] = df_exp.groupby(partition_cols)[lag_col].shift(lag)

    # Assert equality of dataframe
    pd.testing.assert_frame_equal(
        df_exp.sort_values(by=df_exp.columns.tolist()).reset_index(drop=True),
        df_chk.sort_values(by=df_chk.columns.tolist()).reset_index(drop=True),
    )
