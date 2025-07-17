"""Unit-test for spark DropColumns."""

import random

import pandas as pd
import pytest
from pyspark.sql.types import IntegerType, StructField, StructType

from nlsn.nebula.spark_transformers import DropColumns
from nlsn.tests.auxiliaries import get_expected_columns

from ._shared import SharedDropColumns


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    """Create initial DataFrame."""
    columns = SharedDropColumns.columns
    data = SharedDropColumns.generate_data()
    fields = [StructField(i, IntegerType()) for i in columns]
    schema = StructType(fields)
    return spark.createDataFrame(data, schema)


def test_drop_columns(df_input):
    """Test DropColumns transformer."""
    # Set everything to None
    columns, regex, glob = None, None, None

    # Pick just one combination. The full set is tested in Pandas/Polars
    choice = random.choice(["columns", "regex", "glob"])

    # Pick from 1 on, index 0 is None
    if choice == "columns":
        columns = random.choice(SharedDropColumns.columns_params[1][1:])
    elif choice == "regex":
        regex = random.choice(SharedDropColumns.regex_params[1][1:])
    else:
        glob = random.choice(SharedDropColumns.glob_params[1][1:])

    input_columns = df_input.columns

    cols2drop = get_expected_columns(df_input, columns, regex, glob)

    exp_cols = [i for i in input_columns if i not in cols2drop]
    df_exp_pd = df_input.select(exp_cols).toPandas()

    t = DropColumns(columns=columns, regex=regex, glob=glob)
    df_out = t.transform(df_input)

    chk_cols = df_out.columns
    assert chk_cols == exp_cols

    df_chk_pd = df_out.toPandas()
    pd.testing.assert_frame_equal(df_chk_pd, df_exp_pd)


def test_drop_columns_not_present(df_input):
    """Ensure DropColumns allows not existent columns by default."""
    t = DropColumns(columns=["not_exists"])
    df_chk = t.transform(df_input)
    assert df_chk.columns == df_input.columns
