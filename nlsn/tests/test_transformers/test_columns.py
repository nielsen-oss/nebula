"""Unit-test for spark DropColumns."""

import random

import pandas as pd
import pytest
from pyspark.sql.types import IntegerType, StructField, StructType

from nlsn.nebula.transformers import DropColumns
from nlsn.tests.auxiliaries import get_expected_columns
from itertools import product
from random import randint, sample

import numpy as np

_columns_product = [f"{a}{b}" for a, b in product(["c", "d"], range(5))]


def _generate_data_column_selector():
    n_rows = 10
    shape = (n_rows, len(_columns_product))
    return np.random.randint(0, 100, shape).tolist()


class TestDropColumns:

    @staticmethod
    @pytest.fixture(scope="class", name="df_input")
    def _get_df_input(spark):
        """Create initial DataFrame."""
        data = _generate_data_column_selector()

        fields = [StructField(i, IntegerType()) for i in _columns_product]
        schema = StructType(fields)
        return spark.createDataFrame(data, schema)

    @pytest.mark.parametrize("backend", ["pandas", "polars", "spark"])
    def test_drop_columns(self, spark, backend):
        """Test DropColumns transformer."""
        # Set everything to None
        columns, regex, glob = None, None, None

        # Pick just one combination. The full set is tested in Pandas/Polars
        choice = random.choice(["columns", "regex", "glob"])

        # Pick from 1 on, index 0 is None
        if choice == "columns":
            random_cols = [sample(_columns_product, randint(0, 5)) for _ in range(3)]
            columns = random.choice([None, *random_cols])
        elif choice == "regex":
            regex = random.choice([None, "c1", "c[1-5]", "c[46]"])
        else:
            glob = random.choice([None, "", "c*"])

        df_input = pd.DataFrame(_generate_data_column_selector(), columns=_columns_product)
        input_columns: list[str] = df_input.columns.tolist()

        cols2drop = get_expected_columns(input_columns, columns, regex, glob)

        exp_cols = [i for i in input_columns if i not in cols2drop]
        df_exp_pd = df_input[exp_cols]

        t = DropColumns(columns=columns, regex=regex, glob=glob)

        to_nw: bool = random.random() < 0.5
        df_out = t.transform(convert_backend(df_input, backend=backend, to_nw=to_nw))

        chk_cols = list(df_out.columns)
        assert chk_cols == exp_cols

        assert_frame_equal(df_out, df_exp)

        df_chk_pd = df_out.toPandas()
        pd.testing.assert_frame_equal(df_chk_pd, df_exp_pd)

    def test_drop_columns_not_present(self, df_input):
        """Ensure DropColumns allows not existent columns by default."""
        t = DropColumns(columns=["not_exists"])
        df_chk = t.transform(df_input)
        assert df_chk.columns == df_input.columns
