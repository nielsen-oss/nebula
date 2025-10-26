"""Unit-test for Spark CreateDataFrame."""

import pandas as pd
import pytest
from chispa import assert_df_equality
from pyspark.sql.types import StringType, StructField, StructType

from nlsn.nebula.shared_transformers import CreateDataFrame
from nlsn.nebula.spark_util import is_broadcast
from nlsn.nebula.storage import nebula_storage as ns


@pytest.fixture(scope="module", name="df_input")
def _get_df(spark):
    fields = [
        StructField("c1", StringType(), True),
        StructField("c2", StringType(), True),
    ]
    data = [["a", "aa"], ["b", "bb"]]
    return spark.createDataFrame(data, schema=StructType(fields)).persist()


@pytest.mark.parametrize(
    "data, kwargs",
    [
        (
            [
                ["c1", "cc1"],
                ["d1", "dd1"],
            ],
            {"schema": "c3: string, c4: string"},
        ),
        (
            {
                "c5": ["c2", "cc2"],
                "c6": ["d2", "dd2"],
            },
            None,
        ),
        (
            [
                {"c5": "c2", "c6": "d2"},
                {"c5": "d2", "c6": "dd2"},
            ],
            None,
        ),
    ],
)
@pytest.mark.parametrize("broadcast", [True, False])
def test_create_dataframe(spark, df_input, data, kwargs, broadcast):
    """Test 'CreateDataFrame' transformer."""
    t = CreateDataFrame(data=data, broadcast=broadcast, kwargs=kwargs)
    ns.clear()
    try:
        df_out = t.transform(df_input)
        kwargs = kwargs or {}

        df_exp = None
        if isinstance(data, dict):
            values = list(data.values())[0]
            if isinstance(values, (list, tuple)):
                df_exp = spark.createDataFrame(pd.DataFrame(data))

        if df_exp is None:
            df_exp = spark.createDataFrame(data, **kwargs)

        if broadcast:
            assert is_broadcast(df_out)
        assert_df_equality(df_out, df_exp, ignore_row_order=True)
    finally:
        ns.clear()
