"""Unit-Test 'df_types' module."""

import pandas as pd
import pytest
from pyspark.sql.types import StructType

from nlsn.nebula.df_types import get_dataframe_type


@pytest.mark.parametrize(
    "o, exp",
    [
        (pd.DataFrame(), "pandas"),
        ("spark_placeholder", "spark"),
        ([], None),
    ],
)
def test_get_dataframe_type(spark, o, exp):
    """Unit-Test 'get_dataframe_type' function."""
    if exp is None:
        with pytest.raises(TypeError):
            get_dataframe_type(o)
        return

    if isinstance(o, str) and (o == "spark_placeholder"):
        o = spark.createDataFrame([], schema=StructType([]))

    chk = get_dataframe_type(o)
    assert chk == exp
