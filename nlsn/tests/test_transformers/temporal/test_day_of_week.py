"""Unit-test for DayOfWeek."""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from nlsn.nebula.spark_transformers import DayOfWeek

_data = {
    "2022-10-10": {"day_num": 2, "day_name": "Mon"},
    "2022-10-11": {"day_num": 3, "day_name": "Tue"},
    "2022-10-12": {"day_num": 4, "day_name": "Wed"},
}


@pytest.fixture(scope="module", name="df_input")
def _get_input_data(spark):
    fields = [StructField("col_orig", StringType(), True)]

    dates = [[i] for i in _data]

    # Add some null values or invalid strings.
    data = dates + [[None], [""], ["   "]]

    ret = spark.createDataFrame(data, schema=StructType(fields))
    return ret.withColumn("col_1", F.col("col_orig")).persist()


@pytest.mark.parametrize("output_col, as_string", [("output_col", True), (None, False)])
def test_day_of_week(df_input, as_string, output_col):
    """Unit-test for DayOfWeek transformer."""
    t = DayOfWeek(input_col="col_1", as_string=as_string, output_col=output_col)

    df_chk = t.transform(df_input)
    chk_col = output_col if output_col else "col_1"
    results = df_chk.select("col_orig", chk_col).rdd.map(lambda x: x[:]).collect()

    for date_str, chk in results:
        nd = _data.get(date_str)
        if nd is None:  # Null value or not valid format date
            assert chk is None
        else:
            if as_string:
                exp = nd.get("day_name")
            else:
                exp = nd.get("day_num")
            assert chk == exp
