"""Unit-test for EmptyStringToNull."""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from nlsn.nebula.spark_transformers import EmptyStringToNull


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark: SparkSession):
    fields = [
        StructField("idx", IntegerType(), True),
        StructField("c1", StringType(), True),
        StructField("c2", StringType(), True),
    ]

    data = [
        [0, "a", "b"],
        [1, "a  ", "  b"],
        [2, "  a  ", "  b  "],
        [3, "", ""],
        [4, "   ", "   "],
        [5, None, None],
    ]

    return spark.createDataFrame(data, schema=StructType(fields)).cache()


def _get_expected(s, do_trim):
    if s is None:
        return None
    v = s
    if do_trim:
        v = v.strip()
    return None if v == "" else s


@pytest.mark.parametrize("trim", [True, False])
def test_empty_string_to_null(df_input, trim: bool):
    """Test EmptyStringToNull transformer."""
    columns = ["c1", "c2"]
    t = EmptyStringToNull(columns=columns, trim=trim)
    df_out = t.transform(df_input)

    data = df_out.select("idx", *columns).toPandas().set_index("idx")

    df_input_pd = df_input.toPandas()
    dict_orig_data = df_input_pd.set_index("idx").to_dict(orient="index")

    for idx, row in data.iterrows():
        # idx 0, 1, 2, 3, 4, 5
        row_expected = dict_orig_data[idx]
        for c in columns:
            chk = row[c]
            original_value = row_expected[c]
            exp = _get_expected(original_value, trim)

            msg = f'do_trim={trim} row {idx} expected "{exp}" found "{chk}"'
            assert exp == chk, msg
