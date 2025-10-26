"""Unit-test for CountWhereBool."""

import pytest
from chispa import assert_df_equality
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, StructField, StructType

from nlsn.nebula.spark_transformers import CountWhereBool
from nlsn.nebula.storage import nebula_storage as ns

_COLUMNS = ["c1", "c2", "c3", "c4", "c5"]
_DATA = [
    [True, None, True, False, True],
    [False, None, True, None, None],
    [None, None, True, False, True],
]

_EXP_TRUE = {c: 0 for c in _COLUMNS}
_EXP_FALSE = {c: 0 for c in _COLUMNS}

_N_ROWS = len(_DATA)

for i, c in enumerate(_COLUMNS):
    for j in range(_N_ROWS):
        _EXP_TRUE[c] += _DATA[j][i] is True
        _EXP_FALSE[c] += _DATA[j][i] is False


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    fields = [StructField(i, BooleanType(), True) for i in _COLUMNS]
    df = spark.createDataFrame(_DATA, schema=StructType(fields))
    df = df.withColumn("other", F.lit("string"))
    return df.persist()


@pytest.mark.parametrize("count_type", [True, False])
def test_count_where_bool(df_input, count_type: bool):
    """Test CountWhereTrue transformer."""
    ns.clear()

    t = CountWhereBool(columns=_COLUMNS, count_type=count_type, store_key="count_bool")

    df_chk = t.transform(df_input)
    # Dataframe should be untouched
    assert_df_equality(df_chk, df_input, ignore_row_order=True)

    chk = ns.get("count_bool")

    if count_type:
        assert chk == _EXP_TRUE
    else:
        assert chk == _EXP_FALSE

    ns.clear()
