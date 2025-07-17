"""Unit-test for AssertCount."""
import random

import pytest
from chispa import assert_df_equality
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from nlsn.nebula.spark_transformers import AssertCount
from nlsn.nebula.storage import nebula_storage as ns

_DATA = [
    (1, "a"),
    (2, "b"),
    (3, "c"),
    (4, "d"),
    (5, "e"),
]

_N_INPUT: int = len(_DATA)


_POSSIBLE_COMPARISONS = [
    ({"number": _N_INPUT + 1, "comparison": "ne"}, -1, False),
    ({"number": _N_INPUT, "comparison": "ge"}, -1, False),
    ({"number": _N_INPUT - 1, "comparison": "gt"}, -1, False),
    ({"number": _N_INPUT, "comparison": "gt"}, -1, True),
    ({"number": _N_INPUT, "comparison": "le"}, -1, False),
    ({"number": _N_INPUT + 1, "comparison": "lt"}, -1, False),
    ({"number": _N_INPUT, "comparison": "lt"}, -1, True),
]


@pytest.fixture(scope="module", name="df_input")
def _get_input_df(spark):
    fields = [
        StructField("col1", IntegerType(), True),
        StructField("col2", StringType(), True),
    ]

    return spark.createDataFrame(_DATA, schema=StructType(fields)).persist()


@pytest.mark.parametrize(
    "kwargs, store_value, error",
    [
        ({"number": _N_INPUT}, -1, False),
        ({"store_key": "key"}, _N_INPUT, False),
        ({"number": _N_INPUT, "comparison": "ne"}, -1, True),
        random.choice(_POSSIBLE_COMPARISONS),
    ],
)
def test_assert_count(df_input, kwargs, store_value: int, error: bool):
    """Test AssertCount transformer."""
    ns.clear()

    if "store_key" in kwargs:
        ns.set(kwargs["store_key"], store_value)
    t = AssertCount(**kwargs)
    if error:
        with pytest.raises(AssertionError):
            t.transform(df_input)
    else:
        df_chk = t.transform(df_input)
        assert_df_equality(
            df_chk, df_input, ignore_row_order=True, ignore_nullable=True
        )
