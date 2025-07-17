"""Unit-test for _Partitions."""

import pytest

from nlsn.nebula.spark_transformers.partitions import _Partitions
from nlsn.nebula.spark_util import get_default_spark_partitions

_N_ROWS: int = 1_000


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    data = [[f"{i}"] for i in range(_N_ROWS)]
    return spark.createDataFrame(data, ["c1"]).persist()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_partitions": "x"},
        {"rows_per_partition": "x"},
        {"num_partitions": [1, "x"]},
        {"num_partitions": 0},
        {"rows_per_partition": 0},
        {"num_partitions": 1, "rows_per_partition": 1},
        {"num_partitions": 1, "to_default": True},
        {"rows_per_partition": 1, "to_default": True},
    ],
)
def test__partitions_errors(kwargs):
    """Test _Partitions with wrong parameters."""
    with pytest.raises(AssertionError):
        _Partitions(**kwargs)


@pytest.mark.parametrize(
    "kwargs, exp",
    [
        ({"num_partitions": 10}, 10),
        ({"to_default": True}, None),
        ({"rows_per_partition": 50}, _N_ROWS // 50),
    ],
)
def test_get_num_partitions(df_input, kwargs, exp):
    """Test '_get_requested_partitions' method."""
    t = _Partitions(**kwargs)
    chk = t._get_requested_partitions(df_input, "unit-test")
    if exp is None:
        exp = get_default_spark_partitions(df_input)
    assert chk == exp
