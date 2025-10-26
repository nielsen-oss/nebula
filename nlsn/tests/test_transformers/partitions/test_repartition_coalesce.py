"""Unit-test for CoalescePartitions end Repartition."""

import pytest
from chispa import assert_df_equality
from pyspark.sql.types import StringType, StructField, StructType

from nlsn.nebula.spark_transformers.partitions import CoalescePartitions, Repartition
from nlsn.nebula.spark_util import get_default_spark_partitions

_N_ROWS: int = 60

_PARAMS = [
    ({"num_partitions": 5}, 5),
    ({"to_default": True}, None),
    ({"rows_per_partition": 20}, _N_ROWS // 20),
]


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    fields = [
        StructField("a", StringType(), True),
        StructField("b", StringType(), True),
        StructField("c", StringType(), True),
    ]

    data = [[f"a{i}", f"b{i}", f"c{i}"] for i in range(_N_ROWS)]
    schema = StructType(fields)
    # Number of partitions after a coalesce() cannot be > than current number.
    # Repartition to high number of the input dataframe.
    return spark.createDataFrame(data, schema=schema).repartition(50).persist()


@pytest.mark.parametrize("columns", ["a", ["a", "b"]])
@pytest.mark.parametrize("kwargs, exp_partitions", _PARAMS)
def test_repartition_correct(df_input, kwargs, exp_partitions, columns):
    """Test Repartition transformer."""
    t = Repartition(columns=columns, **kwargs)
    if exp_partitions is None:
        exp_partitions = get_default_spark_partitions(df_input)

    df_chk = t.transform(df_input)
    # trigger
    df_chk.count()  # trigger
    chk_partitions: int = df_chk.rdd.getNumPartitions()
    assert chk_partitions == exp_partitions

    # Assert the input dataframe is not modified
    assert_df_equality(df_chk, df_input, ignore_row_order=True)


@pytest.mark.parametrize("columns", [1, ["a", "a"], ["a", 1]])
def test_repartition_wrong_column_types(columns):
    """Test Repartition with wrong columns types."""
    with pytest.raises(AssertionError):
        Repartition(num_partitions=10, columns=columns)


def test_repartition_wrong_columns(df_input):
    """Test Repartition with wrong columns."""
    t = Repartition(num_partitions=10, columns="wrong")
    with pytest.raises(AssertionError):
        t.transform(df_input)


@pytest.mark.parametrize("kwargs, exp_partitions", _PARAMS)
def test_coalesce_partitions(df_input, kwargs, exp_partitions):
    """Test CoalescePartitions transformer."""
    t = CoalescePartitions(**kwargs)
    if exp_partitions is None:
        exp_partitions = get_default_spark_partitions(df_input)

    df_chk = t.transform(df_input)
    # trigger
    df_chk.count()  # trigger
    chk_partitions: int = df_chk.rdd.getNumPartitions()
    assert chk_partitions == exp_partitions

    # Assert the input dataframe is not modified
    assert_df_equality(df_chk, df_input, ignore_row_order=True)
