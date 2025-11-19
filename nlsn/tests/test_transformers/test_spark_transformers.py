"""Unit-tests for spark-partitions transformers."""

import pytest
from chispa import assert_df_equality
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from nlsn.nebula.spark_util import get_default_spark_partitions
from nlsn.nebula.transformers.spark_transformers import *
from nlsn.nebula.transformers.spark_transformers import _Partitions, CpuInfo

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


@pytest.mark.parametrize("kwargs, exp_partitions", _PARAMS)
def test_coalesce_partitions(df_input, kwargs, exp_partitions):
    """Test CoalescePartitions transformer."""
    t = CoalescePartitions(**kwargs)
    if exp_partitions is None:
        exp_partitions = get_default_spark_partitions(df_input)

    df_chk = t.transform(df_input)
    df_chk.count()  # trigger
    chk_partitions: int = df_chk.rdd.getNumPartitions()
    assert chk_partitions == exp_partitions

    # Assert the input dataframe is not modified
    assert_df_equality(df_chk, df_input, ignore_row_order=True)


def test_log_data_skew(spark):
    """Test LogDataSkew transformer."""
    schema = StructType([StructField("c1", IntegerType(), True)])
    data = [[i] for i in range(100)]
    df = spark.createDataFrame(data, schema=schema)

    t = LogDataSkew()
    t.transform(df)


class TestPartitions:
    """Test private class '_Partitions'."""

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
    def test_invalid(self, kwargs):
        """Test _Partitions with wrong parameters."""
        with pytest.raises(AssertionError):
            _Partitions(**kwargs)

    @pytest.fixture(scope="class", name="df_input")
    def _get_df_input(self, spark):
        data = [[f"{i}"] for i in range(1000)]
        return spark.createDataFrame(data, ["c1"])

    @pytest.mark.parametrize(
        "kwargs, exp",
        [
            ({"num_partitions": 10}, 10),
            ({"to_default": True}, None),
            ({"rows_per_partition": 50}, 1000 // 50),
        ],
    )
    def test_get_num_partitions(self, df_input, kwargs, exp):
        """Test '_get_requested_partitions' method."""
        t = _Partitions(**kwargs)
        chk = t._get_requested_partitions(df_input, "unit-test")
        if exp is None:
            exp = get_default_spark_partitions(df_input)
        assert chk == exp


class TestRepartition:
    """Test Repartition transformer."""

    @pytest.mark.parametrize("columns", ["a", ["a", "b"]])
    @pytest.mark.parametrize("kwargs, exp_partitions", _PARAMS)
    def test_valid(self, df_input, kwargs, exp_partitions, columns):
        t = Repartition(columns=columns, **kwargs)
        if exp_partitions is None:
            exp_partitions = get_default_spark_partitions(df_input)

        df_chk = t.transform(df_input)
        df_chk.count()  # trigger
        chk_partitions: int = df_chk.rdd.getNumPartitions()
        assert chk_partitions == exp_partitions

        # Assert the input dataframe is not modified
        assert_df_equality(df_chk, df_input, ignore_row_order=True)

    @pytest.mark.parametrize("columns", [1, ["a", "a"], ["a", 1]])
    def test_invalid_column_types(self, columns):
        """Invalid columns types."""
        with pytest.raises(AssertionError):
            Repartition(num_partitions=10, columns=columns)

    def test_invalid_columns(self, df_input):
        """Invalid columns."""
        t = Repartition(num_partitions=10, columns="wrong")
        with pytest.raises(AssertionError):
            t.transform(df_input)


def test_cpu_info(spark):
    """Test CpuInfo."""
    schema = StructType([StructField("c1", IntegerType(), True)])
    df = spark.createDataFrame([[1]], schema=schema)

    t = CpuInfo(n_partitions=10)
    t.transform(df)