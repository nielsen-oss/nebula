"""Unit-tests for spark transformers."""

import pytest
from chispa import assert_df_equality
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    MapType,
    StringType,
    StructField,
    StructType
)
from pyspark.sql.utils import AnalysisException

from nlsn.nebula.spark_util import (
    drop_duplicates_no_randomness,
    get_default_spark_partitions,
)
from nlsn.nebula.transformers.spark_transformers import *
from nlsn.nebula.transformers.spark_transformers import _Partitions


def test_cpu_info(spark):
    """Test CpuInfo."""
    schema = StructType([StructField("c1", IntegerType(), True)])
    df = spark.createDataFrame([[1]], schema=schema)

    t = CpuInfo(n_partitions=10)
    t.transform(df)


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
            ({}, None),
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


class TestCoalesceRepartition:
    """Test Coalesce and Repartition transformer."""

    _N_ROWS: int = 60

    _PARAMS = [
        ({"num_partitions": 5}, 5),
        ({}, None),
        ({"rows_per_partition": 20}, _N_ROWS // 20),
    ]

    @pytest.fixture(scope="module", name="df_input")
    def _get_df_input(self, spark):
        fields = [
            StructField("a", StringType(), True),
            StructField("b", StringType(), True),
            StructField("c", StringType(), True),
        ]

        data = [[f"a{i}", f"b{i}", f"c{i}"] for i in range(self._N_ROWS)]
        schema = StructType(fields)
        # Number of partitions after a coalesce() cannot be > than current number.
        # Repartition to high number of the input dataframe.
        return spark.createDataFrame(data, schema=schema).repartition(50).persist()

    @pytest.mark.parametrize("kwargs, exp_partitions", _PARAMS)
    def test_coalesce_partitions(self, df_input, kwargs, exp_partitions):
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

    @pytest.mark.parametrize("columns", ["a", ["a", "b"]])
    @pytest.mark.parametrize("kwargs, exp_partitions", _PARAMS)
    def test_repartition_valid(self, df_input, kwargs, exp_partitions, columns):
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
    def test_repartition_invalid_column_types(self, columns):
        """Invalid columns types."""
        with pytest.raises(AssertionError):
            Repartition(num_partitions=10, columns=columns)

    def test_repartition_invalid_columns(self, df_input):
        """Invalid columns."""
        t = Repartition(num_partitions=10, columns="wrong")
        with pytest.raises(AssertionError):
            t.transform(df_input)


class TestSparkColumnMethod:
    @staticmethod
    @pytest.fixture(scope="class", name="df_input")
    def _get_df_input(spark):
        fields = [StructField("name", StringType(), True)]

        data = [
            ["house"],
            ["cat"],
            ["secondary"],
            [None],
        ]
        return spark.createDataFrame(data, StructType(fields)).persist()

    def test_invalid_method(self, df_input):
        """Test SparkColumnMethod with a wrong method name."""
        with pytest.raises(ValueError):
            t = SparkColumnMethod(input_column="name", method="invalid")
            t.transform(df_input)

    def test_invalid_column(self, df_input):
        """Test SparkColumnMethod with a wrong column name."""
        t = SparkColumnMethod(input_column="invalid", method="isNull")
        with pytest.raises(AnalysisException):
            t.transform(df_input)

    def test_valid(self, df_input):
        """Test SparkColumnMethod."""
        t = SparkColumnMethod(
            input_column="name", method="contains", output_column="result", args=["se"]
        )
        df_chk = t.transform(df_input)

        df_exp = df_input.withColumn("result", F.col("name").contains("se"))
        assert_df_equality(df_chk, df_exp, ignore_row_order=True)

    def test_no_args(self, df_input):
        """Test ColumnMethod w/o any arguments and overriding the input column."""
        t = SparkColumnMethod(input_column="name", method="isNull")
        df_chk = t.transform(df_input)

        df_exp = df_input.withColumn("name", F.col("name").isNull())
        assert_df_equality(df_chk, df_exp, ignore_row_order=True)


class TestSparkDropDuplicates:
    @staticmethod
    @pytest.fixture(scope="class", name="df_input")
    def _get_df_input(spark):
        fields = [
            StructField("c1", StringType(), True),
            StructField("c2", StringType(), True),
            StructField("c3", StringType(), True),
        ]

        data = [
            ["a", "d", "h"],
            ["a", "d", None],
            ["a", "d", "h"],
            ["a", "d", "i"],
            ["a", "d", "h"],
            ["a", "d", None],
            ["b", "b", "l"],
            ["b", "b", None],
            ["b", "b", "m"],
            ["b", "b", "m"],
            ["b", "b", "m"],
            ["c", "e", None],
            ["c", "e", None],
            ["c", "e", None],
            ["c", "e", None],
            ["c", "g", None],
            ["c", "g", "n"],
        ]
        return spark.createDataFrame(data, schema=StructType(fields)).persist()

    def test_drop_duplicates_no_subset(self, df_input):
        """Test DropDuplicates transformer w/o subset columns."""
        t = SparkDropDuplicates()
        df_chk = t.transform(df_input)
        df_exp = df_input.drop_duplicates()
        assert_df_equality(df_chk, df_exp, ignore_row_order=True)

    def _test_complex_types(self, df_input):
        mapping = F.create_map(F.lit("a"), F.lit("1"), F.lit("b"), F.lit("2"))
        array = F.array(F.lit("x"), F.lit("y"))

        df_complex = df_input.withColumn("mapping", mapping).withColumn("array", array)

        subset = ["c1", "c2"]
        chk = drop_duplicates_no_randomness(df_complex, subset)

        # Check whether MapType and ArrayType raise any error
        chk.count()

    def test_drop_duplicates_subset(self, df_input):
        """Test the transformer w/ subset columns."""
        # just to check if MapType and ArrayType raise any error
        self._test_complex_types(df_input)

        # check if the same df order in the opposite manner gives the same result
        df_desc = df_input.sort([F.col(i).desc() for i in df_input.columns])
        df_asc = df_input.sort([F.col(i).asc() for i in df_input.columns])

        subset = ["c1", "c2"]
        t = SparkDropDuplicates(columns=subset)
        df_chk_desc = t.transform(df_desc)
        df_chk_asc = t.transform(df_asc)

        assert_df_equality(df_chk_desc, df_chk_asc, ignore_row_order=True)


class TestSparkExplode:
    _DATA = [
        ("row_1", [0, 1, None, 2], {"a": 0, "b": None, "c": 1}),
        ("row_2", [0, None, None], {"a": 0, "b": None}),
        ("row_3", [0], {"c": 3}),
        ("row_4", [None], {"a": 0, "b": None}),
        ("row_5", [], {}),
        ("row_6", None, {"a": 0, "c": 6}),
    ]

    _COL_STRING: str = "row_name"
    _COL_ARRAY: str = "arrays"
    _COL_MAP: str = "mapping"
    _COL_OUTPUT_ARRAY: str = "output"
    _COL_OUTPUT_MAP: list[str] = ["key", "value"]

    @pytest.fixture(scope="class", name="df_input")
    def _get_df_input(self, spark):
        fields = [
            StructField(self._COL_STRING, StringType()),
            StructField(self._COL_ARRAY, ArrayType(IntegerType())),
            StructField(self._COL_MAP, MapType(StringType(), IntegerType())),
        ]
        schema: StructType = StructType(fields)
        return spark.createDataFrame(self._DATA, schema).persist()

    def _check_columns_array_type(self, cols_chk, kwg):
        """Check output columns."""
        # If "output_col" is not provided, the output is stored in the input
        # column (_COL_ARRAY).
        out_col: str = kwg.get("output_cols", self._COL_ARRAY)
        drop_after: bool = kwg["drop_after"]

        # Create the list of expected columns.
        cols_exp = [self._COL_STRING, self._COL_ARRAY, self._COL_MAP]

        # If output column != input column, add it to the expected columns.
        if out_col != self._COL_ARRAY:
            cols_exp.append(self._COL_OUTPUT_ARRAY)

        # Remove the input column if "drop_after" = True and
        # output column != input column.
        if drop_after and (self._COL_ARRAY != out_col):
            cols_exp.remove(self._COL_ARRAY)
        assert cols_chk == cols_exp, kwg

    @pytest.mark.parametrize("output_cols", [["c1"], ["c1", 1], ["c1", "c2", "c3"]])
    def test_invalid_output_columns(self, output_cols):
        """Test Explode transformer with wrong 'output_cols'."""
        with pytest.raises(AssertionError):
            SparkExplode(input_col=self._COL_MAP, output_cols=output_cols)

    @pytest.mark.parametrize("output_cols", [None, "wrong"])
    def test_map_invalid_output_columns(self, df_input, output_cols):
        """Test the transformer with wrong 'output_cols' for MapType."""
        t = SparkExplode(input_col=self._COL_MAP, output_cols=output_cols)
        with pytest.raises(AssertionError):
            t.transform(df_input)

    def test_invalid_input(self, df_input):
        """Test the transformer with <StringType> as input column."""
        t = SparkExplode(input_col=self._COL_STRING, output_cols="x")
        with pytest.raises(AssertionError):
            t.transform(df_input)

    def _get_expected_len(self, idx_data):
        # List of iterables.
        data_col = [i[idx_data] for i in self._DATA]

        # Expected len when "outer" = False, thus null values and empty
        # arrays are discarded.
        base_len = sum(len(i) for i in data_col if i)

        # Number of rows where data is null or the array is empty.
        # This number sum up with base_len when "outer" = True
        null_len = len([True for i in data_col if not i])

        return base_len, null_len

    def test_array(self, df_input):
        """Test the transformer with <ArrayType> as input column."""
        inputs = [
            {"output_cols": self._COL_OUTPUT_ARRAY, "outer": True, "drop_after": True},
            {"output_cols": self._COL_OUTPUT_ARRAY, "outer": True, "drop_after": False},
            {"output_cols": self._COL_OUTPUT_ARRAY, "outer": False, "drop_after": True},
            {"output_cols": self._COL_OUTPUT_ARRAY, "outer": False, "drop_after": False},
            {"outer": True, "drop_after": True},
            {"outer": True, "drop_after": False},
            {"outer": False, "drop_after": True},
            {"outer": False, "drop_after": False},
        ]
        # 1 -> array, 2 -> map
        base_len, null_len = self._get_expected_len(1)

        for kwg in inputs:
            t = SparkExplode(input_col=self._COL_ARRAY, **kwg)
            df_out = t.transform(df_input)

            # Check columns.
            self._check_columns_array_type(df_out.columns, kwg)

            len_chk = df_out.count()
            len_exp = base_len
            if kwg["outer"]:
                len_exp += null_len
            assert len_chk == len_exp, kwg

    def test_mapping(self, df_input):
        """Test the transformer with <MapType> as input column."""
        inputs = [
            {"outer": True, "drop_after": True},
            {"outer": True, "drop_after": False},
            {"outer": False, "drop_after": True},
            {"outer": False, "drop_after": False},
        ]
        # 1 -> array, 2 -> map
        base_len, null_len = self._get_expected_len(2)

        cols_exp = set(df_input.columns).union(set(self._COL_OUTPUT_MAP))
        cols_exp.copy()

        for kwg in inputs:
            t = SparkExplode(
                input_col=self._COL_MAP, output_cols=self._COL_OUTPUT_MAP, **kwg
            )
            df_out = t.transform(df_input)

            set_cols_chk = set(df_out.columns)
            # Check columns.
            if kwg["drop_after"]:
                assert set_cols_chk == (cols_exp - {self._COL_MAP})
            else:
                assert set_cols_chk == cols_exp

            len_chk = df_out.count()
            len_exp = base_len
            if kwg["outer"]:
                len_exp += null_len
            assert len_chk == len_exp, kwg


class TestSparkSqlFunction:
    @staticmethod
    @pytest.fixture(scope="class", name="df_input")
    def _get_df_input(spark):
        fields = [StructField("data", ArrayType(IntegerType()))]
        data = [([2, 1, None, 3],), ([1],), ([],), (None,)]
        return spark.createDataFrame(data, StructType(fields)).persist()

    @pytest.mark.parametrize("asc", [True, False])
    def test_va(self, df_input, asc: bool):
        """Test SqlFunction."""
        t = SparkSqlFunction(
            column="result", function="sort_array", args=["data"], kwargs={"asc": asc}
        )
        df_chk = t.transform(df_input)

        df_exp = df_input.withColumn("result", F.sort_array("data", asc=asc))
        assert_df_equality(df_chk, df_exp, ignore_row_order=True)

    def test_sql_function_no_args(self, df_input):
        """Test SqlFunction w/o any arguments."""
        t = SparkSqlFunction(column="result", function="rand")
        df_chk = t.transform(df_input)

        n_null: int = df_chk.filter(F.col("result").isNull()).count()
        assert n_null == 0
