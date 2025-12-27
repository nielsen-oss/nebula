"""Unit-tests for spark transformers."""

from string import ascii_lowercase

import numpy as np
import pandas as pd
import pytest
from chispa import assert_df_equality
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, MapType, Row
from pyspark.sql.types import IntegerType, StringType, StructField, StructType
from pyspark.sql.utils import AnalysisException

from nebula.auxiliaries import is_list_uniform
from nebula.spark_util import (
    drop_duplicates_no_randomness,
    get_default_spark_partitions,
)
from nebula.transformers.spark_transformers import *
from nebula.transformers.spark_transformers import _Partitions, _Window
from nebula.transformers.spark_transformers import validate_window_frame_boundaries


def test_cache(spark):
    schema = StructType([StructField("c1", IntegerType(), True)])
    df = spark.createDataFrame([[1]], schema=schema)
    t = Cache()
    df_chk = t.transform(df)
    assert df_chk.is_cached


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
    t = LogDataSkew(persist=True)
    t.transform(df)


class TestAggregateOverWindow:
    """Test AggregateOverWindow transformer."""

    @staticmethod
    @pytest.mark.parametrize("order_cols, partition_cols", [("x", None), (None, "x")])
    def test_invalid_alias_overriding(order_cols, partition_cols):
        """Test with overriding aliases."""
        with pytest.raises(AssertionError):
            AggregateOverWindow(
                order_cols=order_cols,
                partition_cols=partition_cols,
                aggregations=[{"agg": "sum", "col": "a", "alias": "x"}],
            )

    @staticmethod
    @pytest.fixture(scope="class", name="df_input")
    def _get_df_input(spark):
        n_rows = 200

        li_ids = np.random.randint(0, 50, n_rows).tolist()
        li_cat = np.random.choice(list(ascii_lowercase), n_rows).tolist()
        li_plt = np.random.choice(list("ABC"), n_rows).tolist()

        data = list(zip(li_ids, li_cat, li_plt))

        fields = [
            StructField("id", IntegerType(), True),
            StructField("category", StringType(), True),
            StructField("platform", StringType(), True),
        ]
        return spark.createDataFrame(data, schema=StructType(fields)).persist()

    @pytest.mark.parametrize("ascending", [True, False])
    @pytest.mark.parametrize(
        "order_cols, rows_between, range_between",
        [
            [None, None, None],
            ["id", None, None],
            [None, (0, 2), None],
            [None, ("start", "end"), None],
            ["id", (0, 2), None],
            ["id", ("start", "end"), None],
            ["id", None, (0, 1)],
            ["id", None, ("start", "end")],
        ],
    )
    def test(self, df_input, ascending, order_cols, rows_between, range_between):
        """Test with multiple cases."""
        aggregations = [
            {"agg": "min", "col": "id", "alias": "min_id"},
            {"agg": "sum", "col": "id", "alias": "sum_id"},
        ]
        t = AggregateOverWindow(
            partition_cols="category",
            order_cols=order_cols,
            aggregations=aggregations,
            ascending=ascending,
            rows_between=rows_between,
            range_between=range_between,
        )
        df_chk = t.transform(df_input)

        window = Window.partitionBy("category")
        if order_cols is not None:
            col_id = F.col("id")
            if not ascending:
                col_id = col_id.desc()
            window = window.orderBy(col_id)
        if rows_between is not None:
            start, end = validate_window_frame_boundaries(*rows_between)
            window = window.rowsBetween(start, end)
        if range_between is not None:
            start, end = validate_window_frame_boundaries(*range_between)
            window = window.rangeBetween(start, end)

        df_exp = df_input.withColumn("min_id", F.min("id").over(window)).withColumn(
            "sum_id", F.sum("id").over(window)
        )

        assert_df_equality(
            df_chk, df_exp, ignore_row_order=True, ignore_column_order=True
        )

    @pytest.mark.parametrize("ascending", [[True, False], [False, True]])
    def test_ascending(self, df_input, ascending):
        """Test with ascending as <list<bool>>."""
        order_cols = ["platform", "id"]

        t = AggregateOverWindow(
            partition_cols="category",
            order_cols=order_cols,
            ascending=ascending,
            aggregations=[{"agg": "first", "col": "id", "alias": "first_value"}],
        )
        df_chk = t.transform(df_input)

        # Set the order
        li_asc = ascending if isinstance(ascending, list) else [ascending] * 2
        orders = [
            F.col(i).asc() if j else F.col(i).desc() for i, j in zip(order_cols, li_asc)
        ]
        win = Window.partitionBy("category").orderBy(orders)
        df_exp = df_input.withColumn("first_value", F.first("id").over(win))

        assert_df_equality(
            df_chk, df_exp, ignore_row_order=True, ignore_column_order=True
        )

    def test_single_aggregation(self, df_input):
        """Test w/o partitioning."""
        t = AggregateOverWindow(
            aggregations=[{"agg": "sum", "col": "id", "alias": "sum_id"}],
            rows_between=("start", "end"),
        )
        df_chk = t.transform(df_input)

        win = Window.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

        df_exp = df_input.withColumn("sum_id", F.sum("id").over(win))
        assert_df_equality(
            df_chk, df_exp, ignore_row_order=True, ignore_column_order=True
        )

    def test_override(self, df_input):
        """Test with 'id' column overridden."""
        t = AggregateOverWindow(
            partition_cols="category",
            aggregations={"agg": "min", "col": "id", "alias": "id"},
        )
        df_chk = t.transform(df_input)

        win = Window.partitionBy("category")

        df_exp = df_input.withColumn("id", F.min("id").over(win).cast("int"))
        assert_df_equality(
            df_chk, df_exp, ignore_row_order=True, ignore_column_order=True
        )


class TestColumnsToMap:
    _data = [(0, 1, 11), (1, 2, None), (2, None, 3)]

    @pytest.fixture(scope="class", name="df_input")
    def _get_df_input(self, spark):
        fields = [
            StructField("idx", IntegerType()),
            StructField("c1", IntegerType()),
            StructField("c2", IntegerType()),
        ]
        schema = StructType(fields)
        return spark.createDataFrame(self._data, schema).persist()

    @staticmethod
    def _check(v_exp, v_chk, cast_values):
        if v_exp is None:
            assert v_chk is None
        else:
            if cast_values == "string":
                assert v_chk == str(v_exp)
            else:
                assert v_chk == v_exp

    @pytest.mark.parametrize(
        "cast_values, drop",
        [
            [None, False],
            ["string", True],
        ],
    )
    def test(self, df_input, cast_values: str, drop: bool):
        """Test ColumnsToMap."""
        t = ColumnsToMap(
            columns=["c1", "c2"],
            output_column="result",
            cast_values=cast_values,
            drop_input_columns=drop,
        )
        df_chk = t.transform(df_input).persist()
        n = df_chk.count()
        assert len(self._data) == n

        set_cols = set(df_chk.columns)

        if drop:
            cols = ["idx", "result"]
            assert set_cols == set(cols)
        else:
            cols = ["idx", "c1", "c2", "result"]
            assert set_cols == set(cols)

        collected: list[Row] = df_chk.select(cols).sort("idx").collect()

        row: Row
        for row in collected:
            d_row = row.asDict()

            idx = d_row["idx"]

            v1_exp = self._data[idx][1]
            v2_exp = self._data[idx][2]

            result = d_row["result"]
            assert set(result.keys()) == {"c1", "c2"}

            v1_chk = result["c1"]
            v2_chk = result["c2"]

            self._check(v1_exp, v1_chk, cast_values)
            self._check(v2_exp, v2_chk, cast_values)


class TestCoalesceRepartition:
    """Test Coalesce and Repartition transformer."""

    _N_ROWS: int = 60

    _PARAMS = [
        ({"num_partitions": 5}, 5),
        ({}, None),
        ({"rows_per_partition": 20}, _N_ROWS // 20),
    ]

    @pytest.fixture(scope="class", name="df_input")
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


class TestLagOverWindow:
    """Test LagOverWindow transformer."""

    @staticmethod
    @pytest.fixture(scope="class", name="df_input")
    def _get_df_input(spark):
        data = [
            ["group_1", 2, 10000],
            ["group_1", 1, 2000],
            ["group_1", 3, 50],
            ["group_1", 4, 40],
            ["group_2", 7, 16000],
            ["group_2", 5, 12000],
            ["group_2", 2, 11000],
            ["group_3", 12, 900],
            ["group_3", 13, 600],
        ]
        schema_str = "group: string, index: int, value: int"
        return spark.createDataFrame(data, schema=schema_str).persist()

    @pytest.mark.parametrize("lag", [2, -1, -2])
    def test(self, df_input, lag):
        partition_cols = ["group"]
        order_cols = ["index"]
        lag_col = "value"
        output_col = "value_lag"

        t = LagOverWindow(
            partition_cols=partition_cols,
            order_cols=order_cols,
            lag_col=lag_col,
            lag=lag,
            output_col=output_col,
        )

        df_chk = t.transform(df_input).toPandas()

        # Assert that the number of nulls in the lagged column is equal to the number of groups
        n_nulls = df_chk[output_col].isna().sum()
        expected_nulls = int(df_chk[partition_cols].nunique()) * abs(lag)
        assert n_nulls == expected_nulls

        # Perform the same operation in Pandas
        df_exp = df_input.toPandas().sort_values(by=order_cols)
        df_exp[output_col] = df_exp.groupby(partition_cols)[lag_col].shift(lag)

        # Assert equality of dataframe
        pd.testing.assert_frame_equal(
            df_exp.sort_values(df_exp.columns.tolist()).reset_index(drop=True),
            df_chk.sort_values(df_chk.columns.tolist()).reset_index(drop=True),
        )


class TestMapToColumns:
    _data = [
        (1, {"a": 1, "b": 2}),
        (2, {"x": None, "y": 20}),
        (3, None),
    ]

    @pytest.fixture(scope="class", name="df_input")
    def _get_df_input(self, spark):
        fields = [
            StructField("id", IntegerType()),
            StructField("map_col", MapType(StringType(), IntegerType())),
        ]
        schema = StructType(fields)
        return spark.createDataFrame(self._data, schema).persist()

    @pytest.mark.parametrize(
        "output_columns",
        [
            ["a", "b", "wrong"],
            [("a", "col_a"), ("c", "col_c")],
            {"a": "col_a", "c": "col_c", "x": "col_x"},
        ],
    )
    def test_valid(self, df_input, output_columns):
        """Test MapToColumns w/ and w/o 'output_column'."""
        t = MapToColumns(input_column="map_col", output_columns=output_columns)
        df_out = t.transform(df_input)

        collected = df_out.rdd.map(lambda x: x.asDict()).collect()

        if isinstance(output_columns, (list, tuple)):
            if is_list_uniform(output_columns, str):
                chk_cols = [(i, i) for i in output_columns]
            else:
                chk_cols = output_columns[:]
        else:
            chk_cols = list(output_columns.items())

        for row in collected:
            input_dict = row["map_col"]
            if not input_dict:
                assert all(row[j] is None for (i, j) in chk_cols)
                continue
            for i, j in chk_cols:
                exp = input_dict.get(i)
                chk = row[j]
                assert exp == chk

    @staticmethod
    def test_missing_input_column():
        """Test MapToColumns passing an empty output column list."""
        with pytest.raises(AssertionError):
            MapToColumns(input_column="id", output_columns=[])

    @staticmethod
    def test_invalid_output_columns_type():
        """Test MapToColumns passing a wrong output column type."""
        with pytest.raises(TypeError):
            MapToColumns(input_column="id", output_columns={"a"})

    def test_non_map_type_column(self, df_input):
        """Test MapToColumns passing a non MapType column."""
        t = MapToColumns(input_column="id", output_columns=["x"])
        with pytest.raises(TypeError):
            t.transform(df_input)

    @staticmethod
    @pytest.mark.parametrize(
        "output_columns",
        [
            ["a", 1],
            ["a", None],
            [None],
            [("a", "col_a"), ("c", "col_a")],
            [("a", "col_a"), ("c", "col_a", "col_b")],
            [("a", "col_a"), ("c", 1)],
            {"a": "col_a", "c": "col_a"},
            {"a": "col_a", "c": 1},
        ],
    )
    def test_invalid_output_columns(output_columns):
        """Test MapToColumns passing a wrong type of 'output_columns'."""
        with pytest.raises(TypeError):
            MapToColumns(input_column="map_col", output_columns=output_columns)


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
        with pytest.raises(ValueError):
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


class TestValidateWindowFrameBoundaries:
    """Unit-test for 'validate_window_frame_boundaries' auxiliary function."""

    def test_valid_window(self):
        # Test valid integer boundaries
        assert validate_window_frame_boundaries(1, 10) == (1, 10)
        assert validate_window_frame_boundaries(-5, 5) == (-5, 5)
        start, end = validate_window_frame_boundaries("start", "end")
        assert start < 0
        assert end > 5

    def test_invalid_window(self):
        # Test None values
        with pytest.raises(ValueError):
            validate_window_frame_boundaries(None, 5)
        with pytest.raises(ValueError):
            validate_window_frame_boundaries(1, None)

        # Test invalid string values
        with pytest.raises(ValueError):
            validate_window_frame_boundaries("invalid", 5)
        with pytest.raises(ValueError):
            validate_window_frame_boundaries(1, "invalid")


class TestSparkColumnMethod:
    """Test SparkColumnMethod transformer."""

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
        with pytest.raises(ValueError):
            t = SparkColumnMethod(input_column="name", method="invalid")
            t.transform(df_input)

    def test_invalid_column(self, df_input):
        t = SparkColumnMethod(input_column="invalid", method="isNull")
        with pytest.raises(AnalysisException):
            t.transform(df_input)

    def test_valid(self, df_input):
        t = SparkColumnMethod(
            input_column="name", method="contains", output_column="result", args=["se"]
        )
        df_chk = t.transform(df_input)

        df_exp = df_input.withColumn("result", F.col("name").contains("se"))
        assert_df_equality(df_chk, df_exp, ignore_row_order=True)

    def test_no_args(self, df_input):
        """Test overriding the input column."""
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
        # This number sums up with base_len when "outer" = True
        null_len = len([True for i in data_col if not i])

        return base_len, null_len

    def test_array(self, df_input):
        """Test the transformer with <ArrayType> as input column."""
        inputs = [
            {"output_cols": self._COL_OUTPUT_ARRAY, "outer": True, "drop_after": True},
            {"output_cols": self._COL_OUTPUT_ARRAY, "outer": True, "drop_after": False},
            {"output_cols": self._COL_OUTPUT_ARRAY, "outer": False, "drop_after": True},
            {
                "output_cols": self._COL_OUTPUT_ARRAY,
                "outer": False,
                "drop_after": False,
            },
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
    def test_with_args(self, df_input, asc: bool):
        """Test SqlFunction."""
        t = SparkSqlFunction(
            column="result", function="sort_array", args=["data"], kwargs={"asc": asc}
        )
        df_chk = t.transform(df_input)

        df_exp = df_input.withColumn("result", F.sort_array("data", asc=asc))
        assert_df_equality(df_chk, df_exp, ignore_row_order=True)

    def test_without_args(self, df_input):
        """Test SqlFunction w/o any arguments."""
        t = SparkSqlFunction(column="result", function="rand")
        df_chk = t.transform(df_input)

        n_null: int = df_chk.filter(F.col("result").isNull()).count()
        assert n_null == 0


class TestWindow:
    """Unit-tests for '_Window' parent class."""

    def test_order_cols_range_between(self):
        """'order_cols' is null when 'range_between' is provided."""
        with pytest.raises(ValueError):
            _Window(
                partition_cols=["a"],
                order_cols=None,
                ascending=False,
                rows_between=None,
                range_between=(0, 10),
            )

    def test_aggregate_over_window_wrong_ascending(self):
        """Wrong ascending length."""
        with pytest.raises(ValueError):
            _Window(
                partition_cols=["a"],
                order_cols=["category", "group"],
                ascending=[True, False, False],
                rows_between=None,
                range_between=None,
            )
