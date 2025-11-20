"""Unit-tests for 'spark_util' module."""

import operator as py_operator
from datetime import datetime, timedelta

import pandas as pd
import pytest
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    NullType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from nlsn.nebula.spark_util import *
from nlsn.nebula.spark_util import (
    ALLOWED_SPARK_NULL_OPERATORS,
    ALLOWED_STANDARD_OPERATORS,
)

_nan = float("nan")


def test_get_column_data_type_name(spark):
    """Test the 'get_column_data_type_name' function."""
    fields = [
        StructField("c1", NullType(), True),
    ]
    df = spark.createDataFrame([[None]], schema=StructType(fields))

    types = [
        (IntegerType(), "integer"),
        (LongType(), "long"),
        ("timestamp", "timestamp"),
        (BooleanType(), "boolean"),
        ("float", "float"),
        ("double", "double"),
        ("decimal(10, 0)", "decimal"),
        ("array<string>", "array"),
        ("map<string,double>", "map"),
    ]
    for t, exp in types:
        df_cast = df.withColumn("c2", F.col("c1").cast(t))
        chk = get_column_data_type_name(df_cast, "c2")
        assert chk == exp


def test_is_valid_number(spark):
    """Test is_valid_number function."""
    fields = [
        StructField("c1", FloatType(), True),
        StructField("exp", BooleanType(), True),  # expected values
    ]
    schema = StructType(fields)

    input_data = [
        [1.5, True],
        [0.0, True],
        [None, False],
        [_nan, False],
    ]

    df = spark.createDataFrame(input_data, schema=schema).withColumn(
        "chk", is_valid_number("c1")
    )

    count: int = df.filter(F.col("chk") == F.col("exp")).count()
    assert count == len(input_data)


def test_is_broadcast(spark):
    """Test 'is_broadcast' function."""
    fields = [
        StructField("c1", StringType(), True),
        StructField("c2", StringType(), True),
    ]
    data = [["a", "aa"], ["b", "bb"]]
    df = spark.createDataFrame(data, schema=StructType(fields))
    assert not is_broadcast(df)
    assert is_broadcast(F.broadcast(df))


class TestSplitDfBoolCondition:
    """Test the 'split_df_bool_condition' function."""

    @staticmethod
    def _check_split_df_bool_condition(df1, exp_1, df2, exp_2, tot):
        assert (exp_1 + exp_2) == tot

        count_1 = df1.count()
        assert count_1 == exp_1

        count_2 = df2.count()
        assert count_2 == exp_2

    def test_split_df_bool_condition(self, spark):
        """Test the 'split_df_bool_condition' function."""
        fields = [
            StructField("c1", StringType(), True),
            StructField("c2", FloatType(), True),
        ]
        schema = StructType(fields)

        input_data = [
            ["a", 0.0],
            ["b", 2.0],
            ["", _nan],
            [None, None],
        ]

        row_count = len(input_data)

        df_input = spark.createDataFrame(input_data, schema=schema)

        cond_str = F.col("c1") == "a"
        df1_str, df2_str = split_df_bool_condition(df_input, cond_str)
        self._check_split_df_bool_condition(df1_str, 1, df2_str, 3, row_count)

        cond_num = F.col("c2") == 2.0
        df1_num, df2_num = split_df_bool_condition(df_input, cond_num)
        self._check_split_df_bool_condition(df1_num, 1, df2_num, 3, row_count)


class TestCacheIfNeeded:
    """Test the 'cache_if_needed' function."""

    @staticmethod
    def _get_input_df(spark):
        fields = [
            StructField("c1", StringType(), True),
            StructField("c2", StringType(), True),
        ]
        data = [
            ["a", "aa"],
            ["b", "bb"],
            ["c", "cc"],
        ]

        schema = StructType(fields)
        return spark.createDataFrame(data, schema=schema)

    def test_non_spark(self):
        """Ensure no errors when the DF is not a spark one."""
        df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
        df_chk = cache_if_needed(df, True)
        assert df_chk is df

    @pytest.mark.parametrize("cached", [False, True])
    @pytest.mark.parametrize("do_cache", [False, True])
    def test_cache_if_needed(self, spark, do_cache, cached):
        """Unit-test for 'cache_if_needed' function."""
        df = self._get_input_df(spark)

        if cached:
            df = df.persist()

        df_chk = cache_if_needed(df, do_cache)

        if cached:
            assert df_chk.is_cached
        else:
            if do_cache:
                assert df_chk.is_cached
            else:
                assert not df_chk.is_cached


class TestCastToSchema:
    """Test the 'cast_to_schema' function."""

    fields_orig = [
        StructField("col_array", ArrayType(IntegerType(), True), True),
        StructField("col_map", MapType(StringType(), IntegerType(), True), True),
        StructField("col_string", IntegerType(), True),
    ]

    fields_cast = [
        StructField("col_string", LongType(), True),
        StructField("col_array", ArrayType(LongType(), True), True),
        StructField("col_map", MapType(StringType(), LongType(), True), True),
    ]

    schema_orig = StructType(fields_orig)
    schema_cast = StructType(fields_cast)

    @pytest.fixture(scope="class", name="df_input")
    def _get_df_input(self, spark):
        """Get input dataframe."""
        data = [
            ([1, 2, 3], {"key1": 4, "key2": 5, "key3": 1}, 3),
            ([4, 5], {"key1": 3, "key2": 2}, 2),
            ([6], {"key1": 2}, 1),
            (None, None, -1),
        ]
        return spark.createDataFrame(data, schema=self.schema_orig)

    @pytest.mark.parametrize(
        "schema, collect", [(schema_orig, False), (schema_cast, False)]
    )
    def test_cast_to_schema_shuffle_columns(self, df_input, schema, collect: bool):
        """Test the 'cast_to_schema' function with shuffled columns."""
        df = df_input.select("col_array", "col_string", "col_map")
        df_chk = cast_to_schema(df, schema)
        assert df_chk.schema == schema
        if collect:  # Just to ensure that the function works fine.
            df_chk.collect()

    def test_cast_to_schema_error(self, df_input):
        """Test 'cast_to_schema' with wrong columns."""
        df_less = df_input.drop("col_string", "col_map")
        with pytest.raises(AssertionError):
            cast_to_schema(df_less, self.schema_orig)

        df_more = df_input.withColumn("col_more", F.lit(True))
        with pytest.raises(AssertionError):
            cast_to_schema(df_more, self.schema_orig)


class TestCompareDFs:
    """Unit-tests for 'compare_dfs' function."""

    def test_dfs_schema_mismatch(self, spark):
        """Test case: Schemas mismatch."""
        data1 = [("Alice", 1)]
        columns1 = ["name", "age"]
        df1 = spark.createDataFrame(data1, columns1)

        data2 = [("Bob", "2")]  # age is string
        columns2 = ["name", "age"]
        df2 = spark.createDataFrame(data2, columns2)

        with pytest.raises(AssertionError):
            compare_dfs(df1, df2)

    def test_dfs_different_columns_no_columns_arg(self, spark):
        """Test case: Different column sets when columns=None."""
        data1 = [("Alice", 1)]
        columns1 = ["name", "age"]
        df1 = spark.createDataFrame(data1, columns1)

        data2 = [("Bob", 2, "USA")]
        columns2 = ["name", "age", "country"]
        df2 = spark.createDataFrame(data2, columns2)

        with pytest.raises(AssertionError):
            compare_dfs(df1, df2, columns=None)

    def test_dfs_row_count_mismatch_raise(self, spark):
        """Test case: Row count mismatch, raise error."""
        data1 = [("Alice", 1), ("Bob", 2)]
        columns = ["name", "age"]
        df1 = spark.createDataFrame(data1, columns)

        data2 = [("Alice", 1)]  # one row less
        df2 = spark.createDataFrame(data2, columns)

        with pytest.raises(AssertionError):
            compare_dfs(df1, df2, raise_if_row_number_mismatch=True)

    def test_dfs_row_count_mismatch_no_raise(self, spark):
        """Test case: Row count mismatch, do not raise."""
        data1 = [("Alice", 1), ("Bob", 2)]
        columns = ["name", "age"]
        df1 = spark.createDataFrame(data1, columns)

        data2 = [("Alice", 1)]  # one row less
        df2 = spark.createDataFrame(data2, columns)

        # Since row count mismatches and return_mismatched_rows are False (default),
        # it will print the count warning and then raise the content mismatch error.
        with pytest.raises(AssertionError):
            compare_dfs(
                df1,
                df2,
                raise_if_row_number_mismatch=False,
                return_mismatched_rows=False,
            )

    def test_dfs_content_mismatch_same_count_raise(self, spark):
        """Test case: Content mismatch, same count, raise error."""
        data1 = [("Alice", 1), ("Bob", 2)]
        columns = ["name", "age"]
        df1 = spark.createDataFrame(data1, columns)

        data2 = [("Alice", 1), ("Charlie", 3)]  # different content
        df2 = spark.createDataFrame(data2, columns)

        with pytest.raises(AssertionError):
            compare_dfs(df1, df2, return_mismatched_rows=False)

    def test_dfs_content_mismatch_diff_count_no_raise_return_diff(self, spark):
        """Test case: Content mismatch, different count, no raise on count, return diffs."""
        data1 = [("Alice", 1), ("Bob", 2), ("David", 4)]
        columns = ["name", "age"]
        df1 = spark.createDataFrame(data1, columns)

        data2 = [("Alice", 1), ("Charlie", 3)]  # Alice matches, Bob/David vs Charlie
        df2 = spark.createDataFrame(data2, columns)

        df1_diff, df2_diff = compare_dfs(
            df1, df2, raise_if_row_number_mismatch=False, return_mismatched_rows=True
        )

        # Check df1_diff: should contain rows ("Bob", 2), ("David", 4)
        assert df1_diff is not None

        df1_diff_pd = df1_diff.toPandas().sort_values("name").reset_index(drop=True)
        assert df1_diff_pd.shape[0] == 2

        df1_diff_exp = pd.DataFrame(
            [{"name": "Bob", "age": 2}, {"name": "David", "age": 4}]
        )
        pd.testing.assert_frame_equal(df1_diff_pd, df1_diff_exp)

        # Check df2_diff: should contain row ("Charlie", 3)
        df2_diff_pd = df2_diff.toPandas()
        assert df2_diff is not None
        assert df2_diff_pd.shape[0] == 1
        assert df2_diff_pd.to_dict("records") == [{"name": "Charlie", "age": 3}]

    def test_dfs_subset_columns_mismatch_return_diff(self, spark):
        """Test case: Compare only a subset of columns, they mismatch, return diffs."""
        data1 = [("Alice", 1, "USA"), ("Bob", 2, "Canada")]
        df1 = spark.createDataFrame(data1, ["name", "age", "country"])

        data2 = [("Alice", 1, "USA"), ("Mark", 3, "Canada")]
        df2 = spark.createDataFrame(data2, ["name", "age", "country"])

        columns_to_compare = ["name", "age"]
        df1_diff, df2_diff = compare_dfs(
            df1, df2, columns=columns_to_compare, return_mismatched_rows=True
        )

        # Check df1_diff: should contain row ("Bob", 2) based on name/age columns
        assert df1_diff is not None
        assert df1_diff.count() == 1
        assert df1_diff.toPandas().to_dict("records") == [{"name": "Bob", "age": 2}]

        # Check df2_diff: should contain row ("Charlie", 3) based on name/age columns
        assert df2_diff is not None
        assert df2_diff.count() == 1
        assert df2_diff.toPandas().to_dict("records") == [{"name": "Mark", "age": 3}]

    def test_dfs_subset_columns_match(self, spark):
        """Test case: Compare only a subset of columns, they match."""
        data1 = [("Alice", 1, "USA"), ("Bob", 2, "Canada")]
        df1 = spark.createDataFrame(data1, ["name", "age", "country"])

        data2 = [("Alice", 1, "UK"), ("Bob", 2, "Mexico")]  # Different country column
        df2 = spark.createDataFrame(data2, ["name", "age", "country"])

        columns_to_compare = ["name", "age"]
        result_df1, result_df2 = compare_dfs(df1, df2, columns=columns_to_compare)

        assert result_df1 is None
        assert result_df2 is None

    def test_dfs_content_mismatch_order_does_not_matter(self, spark):
        """Test case: Content mismatch due to order, but should match."""
        data1 = [("Alice", 1), ("Bob", 2)]
        columns = ["name", "age"]
        df1 = spark.createDataFrame(data1, columns)

        data2 = [("Bob", 2), ("Alice", 1)]  # Same rows, different order
        df2 = spark.createDataFrame(data2, columns)

        result_df1, result_df2 = compare_dfs(df1, df2)

        assert result_df1 is None
        assert result_df2 is None


class TestDropDuplicatesNoRandomness:
    """Test the 'drop_duplicates_no_randomness' function."""

    fields = [
        StructField("platform", StringType(), True),
        StructField("device_type", StringType(), True),
        StructField("os_group", StringType(), True),
    ]

    input_data = [
        ["OTT", "STV", "Android"],
        ["OTT", "STV", None],
        ["OTT", "STV", "Android"],
        ["OTT", "STV", "Linux"],
        ["OTT", "STV", "Android"],
        ["OTT", "STV", None],
        ["DSK", "DSK", "Windows"],
        ["DSK", "DSK", None],
        ["DSK", "DSK", "MacOS"],
        ["DSK", "DSK", "MacOS"],
        ["DSK", "DSK", "MacOS"],
        ["MBL", "TAB", None],
        ["MBL", "TAB", None],
        ["MBL", "TAB", None],
        ["MBL", "TAB", None],
        ["MBL", "PHN", None],
        ["MBL", "PHN", "iOS"],
    ]

    @staticmethod
    def _test_complex_types(df_input):
        mapping = F.create_map(F.lit("a"), F.lit("1"), F.lit("b"), F.lit("2"))
        array = F.array(F.lit("x"), F.lit("y"))

        df_complex = df_input.withColumn("mapping", mapping).withColumn("array", array)

        subset = ["platform", "device_type"]
        chk = drop_duplicates_no_randomness(df_complex, subset)

        # just to check if MapType and ArrayType raise any error
        chk.count()

    @pytest.mark.parametrize("subset", ["", "  ", []])
    def test_drop_duplicates_no_randomness_empty_subset(self, subset):
        """Raise AssertionError if 'subset' is empty."""
        with pytest.raises(AssertionError):
            drop_duplicates_no_randomness(None, subset)

    def test_drop_duplicates_no_randomness(self, spark):
        """Test drop_duplicates_no_randomness function."""
        df = spark.createDataFrame(data=self.input_data, schema=StructType(self.fields))

        # just to check if MapType and ArrayType raise any error
        self._test_complex_types(df)

        # check if the same df order in the opposite manner gives the same result
        df_desc = df.sort([F.col(i).desc() for i in df.columns])
        df_asc = df.sort([F.col(i).asc() for i in df.columns])

        subset = ["platform", "device_type"]
        chk_desc = drop_duplicates_no_randomness(df_desc, subset)
        chk_asc = drop_duplicates_no_randomness(df_asc, subset)

        assert_df_equality(chk_desc, chk_asc, ignore_row_order=True)


class TestEnsureSparkCondition:
    """Test 'ensure_spark_condition' function."""

    @staticmethod
    @pytest.mark.parametrize("op", ALLOWED_SPARK_NULL_OPERATORS)
    def test_valid_null_operator(op):
        """Test with a valid null operator."""
        ensure_spark_condition(op)

    @staticmethod
    @pytest.mark.parametrize("op", ALLOWED_STANDARD_OPERATORS)
    def test_valid_standard_operator(op):
        """Test with a valid standard operator."""
        ensure_spark_condition(op, 1)

    @staticmethod
    def test_valid_rlike():
        """Test valid rlike operator."""
        ensure_spark_condition("rlike", value=r"\d{3}-\d{2}-\d{4}")

    @staticmethod
    def test_both_value_and_col_compare_provided():
        """Test with wrong input."""
        with pytest.raises(ValueError):
            ensure_spark_condition("eq", 10, "col2")

    @staticmethod
    @pytest.mark.parametrize("op", ["rlike", "between"])
    def test_invalid_operator_comparison_column(op: str):
        """Test with not allowed operator for 'compare_col'."""
        with pytest.raises(ValueError):
            ensure_spark_condition(op, compare_col="compare")

    @staticmethod
    @pytest.mark.parametrize("value", ["string", 1, None, [None]])
    @pytest.mark.parametrize("op", ["isin", "isnotin"])
    def test_invalid_isin_value(value, op):
        """Test with not allowed value for 'isin' / 'isnotin' operator."""
        with pytest.raises(TypeError):
            ensure_spark_condition(op, value=value)

    @pytest.mark.parametrize("value", [{1: 2, 3: 4}, [1], [1, 2, 3]])
    def test_invalid_between_values(self, value):
        """Test with not allowed value for between operator."""
        with pytest.raises((ValueError, TypeError)):
            ensure_spark_condition("between", value=value)

    @pytest.mark.parametrize(
        "op", ["contains", "startswith", "endswith", "like", "rlike"]
    )
    def test_invalid_string_operator_values(self, op: str):
        """Test with not allowed value for string operators."""
        with pytest.raises(AssertionError):
            ensure_spark_condition(op, value=1)


class TestGetSparkCondition:
    """Test the 'get_spark_condition' function."""

    @staticmethod
    @pytest.fixture(scope="class", name="df_input")
    def _get_df_input(spark):
        fields = [
            StructField("input_string", StringType(), True),
            StructField("input_long", LongType(), True),
            StructField("input_float", FloatType(), True),
            StructField("input_array", ArrayType(IntegerType()), True),
        ]

        data = [
            ["a", 0, 0.0, [1, 2]],
            ["a1", 1, 1.0, [2, 3]],
            ["a", 2, 2.0, [3, 4]],
            ["a1", 3, 3.0, [4, 5]],
            ["b", 4, 4.0, [5, 6]],
            ["b", 5, 5.0, [6, 7]],
            ["b1", 6, 6.0, [7, 8]],
            ["c1", 7, 7.0, [8, 9]],
            ["c1", 8, 8.0, [9, 10]],
            ["c1", 9, 9.0, [10, 11]],
            ["  ", 10, float("nan"), [None, 12]],
            [None, None, None, None],
        ]

        return spark.createDataFrame(data, schema=StructType(fields)).persist()

    @staticmethod
    def _check(df_input, cmp_str, exp_cond, chk_cond):
        df = df_input.withColumn("expected", exp_cond).withColumn("check", chk_cond)
        n_diff = df.filter(F.col("expected") != F.col("check")).count()
        assert n_diff == 0, f"Test failed when checking {cmp_str}"

    @pytest.mark.parametrize("op", ALLOWED_STANDARD_OPERATORS)
    @pytest.mark.parametrize("col_name", ["input_float", "input_long"])
    def test_py_operator_vs_value(self, df_input, op, col_name):
        """Test with a valid python operator."""
        value = 5
        cmp = getattr(py_operator, op)

        # Create the expected condition
        exp_cond = cmp(F.col(col_name), value)

        if col_name == "input_float":
            if op in {"le", "lt", "ge", "gt"}:
                exp_cond &= ~F.isnan(col_name)

        chk_cond = get_spark_condition(df_input, col_name, op, value=value)
        self._check(df_input, op, exp_cond, chk_cond)

    @pytest.mark.parametrize("op", ["isin", "isnotin"])
    @pytest.mark.parametrize(
        "value",
        [
            ["a"],
            {"a"},
            frozenset("a"),
            ("a",),
            {"a": 1},
            {"a": 1}.keys(),
            {"a": 1}.values(),
        ],
    )
    def test_isin(self, df_input, op: str, value):
        """Test with 'isin' and 'isnotin' operators."""
        col_name = "input_string"
        chk_cond = get_spark_condition(df_input, col_name, op, value=value)

        value = list(value)
        exp_cond = F.col(col_name).isin(value)
        if op == "isnotin":
            exp_cond = ~F.when(exp_cond.isNull(), F.lit(False)).otherwise(
                F.lit(exp_cond)
            )

        self._check(df_input, op, exp_cond, chk_cond)

    @pytest.mark.parametrize(
        "op, value",
        [
            ("contains", "a"),
            ("startswith", "a"),
            ("endswith", "a"),
            ("like", "c_"),
            ("rlike", "ab[0-9]+"),
        ],
    )
    def test_vs_string_value(self, df_input, op, value):
        """Test spark operators with strings."""
        col_name = "input_string"
        spark_col = F.col(col_name)
        exp_cond = getattr(spark_col, op)(value)
        chk_cond = get_spark_condition(df_input, col_name, op, value=value)
        self._check(df_input, op, exp_cond, chk_cond)

    @pytest.mark.parametrize("op", ["isNull", "isNotNull"])
    def test_nulls(self, df_input, op):
        """Test with 'isNull' and 'isNotNull' operator."""
        col_name = "input_float"
        spark_col = F.col(col_name)
        exp_cond = getattr(spark_col, op)()
        chk_cond = get_spark_condition(df_input, col_name, op)
        self._check(df_input, op, exp_cond, chk_cond)

    @pytest.mark.parametrize("op", ["isNaN", "isNotNaN"])
    def test_nans(self, df_input, op):
        """Test with 'isNaN' and 'isNotNaN' operator."""
        col_name = "input_float"
        op = "isNaN"
        exp_cond = F.isnan(F.col(col_name))
        if op == "isNotNaN":
            exp_cond = ~exp_cond
        chk_cond = get_spark_condition(df_input, col_name, op)
        self._check(df_input, op, exp_cond, chk_cond)

    def test_between(self, df_input):
        """Test with the 'between' operator."""
        col_name = "input_float"
        op, value = "between", (1, 6)
        exp_cond = F.col(col_name).between(value[0], value[1])
        chk_cond = get_spark_condition(df_input, col_name, op, value=value)
        self._check(df_input, op, exp_cond, chk_cond)

    def test_array_contains(self, df_input):
        """Test with the 'array_contains' operator."""
        col_name = "input_array"
        op, value = "array_contains", 5
        exp_cond = F.array_contains(col_name, value)
        chk_cond = get_spark_condition(df_input, col_name, op, value=value)
        self._check(df_input, op, exp_cond, chk_cond)

    @pytest.mark.parametrize("op", ["le", "lt", "ge", "gt"])
    def test_timestamp_columns(self, spark, op):
        """Test against 2 timestamp columns."""
        n = 5
        start_date = datetime(2025, 1, 1)
        dates_1 = [start_date + timedelta(days=i) for i in range(n)]
        dates_2 = dates_1.copy()

        dates_1[2] = None  # Spark doesn't have the equivalent of NaT
        dates_1[3] = None

        dates_2[1] -= timedelta(days=1)
        dates_2[2] = None
        dates_2[4] += timedelta(days=1)

        schema = StructType(
            [
                StructField("c1", TimestampType(), True),
                StructField("c2", TimestampType(), True),
            ]
        )

        df = spark.createDataFrame(zip(dates_1, dates_2), schema)
        exp_cond = getattr(py_operator, op)(F.col("c1"), F.col("c2"))
        chk_cond = get_spark_condition(df, "c1", op, compare_col="c2")
        self._check(df, op, exp_cond, chk_cond)


class TestFunctionHashDataFrame:
    """Test the 'hash_dataframe' function."""

    @staticmethod
    @pytest.fixture(scope="class", name="df_input")
    def _get_df_input(spark):
        fields = [
            StructField("c1", StringType(), True),
            StructField("c2", FloatType(), True),
            StructField("c3", ArrayType(IntegerType()), True),
        ]

        data = [
                   ["a", 0.0, [1, 2]],
                   ["  ", float("nan"), [None, 12]],
                   [None, None, None],
               ] * 100

        return spark.createDataFrame(data, schema=StructType(fields)).persist()

    @staticmethod
    def test_wrong_input(df_input):
        """Test with invalid hash name."""
        with pytest.raises(AssertionError):
            hash_dataframe(df_input, "md5", new_col="hashed", return_func=True)

    @staticmethod
    @pytest.mark.parametrize("hash_name", ["md5", "crc32", "sha1", "sha2", "xxhash64"])
    def test_hash_dataframe(df_input, hash_name):
        """Test with different hash types."""
        df1 = hash_dataframe(df_input, hash_name, new_col="hashed")
        df2 = hash_dataframe(
            df_input.repartition(20).sort("c1"), hash_name, new_col="hashed"
        )

        assert_df_equality(
            df1,
            df2,
            ignore_row_order=True,
            ignore_nullable=True,
            allow_nan_equality=True,
        )

    @staticmethod
    def test_hash_dataframe_return_func(df_input):
        """Test with 'return_func' True."""
        hashed_col = hash_dataframe(df_input, "md5", return_func=True)
        df1 = df_input.withColumn("hashed", hashed_col)
        df2 = hash_dataframe(
            df_input.repartition(20).sort("c1"), "md5", new_col="hashed"
        )

        assert_df_equality(
            df1,
            df2,
            ignore_row_order=True,
            ignore_nullable=True,
            allow_nan_equality=True,
        )


class TestSparkSchemaUtilities:
    """Test spark schema utilities."""

    @staticmethod
    @pytest.fixture(scope="class", name="df_input")
    def _get_input_df(spark):
        # value | column name | datatype
        input_data = [
            ("string A", "col_1", StringType()),
            (1.5, "col_2", FloatType()),
            (3, "col_3", IntegerType()),
            ([1, 2, 3], "col_4", ArrayType(IntegerType())),
            ({"a": 1}, "col_5", MapType(StringType(), IntegerType())),
            ([[1, 2], [2, 3]], "col_6", ArrayType(ArrayType(IntegerType()))),
            (
                {"a": {"b": 2}},
                "col_7",
                MapType(StringType(), MapType(StringType(), IntegerType())),
            ),
        ]

        data = [[i[0] for i in input_data]]
        fields = [StructField(*i[1:]) for i in input_data]
        schema = StructType(fields)
        return spark.createDataFrame(data, schema=schema)

    @pytest.mark.parametrize("full_type_name", [True, False])
    def test_get_schema_as_str(self, df_input, full_type_name: bool):
        """Test get_schema_as_str function."""
        li_fields: list[tuple[str, str]] = get_schema_as_str(df_input, full_type_name)

        meth = "simpleString" if full_type_name else "typeName"

        assert len(li_fields) == len(df_input.schema)
        for idx, field in enumerate(df_input.schema):
            name = field.name
            datatype = field.dataType

            name_chk, datatype_str = li_fields[idx]
            assert name == name_chk

            type_name_exp: str = getattr(datatype, meth)()
            assert type_name_exp == datatype_str
