"""Unit-tests for 'spark_util' module."""

import narwhals as nw
import pandas as pd
import pytest
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    StringType,
    StructField,
    StructType,
)

from nebula.nw_util import broadcast_spark
from nebula.spark_util import *

_nan = float("nan")


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


class TestBroadcastUtil:
    """Test broadcasting utils."""

    @staticmethod
    @pytest.fixture(scope="class", name="df")
    def _get_df_input(spark):
        fields = [
            StructField("c1", StringType(), True),
            StructField("c2", StringType(), True),
        ]
        data = [["a", "aa"], ["b", "bb"]]
        return spark.createDataFrame(data, schema=StructType(fields))

    def test_is_broadcast(self, df):
        assert not is_broadcast(df)
        assert is_broadcast(F.broadcast(df))

    @pytest.mark.parametrize("to_nw", [True, False])
    def test_broadcast_spark(self, df, to_nw):
        # this is narwhals util, but related to spark
        if to_nw:
            df = nw.from_native(df)
        df_chk = broadcast_spark(df)
        df_chk = nw.to_native(df_chk)
        assert is_broadcast(df_chk)
