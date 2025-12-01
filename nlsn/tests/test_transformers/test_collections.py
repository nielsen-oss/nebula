import pandas as pd
import polars as pl
import pytest
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql.types import StringType, StructField, StructType

from nlsn.nebula.storage import nebula_storage as ns
from nlsn.nebula.transformers import *
from nlsn.tests.auxiliaries import from_pandas, to_pandas
from nlsn.tests.constants import TEST_BACKENDS


class TestAppendDataFrame:
    @staticmethod
    def _set_dfs(spark, backend: str, to_nw: bool):
        ns.allow_overwriting()
        df1 = pd.DataFrame({"c1": ["c", "d"], "c2": [3, 4]})
        df2 = pd.DataFrame({"c1": ["a", "b"], "c3": [4.5, 5.5]})
        df1 = from_pandas(df1, backend, to_nw, spark=spark)
        df2 = from_pandas(df2, backend, to_nw, spark=spark)
        ns.set("df1", df1)
        ns.set("df2", df2)

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    @pytest.mark.parametrize("to_nw", [True, False])
    @pytest.mark.parametrize("allow_missing", [True, False])
    def test_exact_columns(self, spark, backend: str, to_nw: bool, allow_missing: bool):
        self._set_dfs(spark, backend, to_nw)
        df_pd_in = pd.DataFrame({"c1": ["a", "b"], "c2": [1, 2]})
        df = from_pandas(df_pd_in, backend, to_nw=to_nw, spark=spark)
        t = AppendDataFrame(store_key="df1", allow_missing_columns=allow_missing)
        df_chk = t.transform(df)
        df_chk_pd = to_pandas(df_chk).reset_index(drop=True)
        df_exp = pd.concat([df_pd_in, to_pandas(ns.get("df1"))], axis=0)
        pd.testing.assert_frame_equal(df_chk_pd, df_exp.reset_index(drop=True))

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    @pytest.mark.parametrize("to_nw", [True, False])
    def test_invalid_columns(self, spark, backend: str, to_nw: bool):
        self._set_dfs(spark, backend, to_nw)
        df_pd_in = pd.DataFrame({"c1": ["a", "b"], "c2": [1, 2]})
        df = from_pandas(df_pd_in, backend, to_nw=to_nw, spark=spark)
        t = AppendDataFrame(store_key="df2", allow_missing_columns=False)
        with pytest.raises(ValueError):
            t.transform(df)

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    @pytest.mark.parametrize("to_nw", [True, False])
    def test_missing_columns(self, spark, backend: str, to_nw: bool):
        self._set_dfs(spark, backend, to_nw)
        df_pd_in = pd.DataFrame({"c1": ["a", "b"], "c2": [1, 2]})
        df = from_pandas(df_pd_in, backend, to_nw=to_nw, spark=spark)
        t = AppendDataFrame(store_key="df2", allow_missing_columns=True)
        df_chk = t.transform(df)
        df_chk_pd = to_pandas(df_chk).reset_index(drop=True)
        df_exp = pd.concat([df_pd_in, to_pandas(ns.get("df2"))], axis=0)
        pd.testing.assert_frame_equal(df_chk_pd, df_exp.reset_index(drop=True))


class TestDropNulls:
    """Test suite for DropNulls transformer."""

    @staticmethod
    @pytest.fixture(scope="class", name="df_input_spark")
    def _get_df_input(spark):
        fields = [
            StructField("a_1", StringType(), True),
            StructField("a_2", StringType(), True),
            StructField("a_3", StringType(), True),
            StructField("b_1", StringType(), True),
            StructField("b_2", StringType(), True),
            StructField("b_3", StringType(), True),
        ]
        schema = StructType(fields)

        data = [
            ("1", "11", None, "4", "41", "411"),
            ("1", "12", "120", "4", None, "412"),
            ("1", "12", "120", "4", "41", "412"),
            (None, None, None, None, None, None),
            ("1", "12", "120", "4", "41", None),
            (None, None, None, "4", "41", "412"),
        ]
        return spark.createDataFrame(data, schema=schema).persist()

    def test_spark_no_subset(self, df_input_spark):
        """Test DiscardNulls transformer w/o any subsets."""
        how = "any"
        t = DropNulls(how=how)
        df_chk = t.transform(df_input_spark)
        df_exp = df_input_spark.dropna(how=how)
        assert_df_equality(df_chk, df_exp, ignore_row_order=True)

    @pytest.mark.parametrize("how", ["any", "all"])
    def test_spark_columns_subset(self, df_input_spark, how):
        """Test DiscardNulls transformer selecting specific columns."""
        subset = ["a_1", "b_1"]
        t = DropNulls(columns=subset, how=how)
        df_chk = t.transform(df_input_spark)

        df_exp = df_input_spark.dropna(subset=subset, how=how)

        assert_df_equality(df_chk, df_exp, ignore_row_order=True)

    def test_polars_any_all_columns(self):
        """Test dropping rows with any null/NaN across all columns."""
        df = pl.DataFrame({
            "user_id": [1, 2, 3, 4, 5],
            "age": [25.0, 30.0, 35.0, float('nan'), 45.0],
            "score": [100.0, 200.0, None, 400.0, 500.0],
        })

        t = DropNulls(how="any", drop_na=True)
        result = t.transform(df)

        # Rows 3 (null) and 4 (NaN) are dropped
        assert len(result) == 3
        assert result["user_id"].to_list() == [1, 2, 5]

    def test_polars_all_requires_all_missing(self):
        """Test dropping rows only when ALL values are null/NaN."""
        df = pl.DataFrame({
            "col_a": [1.0, None, None, 4.0],
            "col_b": [10.0, 20.0, None, 40.0],
            "col_c": [100.0, 200.0, None, 400.0],
        })

        t = DropNulls(how="all", drop_na=True)
        result = t.transform(df)

        # Only third row has all nulls
        assert len(result) == 3
        assert result["col_a"][0] == 1.0
        assert result["col_a"][1] is None
        assert result["col_a"][2] == 4.0

    def test_polars_subset_columns(self):
        """Test dropping rows based on nulls in specific columns only."""
        df = pl.DataFrame({
            "user_id": [1, 2, 3, 4],
            "score_primary": [50.0, 100.0, 150.0, None],
            "score_secondary": [100.0, None, 300.0, None],
            "note": [None, "good", None, "excellent"],
        })

        # Only check score columns - ignore note nulls
        t = DropNulls(how="any", columns=["score_primary", "score_secondary"])
        result = t.transform(df)

        # Rows 1 and 3 have nulls in BOTH score columns
        # Rows 2 and 4 have at least one score populated
        assert len(result) == 2
        assert result["user_id"].to_list() == [1, 3]

    def test_polars_with_pattern_and_nan_handling(self):
        """Test dropping rows with NaN vs null distinction."""
        df = pl.DataFrame({
            "user_id": [1, 2, 3, 4],
            "revenue_q1": [100.0, float('nan'), 300.0, 400.0],
            "revenue_q2": [200.0, 250.0, None, 450.0],
            "cost_q1": [50.0, 60.0, 70.0, None],
        })

        # Check revenue columns, including NaN
        t = DropNulls(how="any", glob="revenue_*", drop_na=True)
        result = t.transform(df)

        # Rows 2 (NaN) and 3 (null) have missing revenue values
        assert len(result) == 2
        assert result["user_id"].to_list() == [1, 4]

    def test_polars_ignore_nan(self):
        """Test ignoring NaN when drop_null=False."""
        df = pl.DataFrame({
            "user_id": [1, 2, 3],
            "value": [100.0, float('nan'), None],
        })

        # Only drop actual nulls, not NaN
        t = DropNulls(how="any", drop_na=False)
        result = t.transform(df)

        # Row 3 has null, row 2 has NaN (should remain)
        assert len(result) == 2
        assert result["user_id"].to_list() == [1, 2]
        # Verify NaN is still present
        assert result["value"][1] != result["value"][1]  # NaN != NaN


class TestJoin:
    """Test Join transformer with Polars."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Clear storage before and after each test."""
        ns.clear()
        yield
        ns.clear()

    @pytest.fixture
    def df_left(self):
        """Sample left dataframe."""
        return pl.DataFrame({
            "user_id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Charlie", "David"],
            "age": [25, 30, 35, 40]
        })

    @pytest.fixture
    def df_right(self):
        """Sample right dataframe."""
        return pl.DataFrame({
            "user_id": [2, 3, 4, 5],
            "city": ["NYC", "LA", "Chicago", "Boston"],
            "country": ["USA", "USA", "USA", "USA"]
        })

    def test_inner_join_basic(self, df_left, df_right):
        """Test basic inner join on single column."""
        ns.set("right_table", df_right)

        transformer = Join(store_key="right_table", on="user_id", how="inner")
        result = transformer.transform(df_left)

        assert result.shape == (3, 5)  # 3 matching rows, 5 columns
        assert set(result.columns) == {"user_id", "name", "age", "city", "country"}
        assert result["user_id"].to_list() == [2, 3, 4]

    def test_left_join(self, df_left, df_right):
        """Test left join keeps all left rows."""
        ns.set("right_table", df_right)

        transformer = Join(store_key="right_table", on="user_id", how="left")
        result = transformer.transform(df_left)

        assert result.shape == (4, 5)  # All 4 left rows
        assert result["user_id"].to_list() == [1, 2, 3, 4]
        # First row should have nulls for right columns
        assert result.filter(pl.col("user_id") == 1)["city"][0] is None

    def test_different_column_names(self):
        """Test join with different column names using left_on/right_on."""
        df_orders = pl.DataFrame({
            "order_id": [101, 102, 103],
            "customer_id": [1, 2, 3],
            "amount": [100, 200, 150]
        })

        df_customers = pl.DataFrame({
            "id": [1, 2, 3],
            "customer_name": ["Alice", "Bob", "Charlie"]
        })

        ns.set("customers", df_customers)

        transformer = Join(
            store_key="customers",
            left_on="customer_id",  # this column will not be present in the output
            right_on="id",
            how="inner"
        )
        result = transformer.transform(df_orders)

        assert result.shape == (3, 4)
        assert "customer_id" in result.columns
        assert "customer_name" in result.columns

    def test_anti_join(self, df_left, df_right):
        """Test anti join returns only non-matching left rows."""
        ns.set("right_table", df_right)

        transformer = Join(store_key="right_table", on="user_id", how="anti")
        result = transformer.transform(df_left)

        assert result.shape == (1, 3)  # Only user_id=1 doesn't match
        assert result["user_id"].to_list() == [1]
        assert result["name"].to_list() == ["Alice"]

    def test_suffix_handling(self):
        """Test suffix is applied to overlapping columns."""
        df_a = pl.DataFrame({
            "id": [1, 2],
            "value": [10, 20],
            "status": ["active", "inactive"]
        })

        df_b = pl.DataFrame({
            "id": [1, 2],
            "value": [100, 200],  # Overlapping column
            "category": ["A", "B"]
        })

        ns.set("table_b", df_b)

        transformer = Join(
            store_key="table_b",
            on="id",
            how="inner",
            suffix="_b"
        )
        result = transformer.transform(df_a)

        assert "value" in result.columns
        assert "value_b" in result.columns
        assert result["value"].to_list() == [10, 20]
        assert result["value_b"].to_list() == [100, 200]

    def test_multiple_join_keys(self):
        """Test join on multiple columns."""
        df_sales = pl.DataFrame({
            "region": ["North", "North", "South"],
            "product": ["A", "B", "A"],
            "sales": [100, 150, 200]
        })

        df_targets = pl.DataFrame({
            "region": ["North", "North", "South"],
            "product": ["A", "B", "A"],
            "target": [120, 140, 180]
        })

        ns.set("targets", df_targets)

        transformer = Join(
            store_key="targets",
            on=["region", "product"],
            how="inner"
        )
        result = transformer.transform(df_sales)

        assert result.shape == (3, 4)
        assert set(result.columns) == {"region", "product", "sales", "target"}

    def test_right_join_via_swap(self, df_left, df_right):
        """Test right join is implemented by swapping dataframes."""
        ns.set("right_table", df_right)

        transformer = Join(store_key="right_table", on="user_id", how="right")
        result = transformer.transform(df_left)

        # Right join keeps all right rows (user_id 2,3,4,5)
        assert result.shape == (4, 5)
        assert result["user_id"].to_list() == [2, 3, 4, 5]
        # Last row should have nulls for left columns
        assert result.filter(pl.col("user_id") == 5)["name"][0] is None

    def test_cross_join(self):
        """Test cross join produces cartesian product."""
        df_colors = pl.DataFrame({"color": ["red", "blue"]})
        df_sizes = pl.DataFrame({"size": ["S", "M", "L"]})

        ns.set("sizes", df_sizes)

        transformer = Join(store_key="sizes", how="cross")
        result = transformer.transform(df_colors)

        assert result.shape == (6, 2)  # 2 * 3 = 6 rows
        assert set(result.columns) == {"color", "size"}
