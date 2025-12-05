import operator as py_operator

import narwhals as nw
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


class TestMathOperator:
    """Test suite for MathOperator transformer."""

    _CONST: float = 11.0

    @pytest.fixture(scope="class")
    def df_input(self) -> pl.DataFrame:
        """Create input DataFrame for testing."""
        data = {
            "col1": [1, 2, 4, 0, 0, None, None, 0, None, 4],
            "col2": [3, 5, 0, 4, 0, None, 0, None, 3, None],
        }
        df = pl.DataFrame(data, schema={"col1": pl.Int64, "col2": pl.Int64})

        # Create expected columns for single operation tests
        df = df.with_columns(
            add_column=(pl.col("col1") + pl.col("col2")),
            sub_column=(pl.col("col1") - pl.col("col2")),
            mul_column=(pl.col("col1") * pl.col("col2")),
            div_column=(pl.col("col1") / pl.col("col2")),
            pow_column=(pl.col("col1") ** pl.col("col2")),
            add_constant=(pl.col("col1") + pl.lit(self._CONST)),
            sub_constant=(pl.col("col1") - pl.lit(self._CONST)),
            mul_constant=(pl.col("col1") * pl.lit(self._CONST)),
            div_constant=(pl.col("col1") / pl.lit(self._CONST)),
            pow_constant=(pl.col("col1") ** pl.lit(self._CONST)),
        )
        return df

    @pytest.mark.parametrize(
        "strategy",
        [
            # Both constant and column at the same time (invalid)
            {
                "new_column_name": "col3",
                "strategy": [
                    {"constant": 30, "column": "col1"},
                    {"column": "col2"}
                ],
                "operations": ["sub"],
            },
            # Both constant and column in second operand
            [
                {
                    "new_column_name": "col3",
                    "strategy": [
                        {"constant": 30},
                        {"column": "col1", "cast": "float", "constant": "22"},
                    ],
                    "operations": ["sub"],
                }
            ],
            # Mismatched lengths: 2 operands but 2 operations (should be 1)
            {
                "new_column_name": "col3",
                "strategy": [{"constant": 30}, {"column": "col1"}],
                "operations": ["sub", "sub"],
            },
            # Invalid operation name
            [
                {
                    "new_column_name": "col3",
                    "strategy": [
                        {"constant": 30, "cast": "integer"},
                        {"column": "col1"}
                    ],
                    "operations": ["not_found"],
                }
            ],
        ],
    )
    def test_wrong_strategy(self, df_input, strategy):
        """Test MathOperator with invalid strategy configurations."""
        t = MathOperator(strategy=strategy)
        with pytest.raises(ValueError):
            t.transform(df_input)

    def test_wrong_strategy_type(self):
        """Test MathOperator with wrong strategy type."""
        with pytest.raises(TypeError):
            MathOperator(strategy=10)

    def test_unknown_type_cast(self, df_input):
        """Test MathOperator with unknown type in cast."""
        strategy = {
            "new_column_name": "result",
            "strategy": [
                {"column": "col1", "cast": "unknown_type"},
                {"column": "col2"}
            ],
            "operations": ["add"],
        }
        t = MathOperator(strategy=strategy)
        with pytest.raises(ValueError):
            t.transform(df_input)

    @pytest.mark.parametrize("col_or_const", ["column", "constant"])
    @pytest.mark.parametrize("cast", [None, "float32"])
    @pytest.mark.parametrize("operation", ["add", "sub", "mul", "div", "pow"])
    def test_single_operation(
            self, df_input, operation: str, col_or_const: str, cast
    ):
        """Test MathOperator with single operation using column or constant."""
        second = (
            {"column": "col2"}
            if col_or_const == "column"
            else {"constant": self._CONST}
        )
        strategy = {
            "new_column_name": "result",
            "strategy": [{"column": "col1"}, second],
            "operations": [operation],
        }
        if cast:
            strategy["cast"] = cast

        t = MathOperator(strategy=strategy)
        df_result = t.transform(df_input).select("col1", "col2", "result")

        # Build expected dataframe
        expected_col = f"{operation}_{col_or_const}"
        df_expected = df_input.select(
            "col1",
            "col2",
            pl.col(expected_col).alias("result")
        )
        if cast:
            df_expected = df_expected.with_columns(
                pl.col("result").cast(pl.Float32)
            )

        pl.testing.assert_frame_equal(df_result, df_expected)

    @pytest.mark.parametrize(
        "operations",
        [["add", "div"], ["mul", "div"], ["pow", "sub"]]
    )
    def test_double_operation(self, df_input, operations: list[str]):
        """Test MathOperator with two sequential operations."""
        strategy = {
            "new_column_name": "chk",
            "strategy": [
                {"column": "col1", "cast": "float"},
                {"constant": self._CONST},
                {"column": "col2"},
            ],
            "operations": operations,
        }
        t = MathOperator(strategy=strategy)
        df_result = t.transform(df_input).select("col1", "col2", "chk")

        # Build expected result using Python operators
        operators_map: dict = {
            "add": py_operator.add,
            "sub": py_operator.sub,
            "mul": py_operator.mul,
            "div": py_operator.truediv,
            "pow": py_operator.pow,
        }

        op_0 = operators_map[operations[0]]
        op_1 = operators_map[operations[1]]

        # Build expression: (col1.cast(float) op_0 11.0) op_1 col2
        expr_0 = op_0(pl.col("col1").cast(pl.Float64), pl.lit(self._CONST))
        expr_1 = op_1(expr_0, pl.col("col2"))

        df_expected = df_input.select("col1", "col2").with_columns(
            expr_1.alias("chk")
        )

        pl.testing.assert_frame_equal(df_result, df_expected)

    def test_multiple_columns_at_once(self, df_input):
        """Test creating multiple columns in a single transform."""
        strategy = [
            {
                "new_column_name": "sum_cols",
                "strategy": [{"column": "col1"}, {"column": "col2"}],
                "operations": ["add"],
            },
            {
                "new_column_name": "product_cols",
                "strategy": [{"column": "col1"}, {"column": "col2"}],
                "operations": ["mul"],
            },
        ]
        t = MathOperator(strategy=strategy)
        df_result = t.transform(df_input)

        assert "sum_cols" in df_result.columns
        assert "product_cols" in df_result.columns

        # Verify calculations
        df_expected = df_input.with_columns(
            sum_cols=(pl.col("col1") + pl.col("col2")),
            product_cols=(pl.col("col1") * pl.col("col2")),
        )

        pl.testing.assert_frame_equal(
            df_result.select("sum_cols", "product_cols"),
            df_expected.select("sum_cols", "product_cols")
        )

    def test_chained_columns(self, df_input):
        """Test using a newly created column in subsequent calculation."""
        # Note: This would require two separate transform calls
        # since MathOperator processes all strategies in parallel
        strategy_1 = {
            "new_column_name": "doubled",
            "strategy": [{"column": "col1"}, {"constant": 2}],
            "operations": ["mul"],
        }
        t1 = MathOperator(strategy=strategy_1)
        df_temp = t1.transform(df_input)

        strategy_2 = {
            "new_column_name": "tripled",
            "strategy": [{"column": "doubled"}, {"constant": 1.5}],
            "operations": ["mul"],
        }
        t2 = MathOperator(strategy=strategy_2)
        df_result = t2.transform(df_temp)

        # doubled = col1 * 2, tripled = doubled * 1.5 = col1 * 3
        df_expected = df_input.with_columns(
            tripled=(pl.col("col1") * 3.0)
        )

        pl.testing.assert_frame_equal(
            df_result.select("tripled"),
            df_expected.select("tripled")
        )

    def test_complex_expression(self, df_input):
        """Test complex multi-operation expression."""
        # ((col1 + 10) * col2) / 3
        strategy = {
            "new_column_name": "complex_result",
            "cast": "float",
            "strategy": [
                {"column": "col1"},
                {"constant": 10},
                {"column": "col2"},
                {"constant": 3},
            ],
            "operations": ["add", "mul", "div"],
        }
        t = MathOperator(strategy=strategy)
        df_result = t.transform(df_input).select("complex_result")

        # Build expected
        df_expected = df_input.with_columns(
            complex_result=(
                    ((pl.col("col1") + 10) * pl.col("col2")) / 3
            ).cast(pl.Float64)
        ).select("complex_result")

        pl.testing.assert_frame_equal(df_result, df_expected)

    def test_all_operations_together(self, df_input):
        """Test all supported operations in one complex expression."""
        # (((col1 + 2) - 1) * 3) / 2
        strategy = {
            "new_column_name": "all_ops",
            "strategy": [
                {"column": "col1"},
                {"constant": 2},
                {"constant": 1},
                {"constant": 3},
                {"constant": 2},
            ],
            "operations": ["add", "sub", "mul", "div"],
        }
        t = MathOperator(strategy=strategy)
        df_result = t.transform(df_input).select("all_ops")

        # Build expected manually
        df_expected = df_input.with_columns(
            all_ops=(((pl.col("col1") + 2) - 1) * 3) / 2
        ).select("all_ops")

        pl.testing.assert_frame_equal(df_result, df_expected)

    def test_cast_operands(self, df_input):
        """Test casting individual operands before operations."""
        strategy = {
            "new_column_name": "casted_result",
            "strategy": [
                {"column": "col1", "cast": "float64"},
                {"constant": 10, "cast": "float32"},
            ],
            "operations": ["div"],
        }
        t = MathOperator(strategy=strategy)
        df_result = t.transform(df_input).select("casted_result")

        # Build expected
        df_expected = df_input.with_columns(
            casted_result=pl.col("col1").cast(pl.Float64) / pl.lit(10).cast(pl.Float32)
        ).select("casted_result")

        pl.testing.assert_frame_equal(df_result, df_expected)

    def test_final_cast(self, df_input):
        """Test final cast of result column."""
        strategy = {
            "new_column_name": "result",
            "cast": "int32",
            "strategy": [
                {"column": "col1"},
                {"constant": 2.5},
            ],
            "operations": ["mul"],
        }
        t = MathOperator(strategy=strategy)
        df_result = t.transform(df_input).select("result")

        # Build expected
        df_expected = df_input.with_columns(
            result=(pl.col("col1") * 2.5).cast(pl.Int32)
        ).select("result")

        pl.testing.assert_frame_equal(df_result, df_expected)

    def test_case_insensitive_types(self, df_input):
        """Test that type names are case-insensitive."""
        strategies = [
            {"new_column_name": "r1", "cast": "INT64", "strategy": [{"column": "col1"}], "operations": []},
            {"new_column_name": "r2", "cast": "Float", "strategy": [{"column": "col1"}], "operations": []},
            {"new_column_name": "r3", "cast": "STRING", "strategy": [{"column": "col1"}], "operations": []},
        ]

        for strategy in strategies:
            t = MathOperator(strategy=strategy)
            df_result = t.transform(df_input)
            assert strategy["new_column_name"] in df_result.columns


class TestPivot:
    """Test suite for Pivot transformer."""

    @staticmethod
    @pytest.fixture
    def long_df():
        """Long format DataFrame with sales data."""
        data = {
            "product_id": [1, 1, 1, 2, 2, 2],
            "month": ["jan", "feb", "mar", "jan", "feb", "mar"],
            "revenue": [100, 110, 120, 200, 190, 210],
        }
        return nw.from_dict(data, backend="polars")

    def test_pivot_basic(self, long_df):
        """Test basic pivot with single value column."""
        transformer = Pivot(
            pivot_col="month",
            id_cols="product_id",
            aggregate_function="first",
        )
        result = transformer.transform(long_df)

        assert result.shape == (2, 4)
        assert "product_id" in result.columns
        assert "jan" in result.columns
        assert "feb" in result.columns
        assert "mar" in result.columns

        product_1 = result.filter(nw.col("product_id") == 1)
        assert product_1["jan"].item() == 100
        assert product_1["feb"].item() == 110
        assert product_1["mar"].item() == 120

    def test_pivot_with_id_regex(self):
        """Test pivot with regex for id columns."""
        data = {
            "id_user": [1, 1, 2, 2],
            "id_session": [101, 101, 102, 102],
            "metric_type": ["A", "B", "A", "B"],
            "value": [10, 20, 30, 40],
        }
        df = nw.from_dict(data, backend="polars")

        transformer = Pivot(
            pivot_col="metric_type",
            id_regex="^id_.*",
            aggregate_function="first",
            values_cols=["value"]
        )
        result = transformer.transform(df)

        assert result.shape == (2, 4)
        assert "id_user" in result.columns
        assert "id_session" in result.columns
        assert "A" in result.columns
        assert "B" in result.columns

    def test_pivot_with_id_glob(self):
        """Test pivot with glob pattern for id columns."""
        data = {
            "user_id": [1, 1, 2, 2],
            "user_name": ["Alice", "Alice", "Bob", "Bob"],
            "category": ["X", "Y", "X", "Y"],
            "score": [10, 20, 30, 40],
        }
        df = nw.from_dict(data, backend="polars")

        transformer = Pivot(
            pivot_col="category",
            id_glob="user_*",
            aggregate_function="first",
        )
        result = transformer.transform(df)

        assert result.shape == (2, 4)
        assert "user_id" in result.columns
        assert "user_name" in result.columns

    def test_pivot_with_id_startswith(self):
        """Test pivot with startswith selector for id columns."""
        data = {
            "pk_id": [1, 1, 2, 2],
            "pk_timestamp": [100, 100, 200, 200],
            "other_col": ["foo", "foo", "bar", "bar"],
            "type": ["A", "B", "A", "B"],
            "val": [10, 20, 30, 40],
        }
        df = nw.from_dict(data, backend="polars")

        transformer = Pivot(
            pivot_col="type",
            id_startswith="pk_",
            aggregate_function="first",
        )
        result = transformer.transform(df)

        assert "pk_id" in result.columns
        assert "pk_timestamp" in result.columns
        assert "other_col" not in result.columns

    def test_pivot_with_id_endswith(self):
        """Test pivot with endswith selector for id columns."""
        data = {
            "user_key": [1, 1, 2, 2],
            "session_key": [101, 101, 102, 102],
            "random": ["x", "x", "y", "y"],
            "metric": ["A", "B", "A", "B"],
            "value": [10, 20, 30, 40],
        }
        df = nw.from_dict(data, backend="polars")

        transformer = Pivot(
            pivot_col="metric",
            id_endswith="_key",
            aggregate_function="first",
        )
        result = transformer.transform(df)

        assert "user_key" in result.columns
        assert "session_key" in result.columns
        assert "random" not in result.columns

    def test_pivot_with_values_regex(self):
        """Test pivot with regex selection for value columns."""
        data = {
            "id": [1, 1, 2, 2],
            "type": ["X", "Y", "X", "Y"],
            "metric_1": [10, 20, 30, 40],
            "metric_2": [11, 21, 31, 41],
            "other": [99, 99, 99, 99],
        }
        df = nw.from_dict(data, backend="polars")

        transformer = Pivot(
            pivot_col="type",
            id_cols="id",
            values_regex="^metric_.*",
            aggregate_function="first",
        )
        result = transformer.transform(df)

        assert "other_X" not in result.columns
        assert "other_Y" not in result.columns
        assert "metric_1_X" in result.columns
        assert "metric_2_Y" in result.columns

    def test_pivot_with_values_glob(self):
        """Test pivot with glob pattern for value columns."""
        data = {
            "id": [1, 1, 2, 2],
            "category": ["A", "B", "A", "B"],
            "sales_total": [100, 200, 150, 250],
            "sales_count": [10, 20, 15, 25],
            "cost": [50, 100, 75, 125],
        }
        df = nw.from_dict(data, backend="polars")

        transformer = Pivot(
            pivot_col="category",
            id_cols="id",
            values_glob="sales_*",
            aggregate_function="first",
        )
        result = transformer.transform(df)

        assert "sales_total_A" in result.columns
        assert "sales_count_B" in result.columns
        assert "cost_A" not in result.columns

    def test_pivot_with_values_startswith(self):
        """Test pivot with startswith selector for value columns."""
        data = {
            "id": [1, 1, 2, 2],
            "type": ["X", "Y", "X", "Y"],
            "revenue_usd": [100, 200, 150, 250],
            "revenue_eur": [90, 180, 135, 225],
            "cost": [50, 100, 75, 125],
        }
        df = nw.from_dict(data, backend="polars")

        transformer = Pivot(
            pivot_col="type",
            id_cols="id",
            values_startswith="revenue",
            aggregate_function="first",
        )
        result = transformer.transform(df)

        assert "revenue_usd_X" in result.columns
        assert "revenue_eur_Y" in result.columns
        assert "cost_X" not in result.columns

    def test_pivot_with_values_endswith(self):
        """Test pivot with endswith selector for value columns."""
        data = {
            "id": [1, 1, 2, 2],
            "quarter": ["Q1", "Q2", "Q1", "Q2"],
            "total_sales": [100, 200, 150, 250],
            "total_units": [10, 20, 15, 25],
            "average": [10, 10, 10, 10],
        }
        df = nw.from_dict(data, backend="polars")

        transformer = Pivot(
            pivot_col="quarter",
            id_cols="id",
            values_endswith="_sales",
            aggregate_function="first",
        )
        result = transformer.transform(df)

        assert "Q1" in result.columns
        assert "Q2" in result.columns

    def test_pivot_multiple_selectors_combined(self):
        """Test pivot with multiple selector types combined."""
        data = {
            "pk_user": [1, 1, 2, 2],
            "pk_session": [101, 101, 102, 102],
            "dimension": ["A", "B", "A", "B"],
            "metric_revenue": [100, 200, 150, 250],
            "metric_cost": [50, 100, 75, 125],
            "other": [99, 99, 99, 99],
        }
        df = nw.from_dict(data, backend="polars")

        transformer = Pivot(
            pivot_col="dimension",
            id_startswith="pk_",  # Select pk_user, pk_session
            values_startswith="metric",  # Select metric_revenue, metric_cost
            aggregate_function="first",
        )
        result = transformer.transform(df)

        # ID columns
        assert "pk_user" in result.columns
        assert "pk_session" in result.columns

        # Value columns pivoted
        assert "metric_revenue_A" in result.columns
        assert "metric_cost_B" in result.columns

        # Excluded columns
        assert "other_A" not in result.columns

    def test_pivot_requires_id_selector(self, long_df):
        """Test that at least one id selector must be provided."""
        with pytest.raises(AssertionError):
            Pivot(
                pivot_col="month",
                aggregate_function="first",
            )

    def test_pivot_with_sum_aggregation(self):
        """Test pivot with sum aggregation for duplicates."""
        data = {
            "store": ["A", "A", "A", "B", "B"],
            "product": ["X", "X", "Y", "X", "Y"],
            "sales": [10, 15, 20, 30, 40],
        }
        df = nw.from_dict(data, backend="polars")

        transformer = Pivot(
            pivot_col="product",
            id_cols="store",
            aggregate_function="sum",
        )
        result = transformer.transform(df)

        store_a = result.filter(nw.col("store") == "A")
        assert store_a["X"].item() == 25
        assert store_a["Y"].item() == 20

    def test_pivot_without_values_specified(self, long_df):
        """Test pivot without specifying values (pivots all non-id columns)."""
        transformer = Pivot(
            pivot_col="month",
            id_cols="product_id",
            aggregate_function="first",
        )
        result = transformer.transform(long_df)

        # Should pivot the revenue column automatically
        assert "jan" in result.columns
        assert "feb" in result.columns
        assert "mar" in result.columns


class TestUnpivot:
    """Test suite for Unpivot transformer."""

    @pytest.fixture
    def wide_df(self):
        """Wide format DataFrame with sales data."""
        data = {
            "product_id": [1, 2, 3],
            "category": ["A", "B", "A"],
            "sales_jan": [100, 200, 150],
            "sales_feb": [110, 190, 160],
            "sales_mar": [120, 210, 170],
        }
        return nw.from_native(pl.DataFrame(data))

    def test_with_id_cols_and_melt_cols(self, wide_df):
        """Test basic melt with explicit columns."""
        transformer = Unpivot(
            id_cols=["product_id", "category"],
            melt_cols=["sales_jan", "sales_feb", "sales_mar"],
            variable_col="month",
            value_col="revenue",
        )
        result = transformer.transform(wide_df)

        assert result.shape == (9, 4)  # 3 products × 3 months
        assert set(result.columns) == {"product_id", "category", "month", "revenue"}

        # Check first product's data
        first_product = result.filter(nw.col("product_id") == 1).sort("month")
        months = first_product["month"].to_list()
        revenues = first_product["revenue"].to_list()

        assert months == ["sales_feb", "sales_jan", "sales_mar"]
        assert revenues == [110, 100, 120]

    def test_with_regex(self, wide_df):
        """Test melt using regex patterns."""
        transformer = Unpivot(
            id_cols=["product_id"],
            melt_regex="^sales_.*",
            variable_col="month",
            value_col="amount",
        )
        result = transformer.transform(wide_df)

        # category is NOT in id_cols, so it gets dropped entirely
        assert result.shape == (9, 3)
        assert set(result.columns) == {"product_id", "month", "amount"}

        # Verify we have all 3 months for each product
        product_1 = result.filter(nw.col("product_id") == 1)
        assert product_1.shape[0] == 3

    def test_with_id_regex(self):
        """Test melt with regex for id columns."""
        data = {
            "id_user": [1, 2],
            "id_session": [101, 102],
            "metric_a": [10, 20],
            "metric_b": [30, 40],
        }
        df = nw.from_native(pl.DataFrame(data))

        transformer = Unpivot(
            id_regex="^id_.*",
            melt_regex="^metric_.*",
            variable_col="metric_name",
            value_col="metric_value",
        )
        result = transformer.transform(df)

        assert result.shape == (4, 4)  # 2 rows × 2 metrics
        assert set(result.columns) == {"id_user", "id_session", "metric_name", "metric_value"}

    def test_without_id_cols(self):
        """Test melt discards non-melt columns when no id_cols specified."""
        data = {
            "name": ["Alice", "Bob"],
            "age": [25, 30],
            "score_math": [85, 90],
            "score_english": [88, 92],
        }
        df = nw.from_native(pl.DataFrame(data))

        transformer = Unpivot(
            melt_cols=["score_math", "score_english"],
            variable_col="subject",
            value_col="score",
        )
        result = transformer.transform(df)

        # name and age should be discarded
        assert result.shape == (4, 2)
        assert set(result.columns) == {"subject", "score"}

    def test_single_column(self):
        """Test melt with single melt column."""
        data = {
            "id": [1, 2],
            "value": [10, 20],
        }
        df = nw.from_native(pl.DataFrame(data))

        transformer = Unpivot(
            id_cols="id",
            melt_cols="value",
            variable_col="var",
            value_col="val",
        )
        result = transformer.transform(df)

        assert result.shape == (2, 3)
        assert result["var"].to_list() == ["value", "value"]

    def test_preserves_id_values(self, wide_df):
        """Test that id column values are preserved correctly."""
        transformer = Unpivot(
            id_cols="product_id",
            melt_cols=["sales_jan", "sales_feb"],
            variable_col="month",
            value_col="sales",
        )
        result = transformer.transform(wide_df)

        # Each product_id should appear twice (once per month)
        product_counts = result.group_by("product_id").agg(nw.len()).sort("product_id")
        counts = product_counts["len"].to_list()
        assert counts == [2, 2, 2]

    def test_requires_melt_specification(self, wide_df):
        """Test that either melt_cols or melt_regex must be provided."""
        with pytest.raises(AssertionError):
            Unpivot(
                id_cols=["product_id"],
                variable_col="var",
                value_col="val",
            )

    def test_empty_selection(self):
        """Test melt when regex matches no columns."""
        data = {"id": [1, 2], "value": [10, 20]}
        df = nw.from_native(pl.DataFrame(data))

        transformer = Unpivot(
            id_cols="id",
            melt_regex="^nonexistent_.*",
            variable_col="var",
            value_col="val",
        )

        # Should handle gracefully - unpivot with empty index
        result = transformer.transform(df)
        assert "id" in result.columns

    def test_with_nulls(self):
        """Test melt preserves null values."""
        data = {
            "id": [1, 2],
            "sales_jan": [100, None],
            "sales_feb": [None, 200],
        }
        df = nw.from_native(pl.DataFrame(data))

        transformer = Unpivot(
            id_cols="id",
            melt_cols=["sales_jan", "sales_feb"],
            variable_col="month",
            value_col="sales",
        )
        result = transformer.transform(df)

        assert result.shape == (4, 3)
        # Nulls should be preserved in the melted column
        nulls = result.filter(nw.col("sales").is_null())
        assert nulls.shape[0] == 2


class TestFilterValidation:
    """Test Filter initialization validation."""

    def test_double_negation_is_not_in_rejected(self):
        """Test that perform='remove' + is_not_in raises error."""
        with pytest.raises(ValueError, match="double negation"):
            Filter(
                input_col="status",
                perform="remove",
                operator="is_not_in",
                value=["active"],
            )

    def test_double_negation_is_not_null_rejected(self):
        """Test that perform='remove' + is_not_null raises error."""
        with pytest.raises(ValueError, match="double negation"):
            Filter(
                input_col="age",
                perform="remove",
                operator="is_not_null",
            )

    def test_double_negation_is_not_nan_rejected(self):
        """Test that perform='remove' + is_not_nan raises error."""
        with pytest.raises(ValueError, match="double negation"):
            Filter(
                input_col="score",
                perform="remove",
                operator="is_not_nan",
            )

    def test_invalid_perform_value_rejected(self):
        """Test that invalid perform values are rejected."""
        with pytest.raises((ValueError, AssertionError)):
            Filter(
                input_col="age",
                perform="maybe",  # Invalid
                operator="gt",
                value=18,
            )


class TestFilter:
    """Test Filter transformer."""

    # -------------- perform="keep" --------------

    @pytest.fixture(scope="class")
    def df(self):
        """Create a DataFrame with nulls and NaNs for testing."""
        return nw.from_native(
            pd.DataFrame({
                "age": [15, 25, 35, None],
                "score": [50.0, 75.0, float("nan"), 90.0],
                "name": ["Alice", "Bob", "Charlie", "Dave"],
                "status": ["active", "pending", "active", None],
            })
        )

    def test_keep_with_eq(self, df):
        """Test keep with equality - nulls are excluded."""
        t = Filter(input_col="status", perform="keep", operator="eq", value="active")
        result = t.transform(df)

        result_native = nw.to_native(result)
        assert len(result_native) == 2  # Alice and Charlie
        assert set(result_native["name"]) == {"Alice", "Charlie"}

    def test_keep_with_ne(self, df):
        """Test keep with not-equal - nulls are excluded."""
        t = Filter(input_col="status", perform="keep", operator="ne", value="active")
        result = t.transform(df)

        result_native = nw.to_native(result)
        # "pending" only - Dave's null is excluded (null != "active" → null → excluded)
        assert len(result_native) == 2
        assert result_native["name"].tolist() == ["Bob", "Dave"]

    def test_keep_with_gt_excludes_nulls(self, df):
        """Test that comparisons exclude nulls (null > 20 → null → excluded)."""
        t = Filter(input_col="age", perform="keep", operator="gt", value=20)
        result = t.transform(df)

        result_native = nw.to_native(result)
        # Bob (25) and Charlie (35) - Dave's null excluded
        assert len(result_native) == 2
        assert set(result_native["name"]) == {"Bob", "Charlie"}

    def test_keep_with_gt_excludes_nans(self, df):
        """Test that comparisons exclude NaNs (NaN > 60 → False)."""
        t = Filter(input_col="score", perform="keep", operator="gt", value=60)
        result = t.transform(df)

        result_native = nw.to_native(result)
        # Dave (90) + Charlie's NaN
        assert len(result_native) == 2
        assert result_native["name"].tolist() == ["Bob", "Dave"]

    def test_keep_with_is_null(self, df):
        """Test explicit null checking."""
        t = Filter(input_col="age", perform="keep", operator="is_null")
        result = t.transform(df)

        result_native = nw.to_native(result)
        assert len(result_native) == 1
        assert result_native["name"].iloc[0] == "Dave"

    def test_keep_with_is_not_null(self, df):
        """Test explicit not-null checking."""
        t = Filter(input_col="age", perform="keep", operator="is_not_null")
        result = t.transform(df)

        result_native = nw.to_native(result)
        assert len(result_native) == 3  # Alice, Bob, Charlie

    def test_keep_with_is_nan(self, df):
        """Test explicit NaN checking (distinct from null)."""
        t = Filter(input_col="score", perform="keep", operator="is_nan")
        result = t.transform(df)

        result_native = nw.to_native(result)
        assert len(result_native) == 1
        assert result_native["name"].iloc[0] == "Charlie"

    def test_keep_with_is_not_nan(self, df):
        """Test explicit not-NaN checking."""
        t = Filter(input_col="score", perform="keep", operator="is_not_nan")
        result = t.transform(df)

        result_native = nw.to_native(result)
        # Alice (50), Bob (75), Dave (90) - Charlie's NaN excluded
        assert len(result_native) == 3
        assert set(result_native["name"]) == {"Alice", "Bob", "Dave"}

    def test_keep_with_is_in(self, df):
        """Test set membership - nulls excluded."""
        t = Filter(
            input_col="status",
            perform="keep",
            operator="is_in",
            value=["active", "pending"],
        )
        result = t.transform(df)

        result_native = nw.to_native(result)
        # Alice (active), Bob (pending), Charlie (active) - Dave's null excluded
        assert len(result_native) == 3
        assert set(result_native["name"]) == {"Alice", "Bob", "Charlie"}

    # -------------- perform="remove" --------------

    def test_remove_with_eq_keeps_nulls(self, df):
        """Test remove with equality - nulls are KEPT (not removed)."""
        t = Filter(input_col="status", perform="remove", operator="eq", value="active")
        result = t.transform(df)

        result_native = nw.to_native(result)
        # Bob (pending) and Dave (null) - null != "active" so it's kept
        assert len(result_native) == 2
        assert set(result_native["name"]) == {"Bob", "Dave"}

    def test_remove_with_gt_keeps_nulls(self, df):
        """Test that remove with comparisons keeps nulls."""
        t = Filter(input_col="age", perform="remove", operator="gt", value=20)
        result = t.transform(df)

        result_native = nw.to_native(result)
        # Remove: Bob (25 > 20) and Charlie (35 > 20)
        # Keep: Alice (15 not > 20) and Dave (null, comparison is null → not removed)
        assert len(result_native) == 2
        assert set(result_native["name"]) == {"Alice", "Dave"}

    def test_remove_with_is_in_keeps_nulls(self, df):
        """Test that remove with is_in keeps nulls (important!)."""
        t = Filter(
            input_col="status",
            perform="remove",
            operator="is_in",
            value=["active"],
        )
        result = t.transform(df)

        result_native = nw.to_native(result)
        # Remove: Alice and Charlie (active is in [active])
        # Keep: Bob (pending not in [active]) and Dave (null not in [active])
        assert len(result_native) == 2
        assert set(result_native["name"]) == {"Bob", "Dave"}

    def test_remove_vs_keep_with_is_in_are_different(self, df):
        """Demonstrate that perform='remove' + is_in handles nulls differently than perform='keep' + is_not_in would."""
        # Remove rows where status is "active"
        t_remove = Filter(
            input_col="status",
            perform="remove",
            operator="is_in",
            value=["active"],
        )
        result_remove = t_remove.transform(df)

        # If we could use keep + is_not_in (which we banned), nulls would be excluded
        # But with remove + is_in, nulls are kept
        result_native = nw.to_native(result_remove)
        assert len(result_native) == 2
        # Critically: Dave (null) is KEPT
        assert "Dave" in set(result_native["name"])

    def test_remove_with_is_null_removes_nulls(self, df):
        """Test that remove + is_null removes null rows."""
        t = Filter(input_col="status", perform="remove", operator="is_null")
        result = t.transform(df)

        result_native = nw.to_native(result)
        # Remove Dave (null) - keep everyone else
        assert len(result_native) == 3
        assert set(result_native["name"]) == {"Alice", "Bob", "Charlie"}

    def test_remove_with_is_nan_removes_nans(self, df):
        """Test that remove + is_nan removes NaN rows."""
        t = Filter(input_col="score", perform="remove", operator="is_nan")
        result = t.transform(df)

        result_native = nw.to_native(result)
        # Remove Charlie (NaN) - keep everyone else
        assert len(result_native) == 3
        assert set(result_native["name"]) == {"Alice", "Bob", "Dave"}

    # -------------- test null and nan --------------

    @pytest.fixture(scope="class")
    def df_null_nan(self):
        """Create a DataFrame that clearly separates null and NaN."""
        return nw.from_native(
            pl.DataFrame({
                "value": [1.0, None, float("nan"), 4.0],
                "label": ["one", "null", "nan", "four"],
            })
        )

    def test_is_null_finds_only_null(self, df_null_nan):
        """Test that is_null finds None but not NaN."""
        t = Filter(input_col="value", perform="keep", operator="is_null")
        result = t.transform(df_null_nan)

        result_native = nw.to_native(result)
        assert len(result_native) == 1
        assert result_native["label"][0] == "null"

    def test_is_nan_finds_only_nan(self, df_null_nan):
        """Test that is_nan finds NaN but not None."""
        t = Filter(input_col="value", perform="keep", operator="is_nan")
        result = t.transform(df_null_nan)

        result_native = nw.to_native(result)
        assert len(result_native) == 1
        assert result_native["label"][0] == "nan"

    def test_is_not_null_excludes_only_null(self, df_null_nan):
        """Test that is_not_null keeps NaN but excludes None."""
        t = Filter(input_col="value", perform="keep", operator="is_not_null")
        result = t.transform(df_null_nan)

        result_native = nw.to_native(result)
        assert len(result_native) == 3  # one, nan, four
        assert set(result_native["label"]) == {"one", "nan", "four"}

    def test_is_not_nan_excludes_only_nan(self, df_null_nan):
        """Test that is_not_nan keeps None but excludes NaN."""
        t = Filter(input_col="value", perform="keep", operator="is_not_nan")
        result = t.transform(df_null_nan)

        result_native = nw.to_native(result)
        assert len(result_native) == 3  # one, null, four
        set_results = set(result_native["label"].to_list())
        assert set_results == {"one", "null", "four"}

    # -------------- test comparison column --------------

    @pytest.fixture(scope="class")
    def df_compare_col(self):
        """Create a DataFrame for column comparison tests."""
        return nw.from_native(
            pd.DataFrame({
                "sales": [100, 200, None, 400],
                "target": [150, 150, 150, 150],
                "name": ["Alice", "Bob", "Charlie", "Dave"],
            })
        )

    def test_keep_with_column_comparison(self, df_compare_col):
        """Test that column comparisons work with perform='keep'."""
        t = Filter(
            input_col="sales",
            perform="keep",
            operator="gt",
            compare_col="target",
        )
        result = t.transform(df_compare_col)

        result_native = nw.to_native(result)
        # Bob (200 > 150) and Dave (400 > 150)
        # Charlie's null is excluded (null > 150 → null → excluded)
        assert len(result_native) == 2
        assert set(result_native["name"]) == {"Bob", "Dave"}

    def test_remove_with_column_comparison_keeps_nulls(self, df_compare_col):
        """Test that remove with column comparison keeps nulls."""
        t = Filter(
            input_col="sales",
            perform="remove",
            operator="gt",
            compare_col="target",
        )
        result = t.transform(df_compare_col)

        result_native = nw.to_native(result)
        # Remove: Bob (200 > 150) and Dave (400 > 150)
        # Keep: Alice (100 not > 150) and Charlie (null, not removed)
        assert len(result_native) == 2
        assert set(result_native["name"]) == {"Alice", "Charlie"}

    # -------------- string operations --------------

    @pytest.fixture(scope="class")
    def df_strings(self):
        """Create a DataFrame with strings and nulls."""
        return nw.from_native(
            pd.DataFrame({
                "email": [
                    "alice@company.com",
                    "bob@external.org",
                    None,
                    "charlie@company.com",
                ],
                "name": ["Alice", "Bob", "Charlie", "Dave"],
            })
        )

    def test_keep_contains(self, df_strings):
        """Test string contains with keep."""
        t = Filter(
            input_col="email",
            perform="keep",
            operator="contains",
            value="company",
        )
        result = t.transform(df_strings.filter(~nw.col("email").is_null()))

        result_native = nw.to_native(result)
        assert len(result_native) == 2
        assert set(result_native["name"]) == {"Alice", "Dave"}

    def test_remove_contains_keeps_nulls(self, df_strings):
        """Test that remove + contains keeps null values."""
        t = Filter(
            input_col="email",
            perform="remove",
            operator="contains",
            value="external",
        )
        result = t.transform(df_strings)

        result_native = nw.to_native(result)
        # Remove: Bob (@external)
        # Keep: Alice, Charlie (null), Dave
        assert len(result_native) == 3
        assert set(result_native["name"]) == {"Alice", "Charlie", "Dave"}

    def test_starts_with(self, df_strings):
        """Test string starts_with."""
        t = Filter(
            input_col="email",
            perform="keep",
            operator="starts_with",
            value="alice",
        )
        result = t.transform(df_strings.filter(~nw.col("email").is_null()))

        result_native = nw.to_native(result)
        assert len(result_native) == 1
        assert result_native["name"].iloc[0] == "Alice"


class TestWhen:
    """Test When transformer."""

    # ------------- test init validations -------------

    def test_requires_output_in_condition(self):
        """Test that conditions must specify an output."""
        with pytest.raises(ValueError, match="must specify either"):
            When(
                output_col="result",
                conditions=[
                    {"input_col": "value", "operator": "gt", "value": 5},
                    # Missing output_constant or output_col
                ],
                otherwise_constant="default",
            )

    def test_requires_otherwise(self):
        """Test that otherwise clause is required."""
        with pytest.raises(ValueError, match="Must specify either"):
            When(
                output_col="result",
                conditions=[
                    {"input_col": "value", "operator": "gt", "value": 5, "output_constant": "big"},
                ],
                # Missing otherwise
            )

    def test_validates_operators(self):
        """Test that invalid operators are rejected."""
        with pytest.raises((ValueError, AssertionError)):
            When(
                output_col="result",
                conditions=[
                    {"input_col": "value", "operator": "invalid_op", "value": 5, "output_constant": "x"},
                ],
                otherwise_constant="default",
            )

    # ------------- basic tests -------------

    @pytest.fixture(scope="class")
    def df_basic(self):
        return nw.from_native(
            pl.DataFrame({
                "value": [1, 2, None, 4, 5],
                "name": ["a", "b", "c", "d", "e"],
                "flag": [True, False, True, False, False],
                "col_a": [10, 20, 30, 40, 50],
                "col_b": [100, 200, 300, 400, 500],
            })
        )

    def test_first_match_wins(self, df_basic):
        """Test that first matching condition is used."""
        t = When(
            output_col="result",
            conditions=[
                {"input_col": "value", "operator": "lt", "value": 3, "output_constant": "low"},
                {"input_col": "value", "operator": "lt", "value": 5, "output_constant": "medium"},
            ],
            otherwise_constant="high",
        )
        result = t.transform(df_basic)

        result_native = nw.to_native(result)
        # the high in the middle is due to a null value
        expected = ["low", "low", "high", "medium", "high"]
        assert list(result_native["result"]) == expected

    def test_otherwise_constant(self, df_basic):
        """Test otherwise with constant value."""
        t = When(
            output_col="result",
            conditions=[
                {"input_col": "value", "operator": "eq", "value": 999, "output_constant": "found"},
            ],
            otherwise_constant="not_found",
        )
        result = t.transform(df_basic)

        result_native = nw.to_native(result)
        assert all(result_native["result"] == "not_found")

    def test_otherwise_column(self, df_basic):
        """Test otherwise with column value."""
        t = When(
            output_col="result",
            conditions=[
                {"input_col": "value", "operator": "gt", "value": 10, "output_constant": "big"},
            ],
            otherwise_col="name",
        )
        result = t.transform(df_basic)

        result_native = nw.to_native(result)
        assert list(result_native["result"]) == ["a", "b", "c", "d", "e"]

    def test_output_constant(self, df_basic):
        """Test using constant values as output."""
        t = When(
            output_col="result",
            conditions=[
                {"input_col": "flag", "operator": "eq", "value": True, "output_constant": 1},
            ],
            otherwise_constant=0,
        )
        result = t.transform(df_basic)

        result_native = nw.to_native(result)
        assert list(result_native["result"]) == [1, 0, 1, 0, 0]

    def test_output_col(self, df_basic):
        """Test using column values as output."""
        t = When(
            output_col="result",
            conditions=[
                {"input_col": "flag", "operator": "eq", "value": True, "output_col": "col_a"},
            ],
            otherwise_col="col_b",
        )
        result = t.transform(df_basic)

        result_native = nw.to_native(result)
        assert list(result_native["result"]) == [10, 200, 30, 400, 500]

    # ------------- test null-handling -------------

    def test_null_in_comparison_uses_otherwise(self, df_basic):
        """Test that null comparisons fall through to otherwise."""
        t = When(
            output_col="result",
            conditions=[
                {"input_col": "value", "operator": "gt", "value": 2, "output_constant": "big"},
            ],
            otherwise_constant="small_or_null",
        )
        result = t.transform(df_basic)

        result_native = nw.to_native(result)
        # value: [1, 2, None, 4]
        # 1 > 2: False → otherwise
        # 2 > 2: False → otherwise
        # None > 2: null → otherwise
        # 4 > 2: True → "big"
        assert list(result_native["result"]) == [
            "small_or_null",
            "small_or_null",
            "small_or_null",
            "big",
            "big",
        ]

    def test_explicit_null_check(self, df_basic):
        """Test explicit null checking in conditions."""
        t = When(
            output_col="result",
            conditions=[
                {"input_col": "value", "operator": "is_null", "output_constant": "missing"},
                {"input_col": "value", "operator": "gt", "value": 2, "output_constant": "big"},
            ],
            otherwise_constant="small",
        )
        result = t.transform(df_basic)

        result_native = nw.to_native(result)
        # Null is explicitly caught by first condition
        assert list(result_native["result"]) == ["small", "small", "missing", "big", "big"]

    # ------------- test cast -------------

    def test_cast_to_string(self, df_basic):
        """Test casting all outputs to string."""
        t = When(
            output_col="result",
            conditions=[
                {"input_col": "value", "operator": "eq", "value": 1, "output_constant": 100},
                {"input_col": "value", "operator": "eq", "value": 2, "output_col": "value"},
            ],
            otherwise_constant=999,
            cast_output="string",
        )
        result = t.transform(df_basic)

        result_native = nw.to_native(result)
        assert list(result_native["result"]) == ["100", "2", "999", "999", "999"]

    def test_cast_to_int(self, df_basic):
        """Test casting all outputs to integer."""
        t = When(
            output_col="result",
            conditions=[
                {"input_col": "value", "operator": "lt", "value": 2, "output_constant": 10},
            ],
            otherwise_col="value",
            cast_output="Int64",
        )
        result = t.transform(df_basic)

        result_native = nw.to_native(result)
        assert list(result_native["result"]) == [10, 2, None, 4, 5]

    # ------------- test complex scenarios -------------

    def test_risk_scoring(self):
        """Test a realistic risk scoring scenario."""
        df = nw.from_native(
            pl.DataFrame({
                "amount": [100, 5000, 15000, 500, None],
                "user_type": ["new", "verified", "verified", "new", "verified"],
            })
        )

        t = When(
            output_col="risk_level",
            conditions=[
                # High risk: large amount from new user
                {
                    "input_col": "amount",
                    "operator": "gt",
                    "value": 10000,
                    "output_constant": "high",
                },
                # Medium risk: moderate amount or new user
                {
                    "input_col": "amount",
                    "operator": "gt",
                    "value": 1000,
                    "output_constant": "medium",
                },
                # Null amounts are suspicious
                {
                    "input_col": "amount",
                    "operator": "is_null",
                    "output_constant": "review",
                },
            ],
            otherwise_constant="low",
        )
        result = t.transform(df)

        result_native = nw.to_native(result)
        expected = ["low", "medium", "high", "low", "review"]
        assert list(result_native["risk_level"]) == expected

    def test_column_comparison(self):
        """Test when using column-to-column comparisons."""
        df = nw.from_native(
            pd.DataFrame({
                "actual": [100, 80, 120, None],
                "expected": [100, 100, 100, 100],
            })
        )

        t = When(
            output_col="performance",
            conditions=[
                {
                    "input_col": "actual",
                    "operator": "gt",
                    "compare_col": "expected",
                    "output_constant": "exceeded",
                },
                {
                    "input_col": "actual",
                    "operator": "eq",
                    "compare_col": "expected",
                    "output_constant": "met",
                },
            ],
            otherwise_constant="below",
        )
        result = t.transform(df)

        result_native = nw.to_native(result)
        # [100==100: met, 80<100: below, 120>100: exceeded, None: below]
        assert list(result_native["performance"]) == [
            "met",
            "below",
            "exceeded",
            "below",
        ]
