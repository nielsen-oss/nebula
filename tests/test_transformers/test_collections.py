"""Unit-tests for 'collections' transformers."""

import operator as py_operator

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from nebula.transformers import *


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
        """Test MathOperator with a wrong strategy type."""
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

        # Build an expected result using Python operators
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
