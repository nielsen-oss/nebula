"""Unit-tests for 'reshaping' transformers."""

from random import randint

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from nlsn.nebula.transformers import *
from nlsn.tests.auxiliaries import from_pandas, to_pandas
from nlsn.tests.constants import TEST_BACKENDS


class TestGroupBy:
    """Test GroupBy transformer across multiple backends."""

    @pytest.mark.parametrize("prefix, suffix", [("a", ""), ("", "b"), ("a", "b")])
    @pytest.mark.parametrize(
        "agg",
        [
            {"agg": "sum", "col": "c1"},
            [{"agg": "sum", "col": "c1"}],
            [{"agg": "sum", "col": "c1"}, {"agg": "sum", "col": "c2"}],
        ],
    )
    def test_invalid_single_op(self, agg, prefix, suffix):
        """Test that prefix/suffix raise error when not in single-op format."""
        with pytest.raises(ValueError, match="prefix.*suffix.*allowed only"):
            GroupBy(
                aggregations=agg,
                groupby_columns="x",
                prefix=prefix,
                suffix=suffix
            )

    def test_invalid_aggregation(self, df_input):
        """Test that invalid aggregation functions are caught."""
        with pytest.raises(ValueError, match="aggregation"):
            GroupBy(
                aggregations={"not_a_real_agg": ["c1"]},
                groupby_columns="c0"
            )

    def test_missing_groupby_selection(self):
        """Test that at least one groupby selection method is required."""
        with pytest.raises(AssertionError):
            GroupBy(aggregations={"sum": ["c1"]})

    def test_multiple_groupby_selections_disallowed(self, df_input):
        """Test that only one groupby selection method can be used."""
        with pytest.raises(AssertionError):
            GroupBy(
                aggregations={"sum": ["c1"]},
                groupby_columns="c0",
                groupby_regex="^c"
            )

    @staticmethod
    @pytest.fixture(scope="class")
    def df_input():
        data = [{f"c{i}": randint(1, 1 << 8) for i in range(5)} for _ in range(20)]
        return pd.DataFrame(data)

    @staticmethod
    def __reset_index(df, cols: list[str]) -> pd.DataFrame:
        if isinstance(df, pd.DataFrame):
            if df.index.name in {None, "index"}:
                return df.sort_values(cols).reset_index(drop=True)
            return df.reset_index().sort_values(cols).reset_index(drop=True)
        return to_pandas(df).sort_values(cols).reset_index(drop=True)

    def _compare(self, df_chk, df_exp):
        cols = list(df_exp.columns)
        df_chk = self.__reset_index(df_chk, cols)
        df_exp = self.__reset_index(df_exp, cols)
        pd.testing.assert_frame_equal(df_chk, df_exp, check_dtype=False)

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    @pytest.mark.parametrize("to_nw", [True, False])
    @pytest.mark.parametrize(
        "aggregations",
        [
            [
                {
                    "col": col,
                    "agg": agg,
                    **({"alias": alias} if alias is not None else {})
                }
                for col, agg, alias in zip(
                (f"c{i}" for i in range(3, 5)),
                ("sum", "count"),
                (None, "out"),
            )
            ]
        ],
    )
    @pytest.mark.parametrize("groupby_cols", [["c2"], ["c1", "c2"]])
    def test_multiple_aggregations(
            self, spark, backend, to_nw, df_input, aggregations, groupby_cols
    ):
        """Test multiple aggregations with and without aliases."""
        t = GroupBy(aggregations=aggregations, groupby_columns=groupby_cols)
        df_result = t.transform(from_pandas(df_input, backend, to_nw, spark=spark))

        # Build expected result manually
        df_nw = nw.from_native(df_input)
        agg_exprs = []
        for el in aggregations:
            col_expr = nw.col(el["col"])
            agg_expr = getattr(col_expr, el["agg"])()
            if "alias" in el:
                agg_expr = agg_expr.alias(el["alias"])
            agg_exprs.append(agg_expr)

        df_nw_exp = df_nw.group_by(groupby_cols).agg(agg_exprs)
        self._compare(df_result, df_nw_exp)

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    @pytest.mark.parametrize("to_nw", [True, False])
    @pytest.mark.parametrize("groupby_columns", [["c1"], ["c1", "c2"]])
    def test_single_dict_aggregation(self, spark, backend, to_nw, df_input, groupby_columns):
        """Test a single aggregation provided as a dict."""
        aggregations = {"col": "c3", "agg": "sum", "alias": "result"}
        t = GroupBy(aggregations=aggregations, groupby_columns=groupby_columns)
        df_result = t.transform(df_input)

        df_nw = nw.from_native(df_input)
        df_nw_exp = df_nw.group_by(groupby_columns).agg(
            nw.col("c3").sum().alias("result")
        )
        self._compare(df_result, df_nw_exp)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    @pytest.mark.parametrize("to_nw", [True, False])
    @pytest.mark.parametrize("prefix", ["pre_", ""])
    @pytest.mark.parametrize("suffix", ["_post", ""])
    def test_single_aggregation_multiple_columns(
            self, spark, backend, to_nw, df_input, prefix: str, suffix: str
    ):
        """Test single aggregation on multiple columns with prefix/suffix."""
        t = GroupBy(
            aggregations={"sum": ["c2", "c3"]},
            groupby_columns="c1",
            prefix=prefix,
            suffix=suffix,
        )
        df_result = t.transform(from_pandas(df_input, backend, to_nw, spark=spark))

        df_nw = nw.from_native(df_input)
        agg_exprs = [
            nw.col(col).sum().alias(f"{prefix}{col}{suffix}")
            for col in ["c2", "c3"]
        ]
        df_nw_exp = df_nw.group_by("c1").agg(agg_exprs)
        self._compare(df_result, df_nw_exp)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    @pytest.mark.parametrize("to_nw", [True, False])
    def test_with_regex(self, spark, backend, to_nw, df_input):
        """Test groupby column selection with regex."""
        t = GroupBy(
            aggregations={"sum": ["c3"]},
            groupby_regex="^c[12]$",  # matches c1 and c2
        )
        df_result = t.transform(from_pandas(df_input, backend, to_nw, spark=spark))

        df_nw = nw.from_native(df_input)
        df_nw_exp = df_nw.group_by(["c1", "c2"]).agg(
            nw.col("c3").sum().alias("c3")
        )
        self._compare(df_result, df_nw_exp)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    @pytest.mark.parametrize("to_nw", [True, False])
    def test_with_startswith(self, spark, backend, to_nw, df_input):
        """Test groupby column selection with startswith."""
        # Add a column that doesn't start with 'c'
        df_modified = df_input.copy()
        df_modified["other"] = df_modified["c0"].copy()

        t = GroupBy(
            aggregations={"count": ["c0"]},
            groupby_startswith="c",
            suffix="_count"
        )
        df_result = t.transform(from_pandas(df_modified, backend, to_nw, spark=spark))

        # Should group by all columns starting with 'c'
        groupby_cols = [c for c in df_modified.columns if c.startswith("c")]
        df_nw = nw.from_native(df_input)
        df_nw_exp = df_nw.group_by(groupby_cols).agg(
            nw.col("c0").count().alias("c0_count")
        )
        self._compare(df_result, df_nw_exp)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    @pytest.mark.parametrize("to_nw", [True, False])
    @pytest.mark.parametrize(
        "agg_func",
        ["sum", "mean", "median", "min", "max", "std", "var", "count", "first", "last"]
    )
    def test_various_aggregations(self, spark, backend, to_nw, df_input, agg_func):
        """Test that various common aggregation functions work."""
        t = GroupBy(
            aggregations={agg_func: ["c3"]},
            groupby_columns="c1",
            suffix=f"_{agg_func}"
        )
        df_result = t.transform(from_pandas(df_input, backend, to_nw, spark=spark))

        # Just verify it runs without error and produces expected column
        assert f"c3_{agg_func}" in df_result.columns
        assert "c1" in df_result.columns


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
