"""Unit-tests for 'combining' transformers."""

import pandas as pd
import polars as pl
import pytest

from nlsn.nebula import nebula_storage as ns
from nlsn.nebula.transformers import *
from nlsn.tests.auxiliaries import from_pandas, to_pandas


class TestAppendDataFrame:
    @staticmethod
    def _set_dfs(backend: str, to_nw: bool):
        ns.allow_overwriting()
        df1 = pd.DataFrame({"c1": ["c", "d"], "c2": [3, 4]})
        df2 = pd.DataFrame({"c1": ["a", "b"], "c3": [4.5, 5.5]})
        df1 = from_pandas(df1, backend, to_nw)
        df2 = from_pandas(df2, backend, to_nw)
        ns.set("df1", df1)
        ns.set("df2", df2)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    @pytest.mark.parametrize("to_nw", [True, False])
    @pytest.mark.parametrize("allow_missing", [True, False])
    @pytest.mark.parametrize("store_key", ["df1", "df2"])
    def test(self, backend: str, to_nw: bool, allow_missing: bool, store_key: str):
        self._set_dfs(backend, to_nw)
        df_pd_in = pd.DataFrame({"c1": ["a", "b"], "c2": [1, 2]})
        df = from_pandas(df_pd_in, backend, to_nw=to_nw)
        t = AppendDataFrame(store_key=store_key, allow_missing_cols=allow_missing)
        if store_key == "df2" and (not allow_missing):
            with pytest.raises(ValueError):
                t.transform(df)
            return
        df_chk = t.transform(df)
        df_chk_pd = to_pandas(df_chk).reset_index(drop=True)
        df_exp = pd.concat([df_pd_in, to_pandas(ns.get(store_key))], axis=0)
        pd.testing.assert_frame_equal(df_chk_pd, df_exp.reset_index(drop=True))


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
        """Test anti-join returns only non-matching left rows."""
        ns.set("right_table", df_right)

        transformer = Join(store_key="right_table", on="user_id", how="anti")
        result = transformer.transform(df_left)

        assert result.shape == (1, 3)  # Only user_id=1 doesn't match
        assert result["user_id"].to_list() == [1]
        assert result["name"].to_list() == ["Alice"]

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
