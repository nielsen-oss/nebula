import pandas as pd
import polars as pl
import pytest
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql.types import StringType, StructField, StructType

from nlsn.nebula.storage import nebula_storage as ns
from nlsn.nebula.transformers import DropNulls, AppendDataFrame
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
