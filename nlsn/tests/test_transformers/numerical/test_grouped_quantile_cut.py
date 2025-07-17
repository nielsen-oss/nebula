"""Unit-test for GroupedQuantileCut."""

import numpy as np
import pandas as pd
import pytest
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from nlsn.nebula.spark_transformers import GroupedQuantileCut


def test_grouped_quantile_cut_init():
    """Ensure the quantiles are sorted and extended to 0 and 1."""
    q = [0.5, 0.1]
    t = GroupedQuantileCut(
        quantiles=q,
        groupby_cols="a",
        input_col="b",
    )
    exp = [0.0] + sorted(q) + [1.0]
    assert t._q == exp


class TestGroupedQuantileCutError:
    """Test invalid initialization."""

    @staticmethod
    @pytest.mark.parametrize("fillna, start_value", [(1, 0), (0, 1), (1, 1)])
    def test_init_conflicting_params(fillna: int, start_value: int):
        """Test invalid scalar quantiles."""
        kws = {
            "quantiles": [0.5, 0.1],
            "groupby_cols": "group",
            "input_col": "value",
            "retbins": True,
            "fillna": fillna,
            "start_value": start_value,
        }
        with pytest.raises(ValueError):
            GroupedQuantileCut(**kws)

    @staticmethod
    def test_init_invalid_quantiles_int():
        """Test invalid scalar quantiles."""
        with pytest.raises(ValueError):
            GroupedQuantileCut(quantiles=1, groupby_cols="group", input_col="value")
        with pytest.raises(TypeError):
            GroupedQuantileCut(quantiles=2.5, groupby_cols="group", input_col="value")

    @staticmethod
    def test_init_invalid_quantiles_list():
        """Test invalid iterable quantiles."""
        with pytest.raises(ValueError):
            GroupedQuantileCut(
                quantiles=[0.1, "bad"], groupby_cols="group", input_col="value"
            )
        with pytest.raises(ValueError):
            GroupedQuantileCut(
                quantiles=[-0.1, 0.5], groupby_cols="group", input_col="value"
            )
        with pytest.raises(ValueError):
            GroupedQuantileCut(
                quantiles=[0.5, 1.1], groupby_cols="group", input_col="value"
            )

    @staticmethod
    def test_init_invalid_fillna_start_value():
        """Test invalid start value and fillna values."""
        with pytest.raises(TypeError):
            GroupedQuantileCut(
                quantiles=4, groupby_cols="group", input_col="value", fillna="abc"
            )
        with pytest.raises(TypeError):
            GroupedQuantileCut(
                quantiles=4, groupby_cols="group", input_col="value", start_value=0.5
            )

    @staticmethod
    def test_no_groupby_cols():
        """Test without groupby column(s)."""
        with pytest.raises(ValueError):
            GroupedQuantileCut(
                quantiles=3,
                groupby_cols=[],
                input_col="value",
                output_col="q_value",
            )


_DATA = [
    ("A", 10),
    ("A", 20),
    ("A", 30),
    ("A", 40),
    ("A", 50),
    ("B", 5),
    ("B", 15),
    ("B", 25),
    ("C", 100),
    ("C", 100),
    ("C", 100),  # Duplicates in C
    ("D", 5),  # Single element group
    ("E", 7),
    ("E", 7),  # Duplicate in E, small group
    ("F", 1),
    ("F", 2),
    ("F", 3),
    ("F", 4),
    ("F", 5),  # Many quantiles
]

_DF_INPUT_PD = pd.DataFrame(_DATA, columns=["group", "value"])

fields = [
    StructField("group", StringType(), True),
    StructField("value", IntegerType(), True),
]


class TestGroupedQuantileCut:
    @pytest.fixture(scope="class")
    def sample_data_df(self, spark):
        """Provides a sample Spark DataFrame for testing."""
        return spark.createDataFrame(_DATA, StructType(fields)).persist()

    @staticmethod
    def _get_expected_pdf(
        df_pd: pd.DataFrame,
        *,
        quantiles,
        groupby_cols,
        input_col: str,
        output_col=None,
        fillna=0,
        start_value=0,
        retbins: bool = False,
    ):
        list_df_exp = []
        for _group, df_grouped in df_pd.groupby(groupby_cols):
            ar = df_grouped[input_col].values
            n = len(ar)
            if retbins:
                if n <= 1:
                    df_grouped["_q_"] = [[]] * n
                    list_df_exp.append(df_grouped)
                    continue
                _, ar_rank = pd.qcut(
                    ar, quantiles, labels=False, duplicates="drop", retbins=True
                )
                df_grouped["_q_"] = [ar_rank.tolist()] * n
                list_df_exp.append(df_grouped)
            else:
                if n <= 1:
                    df_grouped["_q_"] = np.int32(start_value)
                    list_df_exp.append(df_grouped)
                    continue
                cuts = (
                    pd.qcut(ar, quantiles, labels=False, duplicates="drop")
                    + start_value
                )
                cuts[np.isnan(cuts)] = fillna
                df_grouped["_q_"] = cuts.astype(np.int32)
                list_df_exp.append(df_grouped)

        # Combine all expected dataframes
        df_exp = pd.concat(list_df_exp, axis=0)

        by = groupby_cols[:] if isinstance(groupby_cols, list) else [groupby_cols]
        by += ["value"]

        df_exp = df_exp.sort_values(by=by).reset_index(drop=True)

        if output_col is None:
            df_exp["value"] = df_exp["_q_"]
            df_exp.drop(columns=["_q_"], inplace=True)
        else:
            df_exp = df_exp.rename(columns={"_q_": output_col})
        return df_exp

    @staticmethod
    def _assert(df_chk_spark, df_chk: pd.DataFrame, df_exp: pd.DataFrame, params):
        if "output_col" in params:
            output_col = params["output_col"]
            assert output_col in df_chk_spark.columns
            assert df_chk_spark.schema[output_col].dataType == IntegerType()
        else:
            assert df_chk_spark.schema[params["input_col"]].dataType == IntegerType()

        pd.testing.assert_frame_equal(df_chk, df_exp, check_dtype=False)

    @pytest.mark.parametrize(
        "params",
        [
            {  # quantiles as int
                "quantiles": 4,
                "groupby_cols": "group",
                "input_col": "value",
                "output_col": "q_value",
            },
            {  # no putput column
                "quantiles": 3,
                "groupby_cols": "group",
                "input_col": "value",
            },
            {  # quantiles as list
                "quantiles": [0.0, 0.25, 0.5, 0.75, 1.0],
                "groupby_cols": "group",
                "input_col": "value",
                "output_col": "q_value",
            },
            {  # start_value
                "quantiles": 3,
                "groupby_cols": "group",
                "input_col": "value",
                "output_col": "q_value",
                "start_value": 10,
            },
            {  # fillna
                "quantiles": 3,
                "groupby_cols": "group",
                "input_col": "value",
                "output_col": "q_value",
                "fillna": -99,
            },
            {  # start_value & fillna
                "quantiles": 3,
                "groupby_cols": "group",
                "input_col": "value",
                "output_col": "q_value",
                "start_value": 10,
                "fillna": -99,
            },
        ],
    )
    def test_quantiles(self, sample_data_df, params):
        """Test with basic quantiles."""
        t = GroupedQuantileCut(**params)
        df_chk_spark = t.transform(sample_data_df)
        df_chk = (
            df_chk_spark.toPandas()
            .sort_values(by=["group", "value"])
            .reset_index(drop=True)
        )

        df_exp = self._get_expected_pdf(_DF_INPUT_PD, **params)

        self._assert(df_chk_spark, df_chk, df_exp, params)

    def test_non_integer_input_col(self, spark):
        """Test with non-integer input_col."""
        data = [("A", 10.5), ("A", 20.1), ("A", 30.9), ("B", 5.0), ("B", 15.0)]
        schema = StructType(
            [
                StructField("group", StringType(), True),
                StructField("value", DoubleType(), True),
            ]
        )
        df = spark.createDataFrame(data, schema)

        params = {
            "quantiles": 24,
            "groupby_cols": "group",
            "input_col": "value",
            "output_col": "q_value",
        }
        t = GroupedQuantileCut(**params)
        df_chk_spark = t.transform(df)
        df_chk = (
            df_chk_spark.toPandas()
            .sort_values(by=["group", "value"])
            .reset_index(drop=True)
        )

        df_input_pd = pd.DataFrame(data, columns=["group", "value"])
        df_exp = self._get_expected_pdf(df_input_pd, **params)

        self._assert(df_chk_spark, df_chk, df_exp, params)

    def test_multi_groupby_cols(self, spark):
        """Test with multiple groupby columns."""
        data = [
            ("X", "P", 10),
            ("X", "P", 20),
            ("X", "P", 30),
            ("X", "Q", 5),
            ("X", "Q", 15),
            ("Y", "P", 100),
            ("Y", "P", 100),  # Duplicates in Y, P
        ]
        schema = StructType(
            [
                StructField("col1", StringType(), True),
                StructField("col2", StringType(), True),
                StructField("value", IntegerType(), True),
            ]
        )
        df = spark.createDataFrame(data, schema)

        params = {
            "quantiles": 2,
            "groupby_cols": ["col1", "col2"],
            "input_col": "value",
            "output_col": "q_value",
        }

        t = GroupedQuantileCut(**params)
        df_chk_spark = t.transform(df)
        df_chk = (
            df_chk_spark.toPandas()
            .sort_values(by=["col1", "col2", "value"])
            .reset_index(drop=True)
        )

        df_input_pd = pd.DataFrame(data, columns=["col1", "col2", "value"])
        df_exp = self._get_expected_pdf(df_input_pd, **params)

        self._assert(df_chk_spark, df_chk, df_exp, params)

    @pytest.mark.parametrize("quantiles", [3, [0.0, 0.25, 0.5, 0.75, 1.0]])
    def test_retbins(self, sample_data_df, quantiles):
        """Test with retbins=True."""
        params = {
            "quantiles": quantiles,
            "groupby_cols": "group",
            "input_col": "value",
            "output_col": "q_value",
            "retbins": True,
        }
        t = GroupedQuantileCut(**params)
        df_chk_spark = t.transform(sample_data_df)

        type_chk = df_chk_spark.schema[params["output_col"]].dataType
        type_exp = ArrayType(DoubleType(), True)
        assert type_chk == type_exp

        df_chk = (
            df_chk_spark.toPandas()
            .sort_values(by=["group", "value"])
            .reset_index(drop=True)
        )

        df_exp = self._get_expected_pdf(_DF_INPUT_PD, **params)

        pd.testing.assert_frame_equal(df_chk, df_exp, check_dtype=False)
