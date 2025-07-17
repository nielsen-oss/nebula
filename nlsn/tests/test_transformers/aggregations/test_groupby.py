"""Unit-test for GroupBy."""

from random import randint

import pytest
from chispa import assert_df_equality
from pyspark.sql import functions as F

from nlsn.nebula.spark_transformers import GroupBy


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    data = [{f"c{i}": randint(1, 1 << 8) for i in range(5)} for _ in range(20)]
    return spark.createDataFrame(data).persist()


@pytest.mark.parametrize("prefix, suffix", [("a", ""), ("", "b"), ("a", "b")])
@pytest.mark.parametrize(
    "agg",
    [
        {"agg": "sum", "col": "c1"},
        [{"agg": "sum", "col": "c1"}],
        [{"agg": "sum", "col": "c1"}, {"agg": "sum", "col": "c2"}],
    ],
)
def test_groupby_invalid_single_op(agg, prefix, suffix):
    """Test prefix / suffix when not allowed."""
    with pytest.raises(ValueError):
        GroupBy(aggregations=agg, groupby_columns="x", prefix=prefix, suffix=suffix)


@pytest.mark.parametrize(
    "aggregations",
    [
        [
            {"col": col, "agg": agg, **({"alias": alias} if alias is not None else {})}
            for col, agg, alias in zip(
                (f"c{i}" for i in range(3, 5)),
                ("sum", "count"),
                (None, "out"),
            )
        ]
    ],
)
@pytest.mark.parametrize("groupby_cols", [["c2"], ["c1", "c2"]])
def test_groupby_multiple_aggregations(df_input, aggregations, groupby_cols):
    """Test multiple aggregations."""
    t = GroupBy(aggregations=aggregations, groupby_columns=groupby_cols)

    df_chk = t.transform(df_input)

    list_agg = []
    for el in aggregations:
        col: F.col = getattr(F, el["agg"])(el["col"])
        if "alias" in el:
            col = col.alias(el["alias"])
        list_agg.append(col)

    df_exp = df_input.groupBy(groupby_cols).agg(*list_agg)
    assert_df_equality(df_chk, df_exp, ignore_row_order=True)


@pytest.mark.parametrize("groupby_columns", [["c1"], ["c1", "c2"]])
def test_group_by_single_dict_aggregation(df_input, groupby_columns):
    """Test a single aggregation as <dict>."""
    aggregations = {"col": "c3", "agg": "sum", "alias": "result"}
    t = GroupBy(aggregations=aggregations, groupby_columns=groupby_columns)

    df_chk = t.transform(df_input)

    df_exp = df_input.groupBy(groupby_columns).agg(F.sum("c3").alias("result"))
    assert_df_equality(df_chk, df_exp, ignore_row_order=True)


@pytest.mark.parametrize("prefix", ["pre_", ""])
@pytest.mark.parametrize("suffix", ["_post", ""])
def test_groupby_single_aggregation_multiple_columns(
    df_input, prefix: str, suffix: str
):
    """Test a single aggregation on multiple columns."""
    t = GroupBy(
        aggregations={"countDistinct": ["c2", "c3"]},
        groupby_columns="c1",
        prefix=prefix,
        suffix=suffix,
    )

    df_chk = t.transform(df_input)

    agg = [F.countDistinct(f"{i}").alias(f"{prefix}{i}{suffix}") for i in ["c2", "c3"]]
    df_exp = df_input.groupBy("c1").agg(*agg)
    assert_df_equality(df_chk, df_exp, ignore_row_order=True)
