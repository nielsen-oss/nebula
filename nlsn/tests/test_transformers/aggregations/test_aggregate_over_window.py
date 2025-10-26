"""Unit-test for AggregateOverWindow."""

from string import ascii_lowercase

import numpy as np
import pytest
from chispa import assert_df_equality
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from nlsn.nebula.spark_transformers import AggregateOverWindow
from nlsn.nebula.spark_transformers.aggregations import validate_window_frame_boundaries

_AGGREGATIONS = [
    {"agg": "min", "col": "id", "alias": "min_id"},
    {"agg": "sum", "col": "id", "alias": "sum_id"},
]


@pytest.mark.parametrize("order_cols, partition_cols", [("x", None), (None, "x")])
def test_aggregate_over_window_alias_overriding(order_cols, partition_cols):
    """Test AggregateOverWindow transformer with overriding aliases."""
    with pytest.raises(AssertionError):
        AggregateOverWindow(
            order_cols=order_cols,
            partition_cols=partition_cols,
            aggregations=[{"agg": "sum", "col": "a", "alias": "x"}],
        )


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    n_rows = 200

    li_ids = np.random.randint(0, 50, n_rows).tolist()
    li_cat = np.random.choice(list(ascii_lowercase), n_rows).tolist()
    li_plt = np.random.choice(list("ABC"), n_rows).tolist()

    data = list(zip(li_ids, li_cat, li_plt))

    fields = [
        StructField("id", IntegerType(), True),
        StructField("category", StringType(), True),
        StructField("platform", StringType(), True),
    ]

    return spark.createDataFrame(data, schema=StructType(fields)).persist()


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize(
    "order_cols, rows_between, range_between",
    [
        [None, None, None],
        ["id", None, None],
        [None, (0, 2), None],
        [None, ("start", "end"), None],
        ["id", (0, 2), None],
        ["id", ("start", "end"), None],
        ["id", None, (0, 1)],
        ["id", None, ("start", "end")],
    ],
)
def test_aggregate_over_window(
    df_input, ascending, order_cols, rows_between, range_between
):
    """Test AggregateOverWindow with multiple cases."""
    t = AggregateOverWindow(
        partition_cols="category",
        order_cols=order_cols,
        aggregations=_AGGREGATIONS,
        ascending=ascending,
        rows_between=rows_between,
        range_between=range_between,
    )
    df_chk = t.transform(df_input)

    window = Window.partitionBy("category")
    if order_cols is not None:
        col_id = F.col("id")
        if not ascending:
            col_id = col_id.desc()
        window = window.orderBy(col_id)
    if rows_between is not None:
        start, end = validate_window_frame_boundaries(*rows_between)
        window = window.rowsBetween(start, end)
    if range_between is not None:
        start, end = validate_window_frame_boundaries(*range_between)
        window = window.rangeBetween(start, end)

    df_exp = df_input.withColumn("min_id", F.min("id").over(window)).withColumn(
        "sum_id", F.sum("id").over(window)
    )

    assert_df_equality(df_chk, df_exp, ignore_row_order=True, ignore_column_order=True)


@pytest.mark.parametrize("ascending", [[True, False], [False, True]])
def test_aggregate_over_window_ascending(df_input, ascending):
    """Test AggregateOverWindow with ascending as <list<bool>>."""
    order_cols = ["platform", "id"]

    t = AggregateOverWindow(
        partition_cols="category",
        order_cols=order_cols,
        ascending=ascending,
        aggregations=[{"agg": "first", "col": "id", "alias": "first_value"}],
    )
    df_chk = t.transform(df_input)

    # Set the order
    li_asc = ascending if isinstance(ascending, list) else [ascending] * 2
    orders = [
        F.col(i).asc() if j else F.col(i).desc() for i, j in zip(order_cols, li_asc)
    ]
    win = Window.partitionBy("category").orderBy(orders)
    df_exp = df_input.withColumn("first_value", F.first("id").over(win))

    assert_df_equality(df_chk, df_exp, ignore_row_order=True, ignore_column_order=True)


def test_aggregate_over_partition_rows_single_aggregation(df_input):
    """Test AggregateOverWindow w/o partitioning."""
    t = AggregateOverWindow(
        aggregations=[{"agg": "sum", "col": "id", "alias": "sum_id"}],
        rows_between=("start", "end"),
    )
    df_chk = t.transform(df_input)

    win = Window.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

    df_exp = df_input.withColumn("sum_id", F.sum("id").over(win))
    assert_df_equality(df_chk, df_exp, ignore_row_order=True, ignore_column_order=True)


def test_aggregate_over_window_override(df_input):
    """Test AggregateOverWindow with overriding the column 'id'."""
    t = AggregateOverWindow(
        partition_cols="category",
        aggregations={"agg": "min", "col": "id", "alias": "id"},
    )
    df_chk = t.transform(df_input)

    win = Window.partitionBy("category")

    df_exp = df_input.withColumn("id", F.min("id").over(win).cast("int"))
    assert_df_equality(df_chk, df_exp, ignore_row_order=True, ignore_column_order=True)
