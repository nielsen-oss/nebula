"""Unit-test for MapToColumns."""

from typing import List

import pytest
from pyspark.sql.types import IntegerType, Row, StructField, StructType

from nlsn.nebula.spark_transformers import ColumnsToMap

_data = [(0, 1, 11), (1, 2, None), (2, None, 3)]


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    fields = [
        StructField("idx", IntegerType()),
        StructField("c1", IntegerType()),
        StructField("c2", IntegerType()),
    ]
    schema = StructType(fields)
    return spark.createDataFrame(_data, schema).persist()


def _check(v_exp, v_chk, cast_values):
    if v_exp is None:
        assert v_chk is None
    else:
        if cast_values == "string":
            assert v_chk == str(v_exp)
        else:
            assert v_chk == v_exp


@pytest.mark.parametrize(
    "cast_values, drop",
    [
        [None, False],
        ["string", True],
    ],
)
def test_columns_to_map(df_input, cast_values: str, drop: bool):
    """Test ColumnsToMap."""
    t = ColumnsToMap(
        columns=["c1", "c2"],
        output_column="result",
        cast_values=cast_values,
        drop_input_columns=drop,
    )
    df_chk = t.transform(df_input).persist()
    n = df_chk.count()
    assert len(_data) == n

    set_cols = set(df_chk.columns)

    if drop:
        cols = ["idx", "result"]
        assert set_cols == set(cols)
    else:
        cols = ["idx", "c1", "c2", "result"]
        assert set_cols == set(cols)

    collected: List[Row] = df_chk.select(cols).sort("idx").collect()

    row: Row
    for row in collected:
        d_row = row.asDict()

        idx = d_row["idx"]

        v1_exp = _data[idx][1]
        v2_exp = _data[idx][2]

        result = d_row["result"]
        assert set(result.keys()) == {"c1", "c2"}

        v1_chk = result["c1"]
        v2_chk = result["c2"]

        _check(v1_exp, v1_chk, cast_values)
        _check(v2_exp, v2_chk, cast_values)
