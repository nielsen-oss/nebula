"""Unit-test for MapToColumns."""

import pytest
from pyspark.sql.types import IntegerType, MapType, StringType, StructField, StructType

from nlsn.nebula.auxiliaries import is_list_uniform
from nlsn.nebula.spark_transformers import MapToColumns

_data = [
    (1, {"a": 1, "b": 2}),
    (2, {"x": None, "y": 20}),
    (3, None),
]


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    fields = [
        StructField("id", IntegerType()),
        StructField("map_col", MapType(StringType(), IntegerType())),
    ]
    schema = StructType(fields)
    return spark.createDataFrame(_data, schema).persist()


@pytest.mark.parametrize(
    "output_columns",
    [
        ["a", "b", "wrong"],
        [("a", "col_a"), ("c", "col_c")],
        {"a": "col_a", "c": "col_c", "x": "col_x"},
    ],
)
def test_map_to_columns(df_input, output_columns):
    """Test MapToColumns w/ and w/o 'output_column'."""
    t = MapToColumns(input_column="map_col", output_columns=output_columns)
    df_out = t.transform(df_input)

    collected = df_out.rdd.map(lambda x: x.asDict()).collect()

    if isinstance(output_columns, (list, tuple)):
        if is_list_uniform(output_columns, str):
            chk_cols = [(i, i) for i in output_columns]
        else:
            chk_cols = output_columns[:]
    else:
        chk_cols = list(output_columns.items())

    for row in collected:
        input_dict = row["map_col"]
        if not input_dict:
            assert all(row[j] is None for (i, j) in chk_cols)
            continue
        for i, j in chk_cols:
            exp = input_dict.get(i)
            chk = row[j]
            assert exp == chk


def test_map_to_columns_with_missing_input_column():
    """Test MapToColumns passing an empty output column list."""
    with pytest.raises(AssertionError):
        MapToColumns(input_column="id", output_columns=[])


def test_map_to_columns_with_wrong_output_columns_type():
    """Test MapToColumns passing a wrong output column type."""
    with pytest.raises(TypeError):
        MapToColumns(input_column="id", output_columns={"a"})


def test_map_to_columns_with_non_map_type_column(df_input):
    """Test MapToColumns passing a non MapType column."""
    t = MapToColumns(input_column="id", output_columns=["x"])
    with pytest.raises(TypeError):
        t.transform(df_input)


@pytest.mark.parametrize(
    "output_columns",
    [
        ["a", 1],
        ["a", None],
        [None],
        [("a", "col_a"), ("c", "col_a")],
        [("a", "col_a"), ("c", "col_a", "col_b")],
        [("a", "col_a"), ("c", 1)],
        {"a": "col_a", "c": "col_a"},
        {"a": "col_a", "c": 1},
    ],
)
def test_map_to_columns_with_wrong_output_columns(output_columns):
    """Test MapToColumns passing a wrong type of 'output_columns'."""
    with pytest.raises(TypeError):
        MapToColumns(input_column="map_col", output_columns=output_columns)
