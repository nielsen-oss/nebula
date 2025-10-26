"""Unit-test for KeysToArray."""

import pytest
from pyspark.sql.types import IntegerType, MapType, StringType, StructField, StructType

from nlsn.nebula.spark_transformers import KeysToArray

_data = [
    (1, {"a": 1, "b": 2}),
    (2, {"x": 10, "y": 20}),
    (3, None),
]


def _sort_results(li) -> list:
    return sorted(li, key=lambda x: x[0])


def _extract_elements(data, sort: bool) -> list:
    func = sorted if sort else set
    ret = []
    for id_, keys in data:
        if keys is None:
            ret.append((id_, None))
        else:
            ret.append((id_, func(keys)))
    return _sort_results(ret)


def _get_expected(sort: bool) -> list:
    func = sorted if sort else set
    ret = []
    for id_, nd in _data:
        if nd is None:
            ret.append((id_, None))
        else:
            ret.append((id_, func(nd.keys())))
    return _sort_results(ret)


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    fields = [
        StructField("id", IntegerType()),
        StructField("map_col", MapType(StringType(), IntegerType())),
    ]
    schema = StructType(fields)
    return spark.createDataFrame(_data, schema).persist()


@pytest.mark.parametrize("output_column", [None, "new_col"])
@pytest.mark.parametrize("sort", [True, False])
def test_map_keys(df_input, output_column, sort: bool):
    """Test KeysToArray w/ and w/o 'output_column'."""
    t = KeysToArray(input_column="map_col", output_column=output_column, sort=sort)
    df_out = t.transform(df_input)

    if output_column:
        chk_cols = ["id", output_column]
    else:
        chk_cols = ["id", "map_col"]

    collected = df_out.select(chk_cols).rdd.map(lambda x: x[:]).collect()
    chk = _extract_elements(collected, sort)
    exp = _get_expected(sort)
    assert chk == exp


def test_map_keys_with_missing_input_column(df_input):
    """Test KeysToArray passing a non-existing column."""
    t = KeysToArray(input_column="non_existing_col")
    with pytest.raises(AssertionError):
        t.transform(df_input)


def test_map_keys_with_non_map_type_column(df_input):
    """Test KeysToArray passing a non MapType column."""
    t = KeysToArray(input_column="id")
    with pytest.raises(TypeError):
        t.transform(df_input)
