"""Unit-test for AddTypedColumns."""

import pyspark.sql.functions as F
import pytest
from pyspark.sql.types import FloatType, StructField, StructType

from nlsn.nebula.spark_transformers import AddTypedColumns


@pytest.fixture(scope="module", name="df_input")
def _get_input_df(spark):
    fields = [
        StructField("a", FloatType(), True),
        StructField("b", FloatType(), True),
        StructField("c", FloatType(), True),
    ]

    data = [
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
    ]

    return spark.createDataFrame(data, schema=StructType(fields)).persist()


def test_add_typed_columns_error():
    """Test AddTypedColumns with invalid input."""
    # Not allowed type.
    with pytest.raises(AssertionError):
        AddTypedColumns(columns="string")

    # Not pairs
    columns = [("c1", "t1"), ("c2", "t2", "x"), ("c3", "t3")]
    with pytest.raises(AssertionError):
        AddTypedColumns(columns=columns)

    # Not allowed dictionary
    columns = {1: "string"}
    with pytest.raises(AssertionError):
        AddTypedColumns(columns=columns)

    # Not allowed nested dictionary
    columns = {"c1": {"type": "string", "wrong": 1}}
    with pytest.raises(AssertionError):
        AddTypedColumns(columns=columns)


def test_add_typed_columns_empty_input(df_input):
    """Test AddTypedColumns with empty input."""
    AddTypedColumns(columns=None).transform(df_input)
    AddTypedColumns(columns={}).transform(df_input)
    AddTypedColumns(columns=[]).transform(df_input)


def _assert_fields(df_input, df_out):
    fields_chk = df_out.schema.fields
    fields_input = df_input.schema.fields
    n_input = len(fields_input)

    assert fields_input == fields_chk[:n_input]

    new_fields = fields_chk[n_input:]
    assert len(new_fields) == (len(fields_chk) - n_input)

    # Assert input column are untouched
    for c in df_input.columns:
        n_null = df_out.filter(F.col(c).isNull()).count()
        assert n_null == 0

    return new_fields


def test_add_typed_columns_null_values(df_input):
    """Test AddTypedColumns with valid input and null values."""
    input_columns = [
        ("c", "string"),
        ("d", "integer"),  # "integer", not "int"
    ]

    t = AddTypedColumns(columns=input_columns)
    df_out = t.transform(df_input)

    new_fields = _assert_fields(df_input, df_out)

    n_rows: int = df_input.count()
    dict_types = dict(input_columns)

    # Assert new columns are properly cast and contains only null
    for field in new_fields:
        column_name: str = field.name
        # typeName returns 'integer' not 'int' as IntegerType name.
        type_name: str = field.dataType.typeName()

        assert dict_types[column_name] == type_name

        n_null = df_out.filter(F.col(column_name).isNull()).count()
        assert n_null == n_rows


def test_add_typed_columns_non_null_values(df_input):
    """Test AddTypedColumns with valid input and default values."""
    input_columns = {
        "b": {"type": "string", "value": "X"},
        "d": {"type": "integer", "value": 1},  # "integer", not "int"
        "e": "float",
    }

    t = AddTypedColumns(columns=input_columns)
    df_out = t.transform(df_input).persist()

    new_fields = _assert_fields(df_input, df_out)

    n_rows: int = df_input.count()

    # Assert new columns are properly cast and contains only null
    for field in new_fields:
        column_name: str = field.name
        # typeName returns 'integer' not 'int' as IntegerType name.
        type_name: str = field.dataType.typeName()

        nd_input = input_columns[column_name]
        if isinstance(nd_input, dict):
            type_exp = nd_input["type"]
            default_value = input_columns[column_name]["value"]
            cond = F.col(column_name) == default_value
        else:
            type_exp = nd_input
            cond = F.col(column_name).isNull()

        assert type_exp == type_name

        n_null = df_out.filter(cond).count()
        assert n_null == n_rows
