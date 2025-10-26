"""Unit-test for DuplicateColumn."""

import pytest
from pyspark.sql.types import StringType, StructField, StructType
from pyspark.sql.utils import AnalysisException

from nlsn.nebula.spark_transformers import DuplicateColumn


def test_duplicate_column_not_present_column(df_input):
    """Test DuplicateColumn passing a not present column."""
    t = DuplicateColumn(col_map={"not_present_column": "x_dup", "y": "y_dup"})
    with pytest.raises(AnalysisException):
        t.transform(df_input)


@pytest.mark.parametrize("as_dict", [True, False])
def test_duplicate_column_wrong_values(as_dict):
    """Test DuplicateColumn passing wrong parameters."""
    mapping = [("x", "x_dup"), ("y", "x_dup")]
    if as_dict:
        kws = {"col_map": dict(mapping)}
    else:
        kws = {"pairs": mapping}
    with pytest.raises(AssertionError):
        DuplicateColumn(**kws)


def test_duplicate_column_keys_values_intersection():
    """Test DuplicateColumn passing a new column name that already exists."""
    with pytest.raises(AssertionError):
        DuplicateColumn(col_map={"x": "x_dup", "y": "x"})


@pytest.mark.parametrize("el", [("i",), ("i", "j", "k")])
def test_duplicate_column_wrong_pairs_length(el):
    """Test DuplicateColumn passing a wrong number of nested elements."""
    with pytest.raises(AssertionError):
        DuplicateColumn(pairs=[("x", "x_dup"), el])


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    """Define test data."""
    fields = [
        StructField(name="x", dataType=StringType(), nullable=True),
        StructField(name="y", dataType=StringType(), nullable=True),
        StructField(name="z", dataType=StringType(), nullable=True),
    ]

    schema = StructType(fields)

    data = [
        ("1", "2", "3"),
        ("4", "5", "6"),
    ]
    return spark.createDataFrame(data=data, schema=schema)


@pytest.mark.parametrize("as_dict", [True, False])
def test_duplicate_column_valid(df_input, as_dict):
    """Test DuplicateColumn using valid parameters."""
    mapping = [("x", "x_dup"), ("y", "z")]
    if as_dict:
        kws = {"col_map": dict(mapping)}
    else:
        kws = {"pairs": mapping}

    set_values = {i[1] for i in mapping}

    t = DuplicateColumn(**kws)
    df_out = t.transform(df_input)

    input_cols = set(df_input.columns)
    new_cols = set(df_out.columns)

    assert input_cols.union(set_values) == new_cols

    df_pd = df_out.toPandas()
    for c1, c2 in mapping:
        s1 = df_pd[c1].tolist()
        s2 = df_pd[c2].tolist()
        assert s1 == s2
