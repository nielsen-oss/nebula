"""Unit-test for EnsureSameKeys."""

import random

import pytest
from pyspark.sql.types import IntegerType, MapType, StringType, StructField, StructType

from nlsn.nebula.spark_transformers import EnsureSameKeys


def _extract_expected_result(rows: list, input_cols: list):
    ret = []
    for row in rows:
        li_dicts = [getattr(row, i) for i in input_cols]
        if any(i is None for i in li_dicts):
            ret.append(False)
        elif all(i == {} for i in li_dicts):
            ret.append(True)
        else:
            li_set = [set(i) for i in li_dicts]
            # Compare each element with the first one
            first_element = li_set[0]
            are_identical = all(i == first_element for i in li_set)
            ret.append(are_identical)
    return ret


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    data = [
        # same keys
        ({}, {}, {}),
        ({"a": 1}, {"a": 2}, {"a": 3}),
        ({"a": 1, "b": 1}, {"a": 2, "b": 2}, {"a": 3, "b": 3}),
        # different keys in the last col
        ({}, {}, {"a": 3}),
        ({"a": 1}, {"a": 2}, {}),
        ({"a": 1, "b": 1}, {"a": 2, "b": 2}, {"a": 3, "b": 3, "c": 3}),
        ({"a": 1, "b": 1}, {"a": 2, "b": 2}, {"a": 3, "c": 3}),
        # different keys
        ({}, {"b": 2}, {"a": 3}),
        ({"a": 1}, {"b": 2}, {}),
        ({"b": 1}, {"a": 2, "b": 2}, {"a": 3, "b": 3, "c": 3}),
        ({"a": 1}, {"b": 2}, {"c": 3}),
        # null values
        ({}, {}, None),
        ({"a": 1}, {"a": 2}, None),
        ({"a": 1, "b": 1}, {"a": 2, "b": 2}, None),
        ({}, None, None),
        (None, None, None),
    ]

    fields = [
        StructField("c1", MapType(StringType(), IntegerType()), True),
        StructField("c2", MapType(StringType(), IntegerType()), True),
        StructField("c3", MapType(StringType(), IntegerType()), True),
    ]
    schema = StructType(fields)
    return spark.createDataFrame(data, schema).persist()


@pytest.mark.parametrize("perform", ["keep", "remove", "mark"])
def test_ensure_same_keys(df_input, perform):
    """Test EnsureSameKeys w/ and w/o 'output_column'."""
    input_columns = random.choice([["c1", "c2"], ["c1", "c2", "c3"]])
    n_input: int = df_input.count()
    output_column = "output" if perform == "mark" else None
    t = EnsureSameKeys(
        input_columns=input_columns,
        perform=perform,
        output_column=output_column,
    )
    df_out = t.transform(df_input)
    rows = df_out.collect()

    li_exp = _extract_expected_result(rows, input_columns)

    if perform == "keep":
        assert n_input > len(rows)
        assert all(li_exp)
    elif perform == "remove":
        assert n_input > len(rows)
        assert not any(li_exp)
    else:
        assert n_input == len(rows)
        li_chk = [getattr(row, "output") for row in rows]
        assert li_chk == li_exp


@pytest.mark.parametrize(
    "input_columns, glob, perform, output_column",
    [
        ["c1", None, "keep", None],  # wrong input_columns
        [None, "d*", "keep", None],  # wrong glob
        [None, "*", "keep", "NO"],  # output_column must not be provided
        [None, "*", "remove", "NO"],  # output_column must not be provided
        [None, "*", "mark", None],  # output_column must be provided
    ],
)
def test_ensure_same_keys_wrong(df_input, input_columns, glob, perform, output_column):
    """Test EnsureSameKeys passing wrong parameters."""
    with pytest.raises(AssertionError):
        t = EnsureSameKeys(
            input_columns=input_columns,
            glob=glob,
            perform=perform,
            output_column=output_column,
        )
        t.transform(df_input)
