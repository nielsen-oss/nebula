"""Unit-test for ColumnsToLowercase."""

from typing import List

import pytest
from pyspark.sql.types import StringType, StructField, StructType

from nlsn.nebula.spark_transformers import ColumnsToLowercase


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    """Define test data."""
    fields = [
        StructField("column_1", StringType(), True),
        StructField("COLUMN_2", StringType(), True),
        StructField("Column_3_with_spaces  ", StringType(), True),
    ]

    data = [
        ("1", "11", "110"),
        ("1", "12", "120"),
    ]
    return spark.createDataFrame(data, schema=StructType(fields))


_params = [
    ("column_1", None, False),
    (["column_1", "Column_3_with_spaces  "], None, True),
    (None, "*", True),
    (None, "*", False),
]


@pytest.mark.parametrize("columns, glob, trim", _params)
def test_columns_to_lowercase(df_input, columns, glob: str, trim: bool):
    """Test ColumnsToLowercase transformer."""
    t = ColumnsToLowercase(columns=columns, glob=glob, trim=trim)
    df_chk = t.transform(df_input)
    cols_chk = df_chk.columns

    # In this unittest 'regex' is not considered and 'glob' is used only w/ '*'
    req_columns: List[str] = []
    if columns:
        req_columns = [req_columns] if isinstance(columns, str) else columns
    elif glob == "*":
        req_columns = df_input.columns
    else:
        raise NotImplementedError("Not implemented test configuration")

    # The order must be preserved.
    cols_exp: List[str] = []
    new_col: str
    for c in df_input.columns:
        if c in req_columns:
            new_col = c.lower()
            if trim:
                new_col = new_col.strip()
        else:
            new_col = c
        cols_exp.append(new_col)

    assert cols_chk == cols_exp
