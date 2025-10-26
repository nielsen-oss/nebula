"""Unit-test for RegexReplace."""

import re
from typing import Dict, List

import pytest
from pyspark.sql import Row
from pyspark.sql.types import StringType, StructField, StructType

from nlsn.nebula.spark_transformers import RegexReplace

_INPUT: List[Dict[str, str]] = [
    {
        "input_string": "Hello, 123! How are 456 and 789?",
        "regex_pattern": r"\d+",
        "replacement": "x",
    },
    {
        "input_string": "Contact us at support@example.com or info@domain.net.",
        "regex_pattern": r"\S+@\S+",  # Matches email addresses
        "replacement": "EMAIL",
    },
    {
        "input_string": "abc123def456ghi789",
        "regex_pattern": r"[a-z]+",
        "replacement": "WORD",
    },
]


def _replace_with_regex(input_string, regex_pattern, replacement):
    return re.sub(regex_pattern, replacement, input_string)


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    """Define test data."""
    fields = []
    data = []
    for i, nd in enumerate(_INPUT):
        fields.append(StructField(f"col_{i}", StringType()))
        data.append(nd["input_string"])

    # 1 row dataframe
    schema = StructType(fields)
    return spark.createDataFrame([data], schema=schema).persist()


def test_regex_replace_single_column(df_input):
    """Test RegexReplace transformer with single column input."""
    n_input = len(_INPUT)
    for i in range(n_input):
        # Select the column to apply the transformer to
        col = f"col_{i}"

        nd = _INPUT[i]

        pattern = nd["regex_pattern"]
        replacement = nd["replacement"]
        t = RegexReplace(columns=col, pattern=pattern, replacement=replacement)
        df_out = t.transform(df_input)

        assert df_out.columns == df_input.columns

        collected: Row = df_out.collect()[0]
        dict_collected = collected.asDict()

        for k_col, value in dict_collected.items():
            if k_col == col:
                input_string = nd["input_string"]
                exp = _replace_with_regex(input_string, pattern, replacement)
                assert value == exp
            else:
                # Get the INPUT index
                idx = int(k_col.lstrip("col_"))
                assert value == _INPUT[idx]["input_string"]


def test_regex_replace_all_columns(df_input):
    """Test RegexReplace transformer on all the dataframe columns."""
    pattern = r"\d+"
    replacement = "NUM"
    t = RegexReplace(columns_glob="*", pattern=pattern, replacement=replacement)

    df_out = t.transform(df_input)

    assert df_out.columns == df_input.columns

    collected: Row = df_out.collect()[0]
    dict_collected = collected.asDict()

    for k_col, value in dict_collected.items():
        # Get the INPUT index
        idx = int(k_col.lstrip("col_"))
        input_string = _INPUT[idx]["input_string"]
        exp = _replace_with_regex(input_string, pattern, replacement)
        assert value == exp
