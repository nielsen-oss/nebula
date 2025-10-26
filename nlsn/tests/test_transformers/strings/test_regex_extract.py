"""Unit-test for RegexExtract."""

import re
from typing import Optional

import pytest
from pyspark.sql.types import StringType, StructField, StructType

from nlsn.nebula.spark_transformers import RegexExtract

_INPUT_COL: str = "regex_col"
_DATA = [
    ["string to parse"],
    ["some other string"],
    ["a third string with more words"],
    ["tooshort"],
]


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    """Define test data."""
    fields = [
        StructField(name=_INPUT_COL, dataType=StringType(), nullable=True),
    ]

    schema = StructType(fields)
    return spark.createDataFrame(data=_DATA, schema=schema).persist()


@pytest.mark.parametrize(
    "input_col, output_col",
    [
        ("c1", None),
        ("c1", "c1"),
    ],
)
def test_regex_extract_wrong_columns(input_col: str, output_col: Optional[str]):
    """Test RegexExtract transformer with wrong column parameters."""
    with pytest.raises(ValueError):
        RegexExtract(
            input_col=input_col,
            pattern="...",
            output_col=output_col,
            drop_input_col=True,
        )


def test_regex_extract_wrong_extract():
    """Test RegexExtract transformer with wrong extract parameter."""
    with pytest.raises(ValueError):
        RegexExtract(
            input_col="c1",
            pattern="...",
            extract=-1,
        )


@pytest.mark.parametrize(
    "output_col, extract, drop_input_col",
    [(None, 0, False), ("regex_extract", 2, True)],
)
def test_regex_extract(df_input, output_col: str, extract: int, drop_input_col: bool):
    """Test RegexExtract transformer."""
    pattern: str = r"(\w+)\s(\w+)"

    t = RegexExtract(
        input_col=_INPUT_COL,
        output_col=output_col,
        pattern=pattern,
        extract=extract,
        drop_input_col=drop_input_col,
    )
    df_chk = t.transform(df_input)

    n_cols_exp = 1
    if output_col:
        n_cols_exp += 1
    if drop_input_col:
        n_cols_exp -= 1

    assert len(df_chk.columns) == n_cols_exp

    col_chk = output_col if output_col else _INPUT_COL
    collected = [i[0] for i in df_chk.select(col_chk).collect()]

    assert len(collected) == len(_DATA)

    set_chk = set(collected)
    li_data = [i[0] for i in _DATA]

    set_exp = set()
    for string in li_data:
        match = re.match(pattern, string)
        if match is None:  # If no match add empty string
            set_exp.add("")
        else:
            set_exp.add(match.group(extract))

    assert set_chk == set_exp
