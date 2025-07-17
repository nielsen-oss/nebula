"""Unit-test for AddPrefixSuffixToColumnNames."""

import pytest
from pyspark.sql.types import StringType, StructField, StructType

from nlsn.nebula.spark_transformers import AddPrefixSuffixToColumnNames
from nlsn.tests.auxiliaries import get_expected_columns


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    """Define test data."""
    fields = [
        StructField("a_1", StringType()),
        StructField("a_2", StringType()),
        StructField("ab_1", StringType()),
        StructField("ab_2", StringType()),
    ]

    data = [
        ("1", "11", "110", "312"),
        ("1", "12", "120", "234"),
    ]
    return spark.createDataFrame(data, schema=StructType(fields))


@pytest.mark.parametrize("columns", [None, [], "a_1", ["a_1", "a_2"]])
@pytest.mark.parametrize("prefix", [None, "pre_"])
@pytest.mark.parametrize("suffix", [None, "_post"])
@pytest.mark.parametrize("regex", [None, "^a", "^z"])
@pytest.mark.parametrize("glob", [None, "*", "", "a*"])
def test_add_prefix_suffix_to_column_names(
    df_input, prefix, suffix, columns, regex, glob
):
    """Test AddPrefixSuffixToColumnNames transformer."""
    if not prefix and not suffix:
        with pytest.raises(AssertionError):
            AddPrefixSuffixToColumnNames(
                columns=columns, regex=regex, glob=glob, prefix=prefix, suffix=suffix
            )
        return

    t = AddPrefixSuffixToColumnNames(
        columns=columns, regex=regex, glob=glob, prefix=prefix, suffix=suffix
    )
    df_out = t.transform(df_input)
    chk_cols = df_out.columns

    prefix = prefix if prefix else ""
    suffix = suffix if suffix else ""

    cols2rename = get_expected_columns(df_input, columns, regex, glob)
    exp_cols = [
        f"{prefix}{c}{suffix}" if c in cols2rename else c for c in df_input.columns
    ]

    assert chk_cols == exp_cols
