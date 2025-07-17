"""Unit-test for spark RenameColumns."""

import pytest
from pyspark.sql.types import FloatType, StringType, StructField, StructType

from nlsn.nebula.spark_transformers import RenameColumns
from nlsn.tests.test_transformers.test_columns._shared import SharedRenameColumns


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    columns = SharedRenameColumns.columns
    fields = [
        StructField(columns[0], StringType(), True),
        StructField(columns[1], FloatType(), True),
    ]
    schema = StructType(fields)
    return spark.createDataFrame(SharedRenameColumns.data, schema=schema)


@pytest.mark.parametrize(*SharedRenameColumns.params)
def test_rename_columns(
    df_input, mapping, columns, columns_renamed, regex_pattern, regex_replacement, exp
):
    """Test RenameColumns transformer."""
    t = RenameColumns(
        mapping=mapping,
        columns=columns,
        columns_renamed=columns_renamed,
        regex_pattern=regex_pattern,
        regex_replacement=regex_replacement,
    )
    df_out = t.transform(df_input)
    assert df_out.columns == exp

    chk_str = {i[0] for i in df_out.select(exp[0]).collect()}
    chk_flt = {i[0] for i in df_out.select(exp[1]).collect()}

    assert chk_str == SharedRenameColumns.set_str
    assert chk_flt == SharedRenameColumns.set_flt
