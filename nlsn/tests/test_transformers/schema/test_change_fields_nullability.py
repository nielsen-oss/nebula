"""Unit-test for ChangeFieldsNullability."""

import pytest
from py4j.protocol import Py4JJavaError
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, StringType, StructField, StructType

from nlsn.nebula.spark_transformers import ChangeFieldsNullability


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    """Define test data."""
    fields = [  # Two nullable fields
        StructField("with_null", StringType(), True),
        StructField("without_null", FloatType(), True),
    ]
    schema = StructType(fields)
    df = spark.createDataFrame([["a", 0.0], [None, 1.0]], schema=schema)

    # add non-nullable field
    return df.withColumn("non_nullable", F.lit("placeholder"))


def test_change_fields_nullability_to_nullable(df_input):
    """Test ChangeFieldsNullability transformer.

    Convert all the fields to nullable.
    """
    t = ChangeFieldsNullability(
        nullable=True,
        glob="*",
        assert_non_nullable=True,  # this parameter is ignored
        persist=True,  # this parameter is ignored
    )

    df_out = t.transform(df_input)

    assert df_out.columns == df_input.columns

    # assert all fields are nullable
    assert all(i.nullable for i in df_out.schema)


def test_change_fields_nullability_to_non_nullable(df_input):
    """Test ChangeFieldsNullability transformer.

    Convert all the fields to non-nullable.
    The field 'with_null' cannot be non-nullable, but it will pass the test
    since no eager operations are triggered.
    """
    t = ChangeFieldsNullability(
        nullable=False,
        glob="*",
        assert_non_nullable=False,
        persist=False,
    )

    df_out = t.transform(df_input)

    assert df_out.columns == df_input.columns

    # assert all fields are non-nullable
    assert all(not i.nullable for i in df_out.schema)


def test_change_fields_nullability_to_non_nullable_error(df_input):
    """Test ChangeFieldsNullability transformer.

    Convert the field 'with_null' to non-nullable.
    This field cannot be non-nullable and must raise a ValueError
    since an eager operation is triggered.
    """
    t = ChangeFieldsNullability(
        nullable=False,
        columns="with_null",
        assert_non_nullable=True,
        persist=True,
    )

    with pytest.raises(Py4JJavaError):
        t.transform(df_input)
