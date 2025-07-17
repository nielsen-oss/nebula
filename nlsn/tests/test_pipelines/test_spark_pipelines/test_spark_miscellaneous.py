"""Miscellaneous spark tests."""

from pyspark.ml.feature import IndexToString as SparkNativeIndexToString

from nlsn.nebula.pipelines.transformer_type_util import is_transformer


def test_is_transformer_spark_native():
    """Test 'is_transformer' function with 'pyspark.ml' native transformer."""
    t = SparkNativeIndexToString(inputCol="indexed", outputCol="label2")
    assert is_transformer(t)
