"""Unit-test for ReplaceWithMap and MapWithFallback transformers."""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from nlsn.nebula.spark_transformers import MapWithFallback, ReplaceWithMap
from nlsn.nebula.spark_transformers.collection import _assert_no_null_keys


class TestAssertNoNullKeys:
    def test_no_null_keys(self):
        """Valid input."""
        d = {"a": 1, "b": 2, "c": None}
        _assert_no_null_keys(d)

    def test_null_key(self):
        """Null key."""
        d = {None: 1, "b": 2}
        with pytest.raises(KeyError):
            _assert_no_null_keys(d)


@pytest.fixture(scope="module", name="df_input")
def _get_df_input(spark):
    schema = StructType([StructField("original_string", StringType(), True)])

    data = ["a", "a1", "a", "a1", "b", "b", "b1", "c1", "c1", "c1", "   ", None]
    nested_data = [[i] for i in data]

    df = spark.createDataFrame(nested_data, schema=schema)
    return df.withColumn("input_string", F.col("original_string")).persist()


def _assert(t, df, output_col, mapping, use_default: bool, default=None):
    df_chk = t.transform(df)
    df_pd = df_chk.toPandas()
    output_col_pandas = output_col if output_col else "input_string"

    for _, row in df_pd.iterrows():
        input_value = row["original_string"]
        chk = row[output_col_pandas]
        fallback_value = default if use_default else input_value
        exp = mapping.get(input_value, fallback_value)
        assert exp == chk, f'expected: "{exp}", found "{chk}"'


@pytest.mark.parametrize("mapping", [{"b": "b_map"}, {"b": "b_map", "c1": None}])
@pytest.mark.parametrize("output_col", [None, "output"])
def test_map_with_fallback(df_input, mapping, output_col):
    """Test MapWithFallback transformer."""
    default = "default"
    t = MapWithFallback(
        input_col="input_string",
        mapping=mapping,
        output_col=output_col,
        default=default,
    )
    _assert(t, df_input, output_col, mapping, use_default=True, default=default)


@pytest.mark.parametrize("output_col", [None, "output"])
def test_replace_with_map(df_input, output_col):
    """Test ReplaceWithMap transformer."""
    replace = {"a": "sub_a", "b1": "sub_b", "c1": None, "z": "Z"}
    t = ReplaceWithMap(input_col="input_string", replace=replace, output_col=output_col)
    _assert(t, df_input, output_col, replace, use_default=False)
