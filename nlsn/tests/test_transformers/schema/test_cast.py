"""Unit-test for Cast."""

from decimal import Decimal

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from nlsn.nebula.transformers.schema import Cast


class TestCastInitialization:
    """Test Cast transformer."""

    @pytest.mark.parametrize("cast", [[("col1", "int64")], "string"])
    def test_init_not_dict_raises(self, cast):
        """Test invalid initialization."""
        with pytest.raises(TypeError):
            Cast(cast=cast)

    def test_pandas(self):
        """Test Pandas."""
        df = pd.DataFrame({
            "int_col": ["1", "2", "3"],
            "float_col": [1, 2, 3],
            "str_col": [1.1, 2.2, 3.3],
            "bool_col": ["true", "false", "true"],
            "dont_cast": [0.1, 0.3, 0.5],
            "date_col": ["2023-01-01", "2023-01-02", "2023-01-03"],
        })

        cast = Cast(cast={
            "int_col": "int64",
            "float_col": "float64",
            "str_col": "str",
            "bool_col": "bool",
            "date_col": "datetime",
        })

        result = cast.transform(df)

        assert result["int_col"].dtype == "int64"
        assert result["float_col"].dtype == "float64"
        assert result["str_col"].dtype == "object"
        assert result["bool_col"].dtype == "bool"
        assert pd.api.types.is_datetime64_any_dtype(result["date_col"])

    def test_pandas_invalid(self):
        """Test that nested types raise error for Pandas."""
        df = pd.DataFrame({"col": [[1, 2], [3, 4]]})

        cast = Cast(cast={"col": "list[int64]"})

        with pytest.raises(ValueError):
            cast.transform(df)

    def test_polars(self):
        """Test casting all valid types in Polars."""
        df = pl.DataFrame({
            # Simple types
            "int_col": ["1", "2", "3"],
            "float_col": [1, 2, 3],
            "str_col": [1.1, 2.2, 3.3],
            "bool_col": [1, 0, 1],
            "int32_col": [1, 2, 3],
            "float32_col": [1.1, 2.2, 3.3],
            "uint32_col": [1, 2, 3],
            "uint64_col": [4, 5, 6],
            # Nested types
            "list_col": [["1", "2"], ["3", "4"], ["5", "6"]],
            "nested_list_col": [[["1", "2"]], [["3", "4"]], [["5", "6"]]],
            "array_col": [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]],
            "struct_col": [
                {"name": "Alice", "age": "30"},
                {"name": "Bob", "age": "25"},
                {"name": "Charlie", "age": "35"}
            ],
            "struct_no_braces_col": [
                {"x": "1", "y": "2"},
                {"x": "3", "y": "4"},
                {"x": "5", "y": "6"}
            ],
            "list_struct_col": [
                [{"id": "1", "val": "10"}],
                [{"id": "2", "val": "20"}],
                [{"id": "3", "val": "30"}]
            ],
            "struct_list_col": [
                {"id": "1", "values": ["10", "20"]},
                {"id": "2", "values": ["30", "40"]},
                {"id": "3", "values": ["50", "60"]}
            ],
        })

        cast = Cast(cast={
            # Simple types
            "int_col": "int64",
            "float_col": "float64",
            "str_col": "str",
            "bool_col": "bool",
            "int32_col": "int32",
            "float32_col": "float32",
            "uint32_col": "uint32",
            "uint64_col": "uint64",
            # Nested types
            "list_col": "list[int64]",
            "nested_list_col": "list[list[int64]]",
            "array_col": "array[int64, 3]",
            "struct_col": "struct[{name: str, age: int64}]",
            "struct_no_braces_col": "struct[x: int64, y: int64]",
            "list_struct_col": "list[struct[{id: int64, val: int64}]]",
            "struct_list_col": "struct[{id: int64, values: list[int64]}]",
        })

        result = cast.transform(df)

        # Assert simple types
        assert result["int_col"].dtype == pl.Int64
        assert result["float_col"].dtype == pl.Float64
        assert result["str_col"].dtype == pl.String
        assert result["bool_col"].dtype == pl.Boolean
        assert result["int32_col"].dtype == pl.Int32
        assert result["float32_col"].dtype == pl.Float32
        assert result["uint32_col"].dtype == pl.UInt32
        assert result["uint64_col"].dtype == pl.UInt64

        # Assert nested types
        assert result["list_col"].dtype == pl.List(pl.Int64)
        assert result["list_col"][0].to_list() == [1, 2]

        assert result["nested_list_col"].dtype == pl.List(pl.List(pl.Int64))

        assert result["array_col"].dtype == pl.Array(pl.Int64, 3)

        assert result["struct_col"].dtype == pl.Struct([
            pl.Field("name", pl.String),
            pl.Field("age", pl.Int64)
        ])

        assert result["struct_no_braces_col"].dtype == pl.Struct([
            pl.Field("x", pl.Int64),
            pl.Field("y", pl.Int64)
        ])

        assert result["list_struct_col"].dtype == pl.List(pl.Struct([
            pl.Field("id", pl.Int64),
            pl.Field("val", pl.Int64)
        ]))

        assert result["struct_list_col"].dtype == pl.Struct([
            pl.Field("id", pl.Int64),
            pl.Field("values", pl.List(pl.Int64))
        ])

    def test_polars_array_invalid_width_raises(self):
        """Test that array without width raises error."""
        df = pl.DataFrame({
            "col": [["1", "2"], ["3", "4"]]
        })

        cast = Cast(cast={"col": "array[int64]"})

        with pytest.raises(ValueError):
            cast.transform(df)

    def test_polars_array_non_integer_width_raises(self):
        """Test that array with non-integer width raises error."""
        df = pl.DataFrame({
            "col": [["1", "2"], ["3", "4"]]
        })
        cast = Cast(cast={"col": "array[int64, abc]"})
        with pytest.raises(ValueError):
            cast.transform(df)

    def test_polars_struct_missing_colon_raises(self):
        """Test that struct field without colon raises error."""
        df = pl.DataFrame({
            "col": [{"x": 1}, {"x": 2}]
        })

        cast = Cast(cast={"col": "struct[{x int64}]"})

        with pytest.raises(ValueError):
            cast.transform(df)

    def test_polars_unknown_simple_type_raises(self):
        """Test that unknown simple type raises error."""
        df = pl.DataFrame({"col": [1, 2, 3]})

        cast = Cast(cast={"col": "unknown_type"})

        with pytest.raises(ValueError):
            cast.transform(df)

    def test_polars_time_duration_types(self):
        """Test time and duration types in Polars.

        Note: Time and Duration types need to be provided in native format,
        not as strings, as Polars doesn't support direct string-to-time casting.
        """
        from datetime import time, timedelta

        df = pl.DataFrame({
            "time_col": [time(12, 30, 45), time(14, 15, 30), time(10, 0, 0)],
            "duration_col": [
                timedelta(microseconds=1000000),
                timedelta(microseconds=2000000),
                timedelta(microseconds=3000000)
            ],
        })

        # These columns already have the correct types, but we can cast to verify
        cast = Cast(cast={
            "time_col": "time",
            "duration_col": "duration"
        })
        result = cast.transform(df)

        assert result["time_col"].dtype == pl.Time
        assert result["duration_col"].dtype == pl.Duration

        # Verify the values are preserved
        assert result["time_col"][0] == time(12, 30, 45)
        assert result["duration_col"][0] == timedelta(microseconds=1000000)

    def test_spark(self, spark):
        """Test casting all valid types in Spark."""
        from pyspark.sql import Row

        df = spark.createDataFrame([
            Row(
                # Simple types
                int_col="1",
                float_col=1,
                str_col=1.1,
                bool_col="true",
                long_col="100",
                double_col=1.5,
                # Nested types
                array_col=["1", "2", "3"],
                struct_col=Row(name="Alice", age="30"),
                map_col={"key1": "1", "key2": "2"},
                nested_array_struct=[Row(id="1", val="10"), Row(id="2", val="20")],
                struct_with_array=Row(id="1", values=["10", "20"]),
            ),
            Row(
                int_col="2",
                float_col=2,
                str_col=2.2,
                bool_col="false",
                long_col="200",
                double_col=2.5,
                array_col=["4", "5", "6"],
                struct_col=Row(name="Bob", age="25"),
                map_col={"key3": "3", "key4": "4"},
                nested_array_struct=[Row(id="3", val="30"), Row(id="4", val="40")],
                struct_with_array=Row(id="2", values=["30", "40"]),
            ),
            Row(
                int_col="3",
                float_col=3,
                str_col=3.3,
                bool_col="true",
                long_col="300",
                double_col=3.5,
                array_col=["7", "8", "9"],
                struct_col=Row(name="Charlie", age="35"),
                map_col={"key5": "5", "key6": "6"},
                nested_array_struct=[Row(id="5", val="50"), Row(id="6", val="60")],
                struct_with_array=Row(id="3", values=["50", "60"]),
            ),
        ])

        cast = Cast(cast={
            # Simple types
            "int_col": "int",
            "float_col": "float",
            "str_col": "string",
            "bool_col": "boolean",
            "long_col": "long",
            "double_col": "double",
            # Nested types
            "array_col": "array<int>",
            "struct_col": "struct<name:string,age:int>",
            "map_col": "map<string,int>",
            "nested_array_struct": "array<struct<id:int,val:int>>",
            "struct_with_array": "struct<id:int,values:array<int>>",
        })

        result = cast.transform(df)
        dtypes_dict = dict(result.dtypes)

        # Assert simple types
        assert dtypes_dict["int_col"] == "int"
        assert dtypes_dict["float_col"] == "float"
        assert dtypes_dict["str_col"] == "string"
        assert dtypes_dict["bool_col"] == "boolean"
        assert dtypes_dict["long_col"] == "bigint"  # long becomes bigint in Spark
        assert dtypes_dict["double_col"] == "double"

        # Assert nested types
        assert dtypes_dict["array_col"] == "array<int>"
        assert "struct<name:string,age:int>" in dtypes_dict["struct_col"]
        assert "map<string,int>" in dtypes_dict["map_col"]
        assert "array<struct<id:int,val:int>>" in dtypes_dict["nested_array_struct"]
        assert "struct<id:int,values:array<int>>" in dtypes_dict["struct_with_array"]

        # Verify actual values are cast correctly
        first_row = result.collect()[0]
        assert first_row["int_col"] == 1
        assert first_row["array_col"] == [1, 2, 3]
        assert first_row["struct_col"]["name"] == "Alice"
        assert first_row["struct_col"]["age"] == 30
        assert first_row["map_col"]["key1"] == 1

    def test_spark_timestamp_decimal(self, spark):
        """Test casting timestamp and decimal types in Spark."""
        df = spark.createDataFrame([
            ("2023-01-01 10:30:00", 1.123456),
            ("2023-01-02 14:15:30", 2.654321),
            ("2023-01-03 08:45:00", 3.987654),
        ], ["timestamp_col", "decimal_col"])

        cast = Cast(cast={
            "timestamp_col": "timestamp",
            "decimal_col": "decimal(10,2)"
        })

        result = cast.transform(df)
        dtypes_dict = dict(result.dtypes)

        assert dtypes_dict["timestamp_col"] == "timestamp"
        assert "decimal(10,2)" in dtypes_dict["decimal_col"]

        first_row = result.collect()[0]
        assert first_row["decimal_col"] == Decimal("1.12")

    def test_narwhals_pandas(self):
        """Test Cast with narwhals-wrapped Pandas DataFrame."""
        df_pd = pd.DataFrame({
            "int_col": ["1", "2", "3"],
            "float_col": [1, 2, 3],
        })
        df_nw = nw.from_native(df_pd)

        cast = Cast(cast={"int_col": "int64", "float_col": "float64"})
        result_nw = cast.transform(df_nw)

        # Should return narwhals DataFrame
        assert isinstance(result_nw, nw.DataFrame)

        # Convert back and check types
        result_pd = nw.to_native(result_nw)
        assert result_pd["int_col"].dtype == "int64"
        assert result_pd["float_col"].dtype == "float64"

    def test_narwhals_polars_simple(self):
        """Test Cast with narwhals-wrapped Polars DataFrame (simple types)."""
        df_pl = pl.DataFrame({
            "int_col": ["1", "2", "3"],
            "str_col": [1, 2, 3],
        })
        df_nw = nw.from_native(df_pl)

        cast = Cast(cast={"int_col": "int64", "str_col": "str"})
        result_nw = cast.transform(df_nw)

        assert isinstance(result_nw, nw.DataFrame)

        result_pl = nw.to_native(result_nw)
        assert result_pl["int_col"].dtype == pl.Int64
        assert result_pl["str_col"].dtype == pl.String

    def test_narwhals_polars_nested(self):
        """Test Cast with narwhals-wrapped Polars DataFrame (nested types)."""
        df_pl = pl.DataFrame({
            "col": [["1", "2"], ["3", "4"]]
        })
        df_nw = nw.from_native(df_pl)

        cast = Cast(cast={"col": "list[int64]"})
        result_nw = cast.transform(df_nw)

        # Should fall back to native and return narwhals
        assert isinstance(result_nw, nw.DataFrame)

        result_pl = nw.to_native(result_nw)
        assert result_pl["col"].dtype == pl.List(pl.Int64)
