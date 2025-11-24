"""Schema transformation operations.

Transformers for modifying dataframe schema:
- Type conversion of existing columns
- Addition of new typed columns
"""

from typing import Any

import narwhals as nw

from nlsn.nebula.base import Transformer
from nlsn.nebula.df_types import get_dataframe_type
from nlsn.nebula.transformers._constants import NW_TYPES, PANDAS_NULLABLE_INTEGERS, PL_TYPES

__all__ = ["AddTypedColumns", "Cast"]


class AddTypedColumns(Transformer):
    def __init__(
            self,
            *,
            columns: list[tuple[str, str]] | dict[str, Any] | None,
    ):
        """Add typed columns if they do not exist in the DF.

        Supports simple types only (int64, float64, str, bool, date, datetime).
        For complex nested types (array, struct, map), use:
          1. AddTypedColumns to create as 'str'
          2. Cast transformer to convert to desired nested type

        Args:
            columns: Column specifications
                - list[tuple]: [(name, dtype), ...]
                - dict: {name: dtype} or {name: {"type": dtype, "value": val}}
                - None/empty: pass-through

        Example:
            # Simple types - works directly
            AddTypedColumns(columns={"age": "int64", "name": "str"})

            # Complex types - use Cast afterward
            AddTypedColumns(columns={"data": "str"})  # Create as string
            Cast(cast={"data": "array<string>"})      # Convert to array
        """
        super().__init__()

        self._columns: dict[str, dict[str, Any]]
        self._skip: bool = False

        if not columns:
            self._skip = True
            return

        if isinstance(columns, dict):
            self._assert_keys_strings(columns)
            self._check_default_value(columns)
            # Sort for repeatability
            columns_raw = sorted(columns.items())

        else:
            if not isinstance(columns, (tuple, list)):
                msg = '"columns" must be <list> | <tuple> | <dict <str, str>>'
                raise AssertionError(msg)
            unique_len = {len(i) for i in columns}
            if unique_len != {2}:
                msg = 'If "columns" is a <list> | <tuple> it must contain '
                msg += "2-element iterables"
                raise AssertionError(msg)
            columns_raw = columns

        # Convert 'columns_raw' into a dictionary like:
        # {"column_name": {"type": datatype, "value": value}}
        self._columns = {}
        for k, obj in columns_raw:
            if isinstance(obj, dict):
                datatype = obj["type"]
                value = obj["value"]
            else:
                datatype = obj
                value = None

            self._columns.update({k: {"type": datatype, "value": value}})

        self._validate_simple_types()

    def _validate_simple_types(self):
        """Ensure only simple types are requested."""
        complex_markers = ['array', 'struct', 'map', 'list', '[', '{']

        for col_name, spec in self._columns.items():
            dtype_str = spec["type"]
            dtype_lower = dtype_str.lower()

            # Check for complex type markers
            if any(marker in dtype_lower for marker in complex_markers):
                raise ValueError(
                    f"Column '{col_name}': AddTypedColumns only supports simple types.\n"
                    f"Found: '{dtype_str}'\n\n"
                    f"For complex types, use a two-step approach:\n"
                    f"  1. AddTypedColumns(columns={{'{col_name}': 'str'}})\n"
                    f"  2. Cast(cast={{'{col_name}': '{dtype_str}'}})"
                )

            # Check if type is recognized
            if dtype_lower not in NW_TYPES:
                supported = ', '.join(sorted(set(NW_TYPES.keys())))
                raise ValueError(
                    f"Column '{col_name}': Unsupported type '{dtype_str}'.\n"
                    f"Supported types: {supported}"
                )

    @staticmethod
    def _assert_keys_strings(dictionary):
        for k in dictionary.keys():
            if not isinstance(k, str):
                msg = "All keys in the dictionary must be <string>"
                raise AssertionError(msg)

    @staticmethod
    def _check_default_value(dictionary):
        _allowed = {"type", "value"}

        for nd in dictionary.values():
            if not isinstance(nd, dict):
                continue

            set_keys = nd.keys()
            if set_keys != _allowed:
                msg = f"Allowed keys in nested dictionary: {_allowed}. "
                msg += f"Found: {set_keys}."
                raise AssertionError(msg)

    def _transform_nw(self, nw_df):
        if self._skip:
            return nw_df

        backend: str = get_dataframe_type(nw.to_native(nw_df))
        is_pandas: bool = backend == "pandas"

        current_cols: set[str] = set(nw_df.columns)
        new_cols_exprs = []
        pandas_nullable_int = {}
        for name, nd in self._columns.items():
            if name in current_cols:  # Do not add new col if already exist
                continue

            value = nd["value"]
            type_name: str = nd["type"]

            if is_pandas and (value is None) and (type_name in PANDAS_NULLABLE_INTEGERS):
                pandas_nullable_int[name] = PANDAS_NULLABLE_INTEGERS[type_name]
                continue

            data_type = NW_TYPES[type_name]
            new_cols_exprs.append(nw.lit(value).cast(data_type).alias(name))

        if new_cols_exprs:
            nw_df = nw_df.with_columns(new_cols_exprs)

        if pandas_nullable_int:
            import pandas as pd

            df_pd = nw.to_native(nw_df)
            for name, dtype in pandas_nullable_int.items():
                df_pd[name] = pd.Series([None] * len(df_pd), dtype=dtype)
            nw_df = nw.from_native(df_pd)

        return nw_df


class Cast(Transformer):
    def __init__(self, *, cast: dict[str, str]):
        """Cast columns to specified data types.

        Args:
            cast (dict[str, str]): Column name to target type mapping.
                For simple types, use standard names: 'int64', 'float64', 'str', 'bool'
                For Spark nested types, use Spark DDL strings: 'array<string>', 'struct<...>'
        """
        if not isinstance(cast, dict):
            raise TypeError("'cast' must be a dictionary")

        super().__init__()
        self._cast: dict[str, str] = cast

    def _transform_nw(self, df):
        """Narwhals implementation for simple types."""
        # Check if any Spark-specific types requested
        spark_types = ['array', 'struct', 'map']
        has_nested = any(
            any(st in dtype.lower() for st in spark_types)
            for dtype in self._cast.values()
        )

        set_types = set(self._cast.values())

        if has_nested or not set_types.issubset(NW_TYPES):
            # Fall back to native implementation for nested types
            df_native = nw.to_native(df)
            df_type = get_dataframe_type(df_native)

            if df_type == "pandas":
                raise ValueError(
                    "Pandas does not support nested types (array, struct, map, list). "
                    f"Cast requested: {self._cast}. Use Polars or Spark instead."
                )

            if df_type == "polars":
                return nw.from_native(self._transform_polars(df_native))

            if df_type == "spark":
                return nw.from_native(self._transform_spark(df_native))

            raise ValueError(
                f"Backend '{df_type}' does not support nested type casting. "
                f"Supported: polars, spark. Cast: {self._cast}"
            )

        # Cast columns
        exprs = []
        for col in df.columns:
            if col in self._cast:
                target_type = self._cast[col].lower()
                if target_type in NW_TYPES:
                    exprs.append(df[col].cast(NW_TYPES[target_type]).alias(col))
                else:
                    raise ValueError(
                        f"Unknown type '{self._cast[col]}' for column '{col}'. "
                        f"Supported: {list(NW_TYPES.keys())}"
                    )
            else:
                exprs.append(df[col])

        return df.select(exprs)

    def _transform_spark(self, df):
        """Spark-specific implementation supporting nested types."""
        import pyspark.sql.functions as F

        cols = [
            F.col(c).cast(self._cast[c]).alias(c) if c in self._cast else c
            for c in df.columns
        ]
        return df.select(*cols)

    def _parse_polars_dtype(self, dtype_str: str):
        """Parse a dtype string and return corresponding Polars dtype.

        Supports:
        - Simple types: 'int64', 'float64', 'str', 'bool', 'date', 'datetime'
        - List types: 'list[int64]', 'list[str]', etc.
        - Array types: 'array[int64, 3]' (fixed width)
        - Struct types: 'struct[{name: str, age: int64}]'

        Args:
            dtype_str: String representation of the dtype

        Returns:
            Polars DataType object
        """
        import polars as pl

        dtype_str = dtype_str.strip().lower()

        # Check for simple types first
        if dtype_str in PL_TYPES:
            return PL_TYPES[dtype_str]

        # Parse List types: list[inner_type]
        if dtype_str.startswith('list[') and dtype_str.endswith(']'):
            inner_str = dtype_str[5:-1]  # Extract content between list[ and ]
            inner_dtype = self._parse_polars_dtype(inner_str)
            return pl.List(inner_dtype)

        # Parse Array types: array[inner_type, width]
        if dtype_str.startswith('array[') and dtype_str.endswith(']'):
            content = dtype_str[6:-1]  # Extract content between array[ and ]
            parts = content.rsplit(',', 1)  # Split on last comma

            if len(parts) != 2:
                raise ValueError(
                    f"Array type must specify width: 'array[type, width]', got '{dtype_str}'"
                )

            inner_str = parts[0].strip()
            try:
                width = int(parts[1].strip())
            except ValueError:
                raise ValueError(
                    f"Array width must be an integer, got '{parts[1].strip()}'"
                )

            inner_dtype = self._parse_polars_dtype(inner_str)
            return pl.Array(inner_dtype, width)

        # Parse Struct types: struct[{field1: type1, field2: type2}]
        if dtype_str.startswith('struct[') and dtype_str.endswith(']'):
            content = dtype_str[7:-1].strip()  # Extract content between struct[ and ]

            # Remove outer braces if present
            if content.startswith('{') and content.endswith('}'):
                content = content[1:-1].strip()

            # Parse field definitions
            fields = []
            if content:  # Not an empty struct
                # Split on commas not inside nested brackets
                field_strs = self._split_struct_fields(content)

                for field_str in field_strs:
                    field_str = field_str.strip()
                    if ':' not in field_str:
                        raise ValueError(
                            f"Struct field must have format 'name: type', got '{field_str}'"
                        )

                    name, type_str = field_str.split(':', 1)
                    name = name.strip()
                    type_str = type_str.strip()

                    field_dtype = self._parse_polars_dtype(type_str)
                    fields.append(pl.Field(name, field_dtype))

            return pl.Struct(fields)

        raise ValueError(
            f"Unknown dtype string '{dtype_str}'. "
            f"Supported formats: simple types, list[type], array[type, width], "
            f"struct[{{field1: type1, field2: type2}}]"
        )

    @staticmethod
    def _split_struct_fields(content: str) -> list[str]:
        """Split struct field definitions, respecting nested brackets.

        Example: 'a: int64, b: list[str], c: struct[{x: int}]'
                 -> ['a: int64', 'b: list[str]', 'c: struct[{x: int}]']
        """
        fields = []
        current = []
        depth = 0

        for char in content:
            if char in '[{':
                depth += 1
                current.append(char)
            elif char in ']}':
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0:
                # Top-level comma - this is a field separator
                fields.append(''.join(current))
                current = []
            else:
                current.append(char)

        if current:  # last field
            fields.append(''.join(current))

        return fields

    def _transform_polars(self, df):
        """Polars-specific implementation supporting nested types."""
        import polars as pl

        # Parse cast dict and build dtype objects
        exprs = []
        for col in df.columns:
            if col in self._cast:
                target_type_str = self._cast[col]

                # Parse and construct the Polars dtype
                dtype = self._parse_polars_dtype(target_type_str)

                exprs.append(pl.col(col).cast(dtype).alias(col))
            else:
                exprs.append(pl.col(col))

        return df.select(exprs)
