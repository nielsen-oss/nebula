"""Dataframe Schema Operations: Casting, Nullability, etc.

These transformers are not supposed to act directly on values, but
some operations (i.e., cast) can affect them.
"""

import narwhals as nw

from nlsn.nebula.base import Transformer
from nlsn.nebula.df_types import get_dataframe_type

__all__ = [
    "Cast",
]


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

    def _transform_nw(self, df: nw.DataFrame) -> nw.DataFrame:
        """Narwhals implementation for simple types."""
        # Map common type names to narwhals types
        type_map = {
            'int': nw.Int64,
            'int64': nw.Int64,
            'int32': nw.Int32,
            'float': nw.Float64,
            'float64': nw.Float64,
            'float32': nw.Float32,
            'str': nw.String,
            'string': nw.String,
            'bool': nw.Boolean,
            'boolean': nw.Boolean,
            'date': nw.Date,
            'datetime': nw.Datetime,
            'duration': nw.Duration,
            'timedelta': nw.Duration,
        }

        # Check if any Spark-specific types requested
        spark_types = ['array', 'struct', 'map']
        has_nested = any(
            any(st in dtype.lower() for st in spark_types)
            for dtype in self._cast.values()
        )

        set_types = set(self._cast.values())

        if has_nested or not set_types.issubset(type_map):
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
                if target_type in type_map:
                    exprs.append(df[col].cast(type_map[target_type]).alias(col))
                else:
                    raise ValueError(
                        f"Unknown type '{self._cast[col]}' for column '{col}'. "
                        f"Supported: {list(type_map.keys())}"
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

        # Simple type mappings
        simple_types = {
            'int': pl.Int64,
            'int64': pl.Int64,
            'int32': pl.Int32,
            'int16': pl.Int16,
            'int8': pl.Int8,
            'uint64': pl.UInt64,
            'uint32': pl.UInt32,
            'uint16': pl.UInt16,
            'uint8': pl.UInt8,
            'float': pl.Float64,
            'float64': pl.Float64,
            'float32': pl.Float32,
            'str': pl.String,
            'string': pl.String,
            'utf8': pl.String,
            'bool': pl.Boolean,
            'boolean': pl.Boolean,
            'date': pl.Date,
            'datetime': pl.Datetime,
            'time': pl.Time,
            'duration': pl.Duration,
        }

        # Check for simple types first
        if dtype_str in simple_types:
            return simple_types[dtype_str]

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
