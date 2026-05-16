"""Type and schema operations."""

import narwhals as nw

from nebula.auxiliaries import validate_keys
from nebula.base import Transformer
from nebula.df_types import get_dataframe_type
from nebula.transformers._constants import NW_TYPES, PANDAS_NULLABLE_INTEGERS, PL_TYPES

__all__ = ["AddLiterals", "Cast"]

_TIME_UNITS = ("us", "ms", "ns")
_TIME_DTYPE_PREFIXES = ("datetime", "duration", "timedelta")


def _split_time_unit(dtype_str: str) -> tuple[str, str | None]:
    """Return (prefix, time_unit) for strings like 'datetime[us]' / 'duration[ms]'.

    If the string is not in the bracketed form, returns (dtype_str, None).
    Raises ValueError if the prefix is one of datetime/duration/timedelta but the
    time unit is not one of us/ms/ns.
    """
    if not (dtype_str.endswith("]") and "[" in dtype_str):
        return dtype_str, None

    prefix, _, rest = dtype_str.partition("[")
    if prefix not in _TIME_DTYPE_PREFIXES:
        return dtype_str, None

    unit = rest[:-1].strip()
    if unit not in _TIME_UNITS:
        raise ValueError(f"Invalid time unit '{unit}' for '{prefix}'. Must be one of: {list(_TIME_UNITS)}")
    return prefix, unit


def _parse_nw_dtype(dtype_str: str):
    """Parse a dtype string into a narwhals DType, or return None if unsupported.

    Supports:
    - Simple types listed in NW_TYPES (e.g. 'int64', 'float64', 'str', 'datetime').
    - Datetime/Duration with a time unit: 'datetime[us]', 'datetime[ms]',
      'datetime[ns]', 'duration[us]', 'duration[ms]', 'duration[ns]'.
      'timedelta' is accepted as an alias for 'duration'.
    """
    dtype_str = dtype_str.strip().lower()

    if dtype_str in NW_TYPES:
        return NW_TYPES[dtype_str]

    prefix, unit = _split_time_unit(dtype_str)
    if unit is None:
        return None

    if prefix == "datetime":
        return nw.Datetime(time_unit=unit)
    return nw.Duration(time_unit=unit)


class AddLiterals(Transformer):
    def __init__(self, *, data: list[dict]):
        """Add literal value columns to DataFrame.

        Args:
            data: List of column specifications, each containing:
                - value: The literal value to add
                - alias: Column name (required, must be string)
                - cast: Optional type to cast to (must be in NW_TYPES)
        """
        exprs: list[nw.Expr] = []
        exprs_pandas: list[nw.Expr] = []
        pandas_nullable_int: dict[str, str] = {}

        for i, row in enumerate(data):
            validate_keys(f"data[{i}]", row, mandatory={"alias"}, optional={"value", "cast"})

            alias = row["alias"]
            value = row.get("value")
            cast = row.get("cast")

            # Validate alias is string
            if not isinstance(alias, str):
                raise TypeError(f"'alias' must be string, got {type(alias).__name__}")

            # Validate cast type
            if cast and cast not in NW_TYPES:
                nw_types = sorted(NW_TYPES.keys())
                raise ValueError(f"Invalid cast type '{cast}'. Must be one of: {nw_types}")

            el_nw = nw.lit(value)
            el_pd = nw.lit(value)
            if cast:
                el_nw = el_nw.cast(NW_TYPES[cast])
                if cast not in {"str", "string", "utf8"}:
                    el_pd = el_pd.cast(NW_TYPES[cast])

            el_nw = el_nw.alias(alias)
            el_pd = el_pd.alias(alias)
            exprs.append(el_nw)

            # Special handling for pandas nullable integers
            if (value is None) and cast and (cast in PANDAS_NULLABLE_INTEGERS):
                pandas_nullable_int[alias] = PANDAS_NULLABLE_INTEGERS[cast]
                continue

            exprs_pandas.append(el_pd)

        super().__init__()
        self._exprs: list[nw.Expr] = exprs
        self._exprs_pandas: list[nw.Expr] = exprs_pandas
        self._pandas_nullable_int: dict[str, str] = pandas_nullable_int

    def _transform_nw(self, nw_df):
        # Add regular columns
        backend: str = get_dataframe_type(nw.to_native(nw_df))
        exprs = self._exprs_pandas if backend == "pandas" else self._exprs

        if exprs:
            nw_df = nw_df.with_columns(*exprs)

        # Handle pandas nullable integers
        if self._pandas_nullable_int:
            if backend == "pandas":
                import pandas as pd

                df_pd = nw.to_native(nw_df)
                for name, dtype in self._pandas_nullable_int.items():
                    df_pd[name] = pd.Series([None] * len(df_pd), dtype=dtype)
                nw_df = nw.from_native(df_pd)

        return nw_df


class Cast(Transformer):
    def __init__(self, *, cast: dict[str, str]):
        """Cast columns to specified data types.

        Args:
            cast (dict[str, str]): Column name to target type mapping.
                For simple types, use standard names: 'int64', 'float64', 'str', 'bool'.
                For datetime/duration with an explicit time unit, use bracket syntax:
                    'datetime[us]', 'datetime[ms]', 'datetime[ns]',
                    'duration[us]', 'duration[ms]', 'duration[ns]'
                ('timedelta' is accepted as an alias for 'duration').
                For Spark nested types, use Spark DDL strings: 'array<string>', 'struct<...>'.
        """
        if not isinstance(cast, dict):
            raise TypeError("'cast' must be a dictionary")

        super().__init__()
        self._cast: dict[str, str] = cast

    def _transform_nw(self, df):
        """Narwhals implementation for types expressible through narwhals."""
        parsed_nw: dict[str, nw.dtypes.DType] = {}
        for col, dtype_str in self._cast.items():
            nw_dtype = _parse_nw_dtype(dtype_str)
            if nw_dtype is None:
                parsed_nw = None
                break
            parsed_nw[col] = nw_dtype

        if parsed_nw is None:
            # Fall back to native implementation (nested types, polars-only types).
            df_native = nw.to_native(df)
            df_type = get_dataframe_type(df_native)

            if df_type in {"pandas", "duckdb"}:
                raise ValueError(
                    f"{df_type.capitalize()} does not support the requested cast types. "
                    f"Cast requested: {self._cast}. Use Polars or Spark instead."
                )

            if df_type == "polars":
                return nw.from_native(self._transform_polars(df_native))

            if df_type == "spark":
                return nw.from_native(self._transform_spark(df_native))

            raise ValueError(  # pragma: no cover
                f"Backend '{df_type}' does not support these cast types. Supported: polars, spark. Cast: {self._cast}"
            )

        exprs = [nw.col(col).cast(parsed_nw[col]).alias(col) if col in parsed_nw else nw.col(col) for col in df.columns]
        return df.select(exprs)

    def _transform_spark(self, df):
        """Spark-specific implementation supporting nested types."""
        import pyspark.sql.functions as F

        cols = [F.col(c).cast(self._cast[c]).alias(c) if c in self._cast else c for c in df.columns]
        return df.select(*cols)

    def _parse_polars_dtype(self, dtype_str: str):
        """Parse a dtype string and return the corresponding Polars dtype.

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

        # Parse Datetime/Duration with time unit: 'datetime[us]', 'duration[ms]', etc.
        prefix, unit = _split_time_unit(dtype_str)
        if unit is not None:  # pragma: no cover
            if prefix == "datetime":
                return pl.Datetime(time_unit=unit)
            return pl.Duration(time_unit=unit)

        # Parse List types: list[inner_type]
        if dtype_str.startswith("list[") and dtype_str.endswith("]"):
            inner_str = dtype_str[5:-1]  # Extract content between list[ and ]
            inner_dtype = self._parse_polars_dtype(inner_str)
            return pl.List(inner_dtype)

        # Parse Array types: array[inner_type, width]
        if dtype_str.startswith("array[") and dtype_str.endswith("]"):
            content = dtype_str[6:-1]  # Extract content between array[ and ]
            parts = content.rsplit(",", 1)  # Split on last comma

            if len(parts) != 2:
                raise ValueError(f"Array type must specify width: 'array[type, width]', got '{dtype_str}'")

            inner_str = parts[0].strip()
            try:
                width = int(parts[1].strip())
            except ValueError:
                raise ValueError(f"Array width must be an integer, got '{parts[1].strip()}'")

            inner_dtype = self._parse_polars_dtype(inner_str)
            return pl.Array(inner_dtype, width)

        # Parse Struct types: struct[{field1: type1, field2: type2}]
        if dtype_str.startswith("struct[") and dtype_str.endswith("]"):
            content = dtype_str[7:-1].strip()  # Extract content between struct[ and ]

            # Remove outer braces if present
            if content.startswith("{") and content.endswith("}"):
                content = content[1:-1].strip()

            # Parse field definitions
            fields = []
            if content:  # Not an empty struct
                # Split on commas not inside nested brackets
                field_strs = self._split_struct_fields(content)

                for field_str_orig in field_strs:
                    field_str = field_str_orig.strip()
                    if ":" not in field_str:
                        raise ValueError(f"Struct field must have format 'name: type', got '{field_str}'")

                    name, type_str = field_str.split(":", 1)
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
            if char in "[{":
                depth += 1
                current.append(char)
            elif char in "]}":
                depth -= 1
                current.append(char)
            elif char == "," and depth == 0:
                # Top-level comma - this is a field separator
                fields.append("".join(current))
                current = []
            else:
                current.append(char)

        if current:  # last field
            fields.append("".join(current))

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
