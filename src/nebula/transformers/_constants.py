import narwhals as nw

from nebula.backend_util import HAS_POLARS

__all__ = [
    "NW_TYPES",
    "PL_TYPES",
    "PANDAS_NULLABLE_INTEGERS",
]

NW_TYPES: dict[str, nw.dtypes.DType] = {
    'int': nw.Int64, 'integer': nw.Int64, 'long': nw.Int64,
    'int8': nw.Int8, 'int16': nw.Int16,
    'int32': nw.Int32, 'int64': nw.Int64,
    'uint8': nw.UInt8, 'uint16': nw.UInt16,
    'uint32': nw.UInt32, 'uint64': nw.UInt64,
    'float': nw.Float64, 'double': nw.Float64,
    'float32': nw.Float32, 'float64': nw.Float64,
    'str': nw.String, 'string': nw.String, 'utf8': nw.String,
    'bool': nw.Boolean, 'boolean': nw.Boolean,
    'date': nw.Date,
    'datetime': nw.Datetime,
    'duration': nw.Duration,
    'timedelta': nw.Duration,
}

PANDAS_NULLABLE_INTEGERS: dict[str, str] = {
    'int': 'Int64', 'integer': 'Int64',
    'int8': 'Int8', 'int16': 'Int16', 'int32': 'Int32', 'int64': 'Int64',
    'uint8': 'UInt8', 'uint16': 'UInt16', 'uint32': 'UInt32', 'uint64': 'UInt64',
}

if HAS_POLARS:
    import polars as pl

    PL_TYPES: dict[str, pl.datatypes.DataType] | None = {
        'int': pl.Int64, 'integer': pl.Int64,
        'int8': pl.Int8, 'int16': pl.Int16,
        'int32': pl.Int32, 'int64': pl.Int64,
        'uint8': pl.UInt8, 'uint16': pl.UInt16,
        'uint32': pl.UInt32, 'uint64': pl.UInt64,
        'float': pl.Float64, 'float32': pl.Float32, 'float64': pl.Float64,
        'str': pl.String, 'string': pl.String, 'utf8': pl.String,
        'bool': pl.Boolean, 'boolean': pl.Boolean,
        'date': pl.Date,
        'datetime': pl.Datetime,
        'duration': pl.Duration, 'timedelta': pl.Duration,
        'time': pl.Time,
    }
else:  # pragma: no cover
    PL_TYPES = None
