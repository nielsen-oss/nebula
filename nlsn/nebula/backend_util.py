"""Check which dataframe libraries are installed."""

# pylint: disable=unused-import

__all__ = [
    "HAS_PANDAS",
    "HAS_POLARS",
    "HAS_SPARK",
]

HAS_PANDAS: bool
HAS_POLARS: bool
HAS_SPARK: bool

try:
    import pandas

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import polars

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import pyspark

    HAS_SPARK = True
except ImportError:
    HAS_SPARK = False
