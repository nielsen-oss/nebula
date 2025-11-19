"""Check which dataframe libraries are installed."""

__all__ = ["HAS_PANDAS", "HAS_POLARS", "HAS_SPARK"]

HAS_PANDAS: bool
HAS_POLARS: bool
HAS_SPARK: bool

try:
    import pandas  # noqa: F401

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import polars  # noqa: F401

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import pyspark  # noqa: F401

    HAS_SPARK = True
except ImportError:
    HAS_SPARK = False
