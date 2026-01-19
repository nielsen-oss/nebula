"""Check which dataframe libraries are installed."""

from typing import NamedTuple

__all__ = [
    "HAS_PANDAS",
    "HAS_POLARS",
    "HAS_SPARK",
    "BackendInfo",
]

# Minimum supported versions
MIN_PANDAS = "1.2.5"
MIN_POLARS = "1.34.0"
MIN_PYSPARK = "3.5.0"


class BackendInfo(NamedTuple):
    """Information about an installed backend."""

    available: bool
    version: str | None
    meets_minimum: bool


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse version string to tuple for comparison."""
    return tuple(int(x) for x in version_str.split(".")[:3])


def _check_backend(module_name: str, min_version: str) -> BackendInfo:
    """Check if a backend is available and meets minimum version."""
    try:
        mod = __import__(module_name)
        version = mod.__version__
        meets_min = _parse_version(version) >= _parse_version(min_version)
        return BackendInfo(available=True, version=version, meets_minimum=meets_min)
    except ImportError:  # pragma: no cover
        return BackendInfo(available=False, version=None, meets_minimum=False)


# Check backends on module import
_PANDAS_INFO = _check_backend("pandas", MIN_PANDAS)
_POLARS_INFO = _check_backend("polars", MIN_POLARS)
_PYSPARK_INFO = _check_backend("pyspark", MIN_PYSPARK)

# Simple flags for backward compatibility
HAS_PANDAS: bool = _PANDAS_INFO.available
HAS_POLARS: bool = _POLARS_INFO.available
HAS_SPARK: bool = _PYSPARK_INFO.available
