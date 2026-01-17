"""Check which dataframe libraries are installed."""

from typing import NamedTuple

__all__ = [
    "HAS_PANDAS",
    "HAS_POLARS",
    "HAS_SPARK",
    "BackendInfo",
    "get_backend_info",
    "require_backend",
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


def get_backend_info(backend: str) -> BackendInfo:
    """Get detailed information about a backend.

    Args:
        backend (str):
            One of 'pandas', 'polars', 'spark'

    Returns:
        BackendInfo with availability and version details
    """
    info_map = {
        "pandas": _PANDAS_INFO,
        "polars": _POLARS_INFO,
        "spark": _PYSPARK_INFO,
    }
    if backend not in info_map:
        raise ValueError(f"Unknown backend: {backend}. Must be one of {list(info_map.keys())}")
    return info_map[backend]


def require_backend(backend: str, *, check_version: bool = True) -> None:
    """Raise ImportError if backend is not available or version is too old.

    Args:
        backend (str):
            One of 'pandas', 'polars', 'spark'
        check_version (bool):
            If True, also check minimum version requirement

    Raises:
        ImportError: If backend not available or version too old
    """
    info = get_backend_info(backend)

    if not info.available:
        raise ImportError(f"{backend} is not installed. Install it with: pip install {backend}")

    if check_version and not info.meets_minimum:
        min_versions = {
            "pandas": MIN_PANDAS,
            "polars": MIN_POLARS,
            "spark": MIN_PYSPARK,
        }
        raise ImportError(
            f"{backend} version {info.version} is installed, but Nebula requires "
            f">= {min_versions[backend]}. Upgrade with: pip install --upgrade {backend}"
        )
