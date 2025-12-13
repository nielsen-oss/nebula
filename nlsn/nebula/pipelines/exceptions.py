"""Pipeline exceptions module."""

from typing import NoReturn

from nlsn.nebula.backend_util import HAS_SPARK

__all__ = [
    "PipelineError",
    "raise_pipeline_error",
]


class PipelineError(Exception):
    """Base exception for Nebula pipeline errors."""
    pass


def _try_enhance_spark_exception(e: Exception, msg: str) -> bool:
    """Attempt to enhance Spark-specific exceptions in-place.

    Returns True if the exception was enhanced, False otherwise.
    """
    if not HAS_SPARK:
        return False

    # Import lazily to avoid issues if Spark internals change
    try:
        from py4j.protocol import Py4JJavaError
        if isinstance(e, Py4JJavaError) and hasattr(e, 'errmsg'):
            e.errmsg = f"{msg}\n{e.errmsg}"
            return True
    except (ImportError, AttributeError):
        pass

    try:
        from pyspark.sql.utils import CapturedException
        if isinstance(e, CapturedException) and hasattr(e, 'desc'):
            e.desc = f"{msg}\n{e.desc}"
            return True
    except (ImportError, AttributeError):
        pass

    return False


def raise_pipeline_error(e: Exception, msg: str) -> NoReturn:
    """Handle pipeline errors with enhanced context.

    Gracefully handles Spark-specific exceptions when available,
    and falls back to standard Python exception handling otherwise.

    Args:
        e: The original exception that was raised
        msg: Custom contextual message to prepend

    Raises:
        The original exception type with enhanced message
    """
    # Try Spark-specific enhancement first
    if _try_enhance_spark_exception(e, msg):
        raise e

    # Fallback: wrap in same exception type with enhanced message
    raise type(e)(f"{msg}\n{e}") from e
