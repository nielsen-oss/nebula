"""Pipeline exceptions module."""

from nlsn.nebula.backend_util import HAS_SPARK

if HAS_SPARK:
    from py4j.protocol import Py4JJavaError
    from pyspark.sql.utils import CapturedException

__all__ = [
    "PipelineError",
    "raise_pipeline_error",
]


class PipelineError(Exception):
    """Base exception for Nebula pipeline errors."""
    pass


def raise_pipeline_error(e: Exception, msg: str) -> None:
    """Handle pipeline errors with enhanced context.

    Gracefully handles Spark-specific exceptions when available,
    and falls back to standard Python exception handling otherwise.

    Args:
        e: The original exception that was raised
        msg: Custom contextual message to prepend

    Raises:
        The original exception type with enhanced message
    """
    if HAS_SPARK and isinstance(e, Py4JJavaError):
        e.errmsg = f"{msg}\n{e.errmsg}"
        raise e

    if HAS_SPARK and isinstance(e, CapturedException):
        e.desc = f"{msg}\n{e.desc}"
        raise e

    # Handle all other exceptions
    raise type(e)(f"{msg}\n{e}") from e
